from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BidirectionalStateSpaceConfig:
    """Fixed configuration for the C268 window-local state-space operator."""

    in_steps: int = 20
    out_steps: int = 20
    width: int = 64
    observed_depth: int = 4
    future_depth: int = 2
    expansion: int = 2
    dropout: float = 0.02
    maximum_normalized_correction: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DiagonalStateScan(nn.Module):
    """Stable channel-wise exponential state scan on an explicit time axis."""

    def __init__(self, width: int, *, bidirectional: bool) -> None:
        super().__init__()
        self.width = int(width)
        self.bidirectional = bool(bidirectional)
        self.input_projection = nn.Conv3d(width, width, 1)
        self.forward_decay_logit = nn.Parameter(torch.zeros(width))
        if self.bidirectional:
            self.backward_decay_logit = nn.Parameter(torch.zeros(width))
            self.merge = nn.Conv3d(2 * width, width, 1)
        else:
            self.register_parameter("backward_decay_logit", None)
            self.merge = nn.Identity()

    @staticmethod
    def _decay(logit: torch.Tensor) -> torch.Tensor:
        # Bounded away from both zero and one: stable but capable of long memory.
        return 0.70 + 0.295 * torch.sigmoid(logit.float())

    @staticmethod
    def _scan(value: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
        batch, channels, steps, height, width = value.shape
        sequence = value.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width, channels, steps
        )
        exponent = torch.arange(
            steps - 1, -1, -1, device=value.device, dtype=torch.float32
        )
        kernel = decay[:, None].pow(exponent[None]).to(sequence.dtype)
        scanned = F.conv1d(
            F.pad(sequence, (steps - 1, 0)),
            kernel[:, None],
            groups=channels,
        )
        chronological = torch.arange(
            steps, device=value.device, dtype=torch.float32
        )
        denominator = torch.cumsum(
            decay[:, None].pow(chronological[None]), dim=1
        ).clamp_min(1e-4)
        scanned = scanned / denominator[None].to(scanned.dtype)
        return scanned.reshape(batch, height, width, channels, steps).permute(
            0, 3, 4, 1, 2
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(value)
        forward = self._scan(projected, self._decay(self.forward_decay_logit))
        if not self.bidirectional:
            return forward
        backward = self._scan(
            projected.flip(2), self._decay(self.backward_decay_logit)
        ).flip(2)
        return self.merge(torch.cat((forward, backward), dim=1))


class StateSpaceTransportBlock(nn.Module):
    """Diagonal temporal state evolution plus local spatial transport."""

    def __init__(
        self,
        width: int,
        expansion: int,
        dropout: float,
        *,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        hidden = width * expansion
        self.norm1 = nn.GroupNorm(groups, width)
        self.state_scan = DiagonalStateScan(width, bidirectional=bidirectional)
        self.spatial = nn.Conv3d(
            width,
            width,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=width,
        )
        self.channel = nn.Conv3d(width, width, 1)
        self.norm2 = nn.GroupNorm(groups, width)
        self.ffn = nn.Sequential(
            nn.Conv3d(width, 2 * hidden, 1),
            nn.GLU(dim=1),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(hidden, width, 1),
        )
        self.state_scale = nn.Parameter(torch.full((1, width, 1, 1, 1), 1e-2))
        self.ffn_scale = nn.Parameter(torch.full((1, width, 1, 1, 1), 1e-2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(value)
        state = self.state_scan(normalized)
        spatial = self.channel(self.spatial(normalized))
        value = value + self.state_scale * (state + spatial)
        return value + self.ffn_scale * self.ffn(self.norm2(value))


class BidirectionalStateSpaceFlowOperator(nn.Module):
    """Scratch 20-to-20 point/interval operator with stable latent state evolution."""

    def __init__(self, config: BidirectionalStateSpaceConfig | None = None) -> None:
        super().__init__()
        self.config = config or BidirectionalStateSpaceConfig()
        if self.config.in_steps != 20 or self.config.out_steps != 20:
            raise ValueError("C268 is fixed to the official 20-to-20 contract")
        width = self.config.width
        # Normalized u/v/p, observed time, and x/y coordinates.
        self.lift = nn.Conv3d(
            6,
            width,
            kernel_size=(1, 5, 5),
            stride=(1, 2, 2),
            padding=(0, 2, 2),
        )
        self.observed_blocks = nn.ModuleList(
            StateSpaceTransportBlock(
                width,
                self.config.expansion,
                self.config.dropout,
                bidirectional=True,
            )
            for _ in range(self.config.observed_depth)
        )
        self.seed_projection = nn.Conv2d(2 * width, width, 1)
        self.future_decay_logit = nn.Parameter(torch.zeros(width))
        self.future_forcing = nn.Parameter(
            torch.empty(1, width, self.config.out_steps, 1, 1)
        )
        nn.init.normal_(self.future_forcing, std=0.02)
        self.future_blocks = nn.ModuleList(
            StateSpaceTransportBlock(
                width,
                self.config.expansion,
                self.config.dropout,
                bidirectional=False,
            )
            for _ in range(self.config.future_depth)
        )
        groups = 8 if width % 8 == 0 else 1
        self.decode_norm = nn.GroupNorm(groups, width)
        self.point_head = nn.Sequential(
            nn.Conv3d(width, width, (1, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(width, 2, 1),
        )
        self.width_head = nn.Sequential(
            nn.Conv3d(width, width // 2, (1, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(width // 2, 2, 1),
        )
        self.trend_scale = nn.Parameter(torch.zeros(self.config.out_steps, 2))
        self.interval_temperature = nn.Parameter(
            torch.tensor(1.0), requires_grad=False
        )
        nn.init.zeros_(self.point_head[-1].weight)
        nn.init.zeros_(self.point_head[-1].bias)
        nn.init.zeros_(self.width_head[-1].weight)
        nn.init.constant_(self.width_head[-1].bias, -2.0)

    @staticmethod
    def _coordinates(source: torch.Tensor) -> torch.Tensor:
        batch, _channels, steps, height, width = source.shape
        time = torch.linspace(
            -1.0, 0.0, steps, device=source.device, dtype=source.dtype
        )[None, None, :, None, None]
        time = time.expand(batch, 1, steps, height, width)
        yy = torch.linspace(-1.0, 1.0, height, device=source.device, dtype=source.dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=source.device, dtype=source.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        spatial = torch.stack((grid_x, grid_y), dim=0)[None, :, None]
        spatial = spatial.expand(batch, 2, steps, height, width)
        return torch.cat((time, spatial), dim=1)

    @staticmethod
    def _normalize(
        flow: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uv = flow[..., :2].float()
        uv_mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        uv_std = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized_uv = (uv - uv_mean) / uv_std
        pressure = flow[..., 2:3].float()
        pressure_mean = pressure.mean(dim=(1, 2, 3), keepdim=True)
        pressure_std = pressure.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized_pressure = (pressure - pressure_mean) / pressure_std
        return torch.cat((normalized_uv, normalized_pressure), dim=-1), uv_mean, uv_std

    def forward(
        self,
        flow: torch.Tensor,
        *,
        return_interval: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        expected = (self.config.in_steps, 32, 64, 3)
        if flow.ndim != 5 or tuple(flow.shape[1:]) != expected:
            raise ValueError(f"expected [B,{expected}], got {tuple(flow.shape)}")
        normalized, uv_mean, uv_std = self._normalize(flow)
        source = normalized.permute(0, 4, 1, 2, 3).contiguous()
        source = torch.cat((source, self._coordinates(source)), dim=1)
        observed = self.lift(source)
        for block in self.observed_blocks:
            observed = block(observed)

        seed = self.seed_projection(
            torch.cat((observed[:, :, -1], observed.mean(dim=2)), dim=1)
        )
        decay = 0.70 + 0.295 * torch.sigmoid(self.future_decay_logit.float())
        horizon = torch.arange(
            1,
            self.config.out_steps + 1,
            device=flow.device,
            dtype=torch.float32,
        )
        powers = decay[:, None].pow(horizon[None]).to(seed.dtype)
        future = seed[:, :, None] * powers[None, :, :, None, None]
        future = future + self.future_forcing.to(future.dtype)
        for block in self.future_blocks:
            future = block(future)
        future = F.interpolate(
            future,
            size=(self.config.out_steps, flow.shape[2], flow.shape[3]),
            mode="trilinear",
            align_corners=False,
        )
        decoded = self.decode_norm(future)
        raw_point = self.point_head(decoded).permute(0, 2, 3, 4, 1)
        correction = self.config.maximum_normalized_correction * torch.tanh(raw_point)
        recent = (
            normalized[:, -1:, ..., :2] - normalized[:, -5:-4, ..., :2]
        ) / 4.0
        trend = recent * self.trend_scale[None, :, None, None, :]
        normalized_point = normalized[:, -1:, ..., :2] + trend + correction
        point_uv = normalized_point * uv_std + uv_mean
        point = torch.cat((point_uv, torch.zeros_like(point_uv[..., :1])), dim=-1)
        if not return_interval:
            return point

        raw_width = self.width_head(decoded).permute(0, 2, 3, 4, 1)
        half_width_uv = (
            F.softplus(raw_width)
            * uv_std
            * self.interval_temperature.clamp_min(1e-3)
        )
        half_width = torch.cat(
            (half_width_uv, torch.zeros_like(half_width_uv[..., :1])), dim=-1
        )
        return point, half_width

    def predict(
        self,
        flow: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del metadata
        return self(flow)

