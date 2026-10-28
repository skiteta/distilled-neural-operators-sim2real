from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DifferentialIntegralConfig:
    """Fixed C301 DIO-lite reachability configuration."""

    in_steps: int = 20
    out_steps: int = 20
    width: int = 96
    depth: int = 6
    heads: int = 4
    expansion: int = 2
    dropout: float = 0.02
    use_pressure: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FixedDerivativeBank(nn.Module):
    """Five fixed finite-difference stencils with a trainable channel mixer."""

    operators = ("dx", "dy", "dxx", "dyy", "dxy")

    def __init__(self, width: int) -> None:
        super().__init__()
        stencils = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]],
                [[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.25, 0.0, -0.25], [0.0, 0.0, 0.0], [-0.25, 0.0, 0.25]],
            ],
            dtype=torch.float32,
        )
        if not torch.equal(stencils.sum(dim=(-2, -1)), torch.zeros(5)):
            raise RuntimeError("derivative stencils must annihilate constants")
        self.width = width
        self.register_buffer("stencils", stencils[:, None], persistent=True)
        self.mix = nn.Conv2d(width * len(self.operators), width, 1, bias=False)

    def raw_response(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4 or hidden.shape[1] != self.width:
            raise ValueError(f"expected [B,{self.width},H,W], got {tuple(hidden.shape)}")
        padded = F.pad(hidden, (1, 1, 1, 1), mode="replicate")
        kernels = self.stencils.repeat(self.width, 1, 1, 1).to(hidden.dtype)
        response = F.conv2d(padded, kernels, groups=self.width)
        batch, _, height, width = response.shape
        return response.reshape(batch, self.width, len(self.operators), height, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        raw = self.raw_response(hidden)
        batch, channels, operators, height, width = raw.shape
        return self.mix(raw.reshape(batch, channels * operators, height, width))


class AxialIntegralMix(nn.Module):
    """Content-adaptive low-rank global kernel over learned x/y subfunctions."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.width = width
        self.project = nn.Conv2d(width, width, 1, bias=False)
        self.x_norm = nn.LayerNorm(width)
        self.y_norm = nn.LayerNorm(width)
        self.x_integral = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.y_integral = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.fuse = nn.Conv2d(width * 2, width, 1, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        latent = self.project(hidden)
        batch, channels, height, width = latent.shape
        x_tokens = self.x_norm(latent.mean(dim=2).transpose(1, 2))
        y_tokens = self.y_norm(latent.mean(dim=3).transpose(1, 2))
        x_update, _ = self.x_integral(
            x_tokens, x_tokens, x_tokens, need_weights=False
        )
        y_update, _ = self.y_integral(
            y_tokens, y_tokens, y_tokens, need_weights=False
        )
        x_field = x_update.transpose(1, 2).unsqueeze(2).expand(
            batch, channels, height, width
        )
        y_field = y_update.transpose(1, 2).unsqueeze(3).expand(
            batch, channels, height, width
        )
        return self.fuse(torch.cat((x_field, y_field), dim=1))


class DifferentialIntegralBlock(nn.Module):
    """Parallel constrained differential and content-adaptive integral update."""

    def __init__(
        self,
        width: int,
        heads: int,
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        hidden = width * expansion
        self.norm1 = nn.GroupNorm(groups, width)
        self.differential = FixedDerivativeBank(width)
        self.integral = AxialIntegralMix(width, heads, dropout)
        self.channel = nn.Conv2d(width, width, 1)
        self.norm2 = nn.GroupNorm(groups, width)
        self.ffn = nn.Sequential(
            nn.Conv2d(width, hidden * 2, 1),
            nn.GLU(dim=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, width, 1),
        )
        self.operator_scale = nn.Parameter(torch.full((1, width, 1, 1), 1e-2))
        self.ffn_scale = nn.Parameter(torch.full((1, width, 1, 1), 1e-2))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(hidden)
        update = (
            self.differential(normalized)
            + self.integral(normalized)
            + self.channel(normalized)
        )
        hidden = hidden + self.operator_scale * update
        return hidden + self.ffn_scale * self.ffn(self.norm2(hidden))


class DifferentialIntegralFlowOperator(nn.Module):
    """Mechanism-compatible DIO-lite direct 20-to-20 flow operator."""

    def __init__(self, config: DifferentialIntegralConfig | None = None) -> None:
        super().__init__()
        self.config = config or DifferentialIntegralConfig()
        if self.config.in_steps != 20 or self.config.out_steps != 20:
            raise ValueError("C301 is fixed to the official 20-to-20 contract")
        if self.config.width % self.config.heads:
            raise ValueError("width must be divisible by heads")
        modalities = 3 if self.config.use_pressure else 2
        descriptor_channels = 4 * modalities
        in_channels = self.config.in_steps * modalities + 2 + descriptor_channels
        self.lift = nn.Conv2d(in_channels, self.config.width, 1)
        self.blocks = nn.ModuleList(
            DifferentialIntegralBlock(
                self.config.width,
                self.config.heads,
                self.config.expansion,
                self.config.dropout,
            )
            for _ in range(self.config.depth)
        )
        groups = 8 if self.config.width % 8 == 0 else 1
        self.final_norm = nn.GroupNorm(groups, self.config.width)
        self.velocity_head = nn.Sequential(
            nn.Conv2d(self.config.width, self.config.width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.config.width, self.config.out_steps * 2, 1),
        )
        self.width_head = nn.Sequential(
            nn.Conv2d(self.config.width, self.config.width // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.config.width // 2, self.config.out_steps * 2, 1),
        )
        self.trend_scale = nn.Parameter(torch.zeros(self.config.out_steps, 2))
        self.interval_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        nn.init.normal_(self.velocity_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.velocity_head[-1].bias)
        nn.init.normal_(self.width_head[-1].weight, std=1e-3)
        nn.init.constant_(self.width_head[-1].bias, -2.5)

    @staticmethod
    def _coordinates(
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).expand(
            batch, -1, -1, -1
        )

    def _features(
        self, flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (self.config.in_steps, 32, 64, 3)
        if flow.ndim != 5 or tuple(flow.shape[1:]) != expected:
            raise ValueError(f"expected [B,{expected}], got {tuple(flow.shape)}")
        uv = flow[..., :2]
        uv_mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        uv_std = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized_uv = (uv - uv_mean) / uv_std
        selected = uv
        if self.config.use_pressure:
            pressure = flow[..., 2:3]
            pressure_mean = pressure.mean(dim=(1, 2, 3), keepdim=True)
            pressure_std = pressure.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
            normalized = torch.cat(
                (normalized_uv, (pressure - pressure_mean) / pressure_std), dim=-1
            )
            selected = flow
        else:
            normalized = normalized_uv
        batch, _, height, width, _ = flow.shape
        flat = normalized.permute(0, 1, 4, 2, 3).reshape(
            batch, -1, height, width
        )
        descriptors = torch.cat(
            (
                selected.mean(dim=(1, 2, 3)),
                selected.std(dim=(1, 2, 3)),
                selected[:, -1].mean(dim=(1, 2)),
                selected[:, -1].std(dim=(1, 2)),
            ),
            dim=-1,
        )
        descriptors = descriptors[:, :, None, None].expand(
            batch, -1, height, width
        )
        coords = self._coordinates(batch, height, width, flow.device, flow.dtype)
        return torch.cat((flat, coords, descriptors), dim=1), uv_mean, uv_std

    def forward(
        self, flow: torch.Tensor, *, return_interval: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features, uv_mean, uv_std = self._features(flow)
        hidden = self.lift(features)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        batch, _, height, width = hidden.shape
        correction = self.velocity_head(hidden).reshape(
            batch, self.config.out_steps, 2, height, width
        ).permute(0, 1, 3, 4, 2)
        base = (flow[:, -1:, ..., :2] - uv_mean) / uv_std
        recent = (flow[:, -1:, ..., :2] - flow[:, -5:-4, ..., :2]) / (
            4.0 * uv_std
        )
        trend = recent * self.trend_scale[None, :, None, None, :]
        prediction_uv = (base + trend + correction) * uv_std + uv_mean
        prediction = torch.cat(
            (prediction_uv, torch.zeros_like(prediction_uv[..., :1])), dim=-1
        )
        if not return_interval:
            return prediction
        raw_width = self.width_head(hidden).reshape(
            batch, self.config.out_steps, 2, height, width
        ).permute(0, 1, 3, 4, 2)
        half_uv = (
            F.softplus(raw_width)
            * uv_std
            * self.interval_temperature.clamp_min(1e-3)
        )
        half = torch.cat((half_uv, torch.zeros_like(half_uv[..., :1])), dim=-1)
        return prediction, half

    def predict(
        self,
        input_array: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del metadata
        return self.forward(input_array)
