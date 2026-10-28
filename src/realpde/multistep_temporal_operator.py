from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MultiStepTemporalConfig:
    """Fixed C280 stride-1/4/5 temporal evolution configuration."""

    in_steps: int = 20
    out_steps: int = 20
    width: int = 24
    strides: tuple[int, ...] = (1, 4, 5)
    encoder_depth: int = 2
    dropout: float = 0.02
    maximum_normalized_correction: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["strides"] = list(self.strides)
        return values


class SpatialResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        channel_groups = 4 if width % 4 == 0 else 1
        self.norm = nn.GroupNorm(groups, width)
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.channel = nn.Sequential(
            nn.Conv2d(width, width, 1, groups=channel_groups),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(width, width, 1, groups=channel_groups),
        )
        self.scale = nn.Parameter(torch.full((1, width, 1, 1), 1e-2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.scale * self.channel(self.depthwise(self.norm(value)))


class StrideEvolutionExpert(nn.Module):
    """One independently parameterized latent advancement map."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        channel_groups = 4 if width % 4 == 0 else 1
        self.norm = nn.GroupNorm(groups, width)
        self.spatial = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.gates = nn.Conv2d(2 * width + 1, 2 * width, 1, groups=1)
        self.candidate = nn.Sequential(
            nn.Conv2d(2 * width + 1, width, 1, groups=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(width, width, 1, groups=channel_groups),
        )
        self.update_scale = nn.Parameter(torch.full((1, width, 1, 1), 0.1))

    def forward(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        mixed = self.spatial(self.norm(state))
        source = torch.cat((mixed, context, phase), dim=1)
        reset, update = self.gates(source).chunk(2, dim=1)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update)
        proposal_source = torch.cat((reset * mixed, context, phase), dim=1)
        proposal = torch.tanh(self.candidate(proposal_source))
        return state + self.update_scale * update * (proposal - state)


class MultiStepTemporalFlowOperator(nn.Module):
    """Parallel stride evolution with a deployment-fixed horizon router."""

    def __init__(self, config: MultiStepTemporalConfig | None = None) -> None:
        super().__init__()
        self.config = config or MultiStepTemporalConfig()
        if self.config.in_steps != 20 or self.config.out_steps != 20:
            raise ValueError("C280 is fixed to the official 20-to-20 contract")
        if tuple(self.config.strides) != (1, 4, 5):
            raise ValueError("C280 has a preregistered stride set of (1, 4, 5)")
        width = self.config.width
        groups = 8 if width % 8 == 0 else 1
        # All twenty normalized u/v frames plus x/y coordinates and two recent
        # increments.  The latent grid is 8x16 to keep the multi-path evolution
        # within the one-pass runtime gate.
        self.lift = nn.Conv2d(46, width, 1)
        self.encoder = nn.ModuleList(
            SpatialResidualBlock(width, self.config.dropout)
            for _ in range(self.config.encoder_depth)
        )
        self.experts = nn.ModuleList(
            StrideEvolutionExpert(width, self.config.dropout)
            for _ in self.config.strides
        )
        route_mask = torch.zeros(self.config.out_steps, len(self.config.strides), dtype=torch.bool)
        for horizon in range(1, self.config.out_steps + 1):
            for index, stride in enumerate(self.config.strides):
                route_mask[horizon - 1, index] = horizon % stride == 0
        self.register_buffer("route_mask", route_mask, persistent=True)
        self.route_logits = nn.Parameter(torch.zeros_like(route_mask, dtype=torch.float32))
        self.decode_norm = nn.GroupNorm(groups, width)
        self.point_head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.GELU(),
            nn.Conv2d(width, 2, 1),
        )
        self.width_head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.GELU(),
            nn.Conv2d(width, 2, 1),
        )
        self.trend_scale = nn.Parameter(torch.zeros(self.config.out_steps, 2))
        nn.init.zeros_(self.point_head[-1].weight)
        nn.init.zeros_(self.point_head[-1].bias)
        nn.init.zeros_(self.width_head[-1].weight)
        nn.init.constant_(self.width_head[-1].bias, -2.0)

    @staticmethod
    def _normalize(flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uv = flow[..., :2].float()
        mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        scale = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        return (uv - mean) / scale, mean, scale

    @staticmethod
    def _spatial_coordinates(
        batch: int,
        height: int,
        width: int,
        source: torch.Tensor,
    ) -> torch.Tensor:
        yy = torch.linspace(-1.0, 1.0, height, device=source.device, dtype=source.dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=source.device, dtype=source.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=0)[None].expand(batch, -1, -1, -1)

    def route_weights(self) -> torch.Tensor:
        masked = self.route_logits.masked_fill(~self.route_mask, -torch.inf)
        return torch.softmax(masked, dim=-1)

    def _encode(self, normalized: torch.Tensor) -> torch.Tensor:
        batch, _steps, height, width, _channels = normalized.shape
        flattened = normalized.permute(0, 1, 4, 2, 3).reshape(batch, 40, height, width)
        coordinates = self._spatial_coordinates(batch, height, width, normalized)
        delta1 = (normalized[:, -1] - normalized[:, -2]).permute(0, 3, 1, 2)
        delta4 = ((normalized[:, -1] - normalized[:, -5]) / 4.0).permute(0, 3, 1, 2)
        source = torch.cat((flattened, coordinates, delta1, delta4), dim=1)
        source = F.avg_pool2d(source, kernel_size=4, stride=4)
        context = self.lift(source)
        for block in self.encoder:
            context = block(context)
        return context

    def _evolve(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[int, torch.Tensor]]]:
        batch = context.shape[0]
        paths: list[dict[int, torch.Tensor]] = []
        for stride, expert in zip(self.config.strides, self.experts, strict=True):
            state = context
            states: dict[int, torch.Tensor] = {}
            for horizon in range(stride, self.config.out_steps + 1, stride):
                phase_value = horizon / self.config.out_steps
                phase = context.new_full((batch, 1, *context.shape[-2:]), phase_value)
                state = expert(state, context, phase)
                states[horizon] = state
            paths.append(states)
        weights = self.route_weights()
        fused = []
        for horizon in range(1, self.config.out_steps + 1):
            value = torch.zeros_like(context)
            for index, states in enumerate(paths):
                if horizon in states:
                    value = value + weights[horizon - 1, index] * states[horizon]
            fused.append(value)
        return torch.stack(fused, dim=2), weights, paths

    def forward(
        self,
        flow: torch.Tensor,
        *,
        return_interval: bool = False,
        return_routes: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (self.config.in_steps, 32, 64, 3)
        if flow.ndim != 5 or tuple(flow.shape[1:]) != expected:
            raise ValueError(f"expected [B,{expected}], got {tuple(flow.shape)}")
        normalized, mean, scale = self._normalize(flow)
        context = self._encode(normalized)
        future, routes, _paths = self._evolve(context)
        latent_height, latent_width = future.shape[-2:]
        decoded = self.decode_norm(future).permute(0, 2, 1, 3, 4).reshape(
            -1, self.config.width, latent_height, latent_width
        )
        raw_point = self.point_head(decoded)
        raw_width = self.width_head(decoded)
        raw_point = F.interpolate(
            raw_point, size=(flow.shape[2], flow.shape[3]), mode="bilinear", align_corners=False
        ).reshape(flow.shape[0], self.config.out_steps, 2, flow.shape[2], flow.shape[3])
        raw_width = F.interpolate(
            raw_width, size=(flow.shape[2], flow.shape[3]), mode="bilinear", align_corners=False
        ).reshape(flow.shape[0], self.config.out_steps, 2, flow.shape[2], flow.shape[3])
        raw_point = raw_point.permute(0, 1, 3, 4, 2)
        raw_width = raw_width.permute(0, 1, 3, 4, 2)
        correction = self.config.maximum_normalized_correction * torch.tanh(raw_point)
        recent = normalized[:, -1] - normalized[:, -5]
        base = normalized[:, -1:, ...].expand(-1, self.config.out_steps, -1, -1, -1)
        base = base + self.trend_scale[None, :, None, None, :] * recent[:, None]
        point_uv = (base + correction) * scale + mean
        half_uv = F.softplus(raw_width) * scale * 0.5
        zero = torch.zeros_like(point_uv[..., :1])
        point = torch.cat((point_uv, zero), dim=-1).to(flow.dtype)
        half = torch.cat((half_uv, zero), dim=-1).to(flow.dtype)
        if return_interval and return_routes:
            return point, half, routes
        if return_interval:
            return point, half
        if return_routes:
            return point, routes
        return point

    def predict(
        self,
        input_array: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del metadata
        return self.forward(input_array)
