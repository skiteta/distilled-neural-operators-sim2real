from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class KhatriRaoConfig:
    """Frozen C371 competition-sized Khatri--Rao operator configuration."""

    in_steps: int = 20
    out_steps: int = 20
    width: int = 32
    depth: int = 3
    rank: int = 2
    spatial_bottleneck: int = 4
    expansion: int = 2
    dropout: float = 0.02
    use_pressure: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CoordinatePairKernel(nn.Module):
    """Learned signed kernel between two one-dimensional coordinate grids."""

    def __init__(
        self,
        input_coordinates: torch.Tensor,
        output_coordinates: torch.Tensor,
        rank: int,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.register_buffer(
            "input_coordinates", input_coordinates.float().clone(), persistent=True
        )
        self.register_buffer(
            "output_coordinates", output_coordinates.float().clone(), persistent=True
        )
        hidden = max(16, 4 * rank)
        self.network = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4 * rank),
        )
        self.dc_logit = nn.Parameter(torch.full((rank,), 1.5))

    def forward(self, *, dtype: torch.dtype) -> torch.Tensor:
        output = self.output_coordinates[:, None]
        source = self.input_coordinates[None, :]
        features = torch.stack(
            (
                output.expand(-1, source.shape[1]),
                source.expand(output.shape[0], -1),
                output - source,
                output * source,
            ),
            dim=-1,
        )
        raw = self.network(features).reshape(
            output.shape[0], source.shape[1], self.rank, 4
        )
        positive = raw[..., 0].permute(2, 0, 1).softmax(dim=-1)
        negative = raw[..., 1].permute(2, 0, 1).softmax(dim=-1)
        carrier = raw[..., 2].permute(2, 0, 1).softmax(dim=-1)
        amplitude = torch.tanh(raw[..., 3].permute(2, 0, 1))
        signed = positive - negative
        dc = torch.sigmoid(self.dc_logit)[:, None, None] * carrier
        return (signed + dc + 0.1 * amplitude / source.shape[1]).to(dtype=dtype)


class KhatriRaoIntegral(nn.Module):
    """Separable non-stationary integral over time, y and x axes."""

    def __init__(
        self,
        *,
        input_time: torch.Tensor,
        output_time: torch.Tensor,
        height: int,
        width: int,
        rank: int,
        channels: int,
    ) -> None:
        super().__init__()
        spatial_y = torch.linspace(-1.0, 1.0, height)
        spatial_x = torch.linspace(-1.0, 1.0, width)
        self.time_kernel = CoordinatePairKernel(input_time, output_time, rank)
        self.y_kernel = CoordinatePairKernel(spatial_y, spatial_y, rank)
        self.x_kernel = CoordinatePairKernel(spatial_x, spatial_x, rank)
        self.rank_weight = nn.Parameter(torch.ones(rank, channels) / rank)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 5:
            raise ValueError(f"expected [B,C,T,H,W], got {tuple(hidden.shape)}")
        time = self.time_kernel(dtype=hidden.dtype)
        vertical = self.y_kernel(dtype=hidden.dtype)
        horizontal = self.x_kernel(dtype=hidden.dtype)
        mixed = torch.einsum("rou,bcuyx->brcoyx", time, hidden)
        mixed = torch.einsum("rhy,brcoyx->brcohx", vertical, mixed)
        mixed = torch.einsum("rwx,brcohx->brcohw", horizontal, mixed)
        return torch.einsum(
            "rc,brctyx->bctyx", self.rank_weight.to(hidden.dtype), mixed
        )


class KhatriRaoBlock(nn.Module):
    def __init__(
        self,
        *,
        input_time: torch.Tensor,
        output_time: torch.Tensor,
        height: int,
        width: int,
        channels: int,
        rank: int,
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.integral = KhatriRaoIntegral(
            input_time=input_time,
            output_time=output_time,
            height=height,
            width=width,
            rank=rank,
            channels=channels,
        )
        self.channel = nn.Conv3d(channels, channels, 1)
        self.norm2 = nn.GroupNorm(groups, channels)
        expanded = channels * expansion
        self.ffn = nn.Sequential(
            nn.Conv3d(channels, expanded * 2, 1),
            nn.GLU(dim=1),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(expanded, channels, 1),
        )
        self.integral_scale = nn.Parameter(torch.full((1, channels, 1, 1, 1), 0.05))
        self.channel_scale = nn.Parameter(torch.full((1, channels, 1, 1, 1), 0.05))
        self.ffn_scale = nn.Parameter(torch.full((1, channels, 1, 1, 1), 0.05))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(hidden)
        hidden = (
            hidden
            + self.integral_scale * self.integral(normalized)
            + self.channel_scale * self.channel(normalized)
        )
        return hidden + self.ffn_scale * self.ffn(self.norm2(hidden))


class KhatriRaoFlowOperator(nn.Module):
    """Direct 20-to-20 complete forecaster with coordinate-pair kernels."""

    def __init__(self, config: KhatriRaoConfig | None = None) -> None:
        super().__init__()
        self.config = config or KhatriRaoConfig()
        if self.config.in_steps != 20 or self.config.out_steps != 20:
            raise ValueError("C371 is fixed to the official 20-to-20 contract")
        if 32 % self.config.spatial_bottleneck or 64 % self.config.spatial_bottleneck:
            raise ValueError("spatial bottleneck must divide 32x64 exactly")
        modalities = 3 if self.config.use_pressure else 2
        latent_height = 32 // self.config.spatial_bottleneck
        latent_width = 64 // self.config.spatial_bottleneck
        observed_time = torch.linspace(-1.0, 0.0, self.config.in_steps)
        future_time = torch.linspace(
            1.0 / self.config.out_steps, 1.0, self.config.out_steps
        )
        self.lift = nn.Conv3d(modalities, self.config.width, 1)
        blocks = []
        for index in range(self.config.depth):
            blocks.append(
                KhatriRaoBlock(
                    input_time=observed_time if index == 0 else future_time,
                    output_time=future_time,
                    height=latent_height,
                    width=latent_width,
                    channels=self.config.width,
                    rank=self.config.rank,
                    expansion=self.config.expansion,
                    dropout=self.config.dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        groups = 8 if self.config.width % 8 == 0 else 1
        self.final_norm = nn.GroupNorm(groups, self.config.width)
        self.velocity_head = nn.Sequential(
            nn.Conv3d(self.config.width, self.config.width, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(self.config.width, 2, 1),
        )
        self.width_head = nn.Sequential(
            nn.Conv3d(self.config.width, self.config.width // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(self.config.width // 2, 2, 1),
        )
        self.trend_scale = nn.Parameter(torch.zeros(self.config.out_steps, 2))
        self.interval_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        nn.init.normal_(self.velocity_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.velocity_head[-1].bias)
        nn.init.normal_(self.width_head[-1].weight, std=1e-3)
        nn.init.constant_(self.width_head[-1].bias, -2.5)

    def _features(
        self, flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (self.config.in_steps, 32, 64, 3)
        if flow.ndim != 5 or tuple(flow.shape[1:]) != expected:
            raise ValueError(f"expected [B,{expected}], got {tuple(flow.shape)}")
        selected = flow if self.config.use_pressure else flow[..., :2]
        mean = selected.mean(dim=(1, 2, 3), keepdim=True)
        std = selected.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized = (selected - mean) / std
        latent = normalized.permute(0, 4, 1, 2, 3)
        latent = F.avg_pool3d(
            latent,
            kernel_size=(1, self.config.spatial_bottleneck, self.config.spatial_bottleneck),
            stride=(1, self.config.spatial_bottleneck, self.config.spatial_bottleneck),
        )
        uv_mean = mean[..., :2]
        uv_std = std[..., :2]
        return latent, uv_mean, uv_std

    def forward(
        self, flow: torch.Tensor, *, return_interval: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        latent, uv_mean, uv_std = self._features(flow)
        hidden = self.lift(latent)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        hidden = F.interpolate(
            hidden,
            size=(self.config.out_steps, 32, 64),
            mode="trilinear",
            align_corners=False,
        )
        correction = self.velocity_head(hidden).permute(0, 2, 3, 4, 1)
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
        raw_width = self.width_head(hidden).permute(0, 2, 3, 4, 1)
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
