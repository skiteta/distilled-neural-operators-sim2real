from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PhysicsAttentionConfig:
    """Fixed C120 state-sliced direct-operator configuration."""

    in_steps: int = 20
    out_steps: int = 20
    width: int = 160
    depth: int = 8
    slices: int = 32
    heads: int = 8
    expansion: int = 3
    dropout: float = 0.03
    use_pressure: bool = True
    use_physics_features: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PhysicsSliceBlock(nn.Module):
    """Aggregate moving pixels into learned physical-state slices, evolve, scatter."""

    def __init__(
        self,
        width: int,
        slices: int,
        heads: int,
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        hidden = width * expansion
        self.width = width
        self.slices = slices
        self.norm1 = nn.GroupNorm(groups, width)
        self.slice_logits = nn.Linear(width, slices)
        self.slice_values = nn.Linear(width, width)
        self.slice_norm = nn.LayerNorm(width)
        self.slice_attention = nn.MultiheadAttention(
            width,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.scatter_projection = nn.Linear(width, width)
        self.local_transport = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            padding=1,
            groups=width,
        )
        self.norm2 = nn.GroupNorm(groups, width)
        self.ffn = nn.Sequential(
            nn.Conv2d(width, hidden * 2, 1),
            nn.GLU(dim=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, width, 1),
        )
        self.state_scale = nn.Parameter(torch.full((1, width, 1, 1), 1e-2))
        self.ffn_scale = nn.Parameter(torch.full((1, width, 1, 1), 1e-2))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = hidden.shape
        normalized = self.norm1(hidden)
        tokens = normalized.flatten(2).transpose(1, 2)
        membership = torch.softmax(
            self.slice_logits(tokens) / (self.width**0.5),
            dim=-1,
        )
        values = self.slice_values(tokens)
        slice_mass = membership.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        slice_tokens = torch.einsum("bns,bnc->bsc", membership, values) / slice_mass
        slice_tokens = self.slice_norm(slice_tokens)
        evolved, _ = self.slice_attention(
            slice_tokens,
            slice_tokens,
            slice_tokens,
            need_weights=False,
        )
        evolved = slice_tokens + evolved
        scattered = torch.einsum("bns,bsc->bnc", membership, evolved)
        scattered = self.scatter_projection(scattered)
        scattered = scattered.transpose(1, 2).reshape(batch, channels, height, width)
        state_update = scattered + self.local_transport(normalized)
        hidden = hidden + self.state_scale * state_update
        return hidden + self.ffn_scale * self.ffn(self.norm2(hidden))


class PhysicsAttentionFlowOperator(nn.Module):
    """Scratch 20-to-20 operator with window-local physical-state slicing."""

    def __init__(self, config: PhysicsAttentionConfig | None = None) -> None:
        super().__init__()
        self.config = config or PhysicsAttentionConfig()
        if self.config.in_steps != 20 or self.config.out_steps != 20:
            raise ValueError("C120 is fixed to the official 20-to-20 contract")
        if self.config.width % self.config.heads:
            raise ValueError("width must be divisible by attention heads")
        modalities = 3 if self.config.use_pressure else 2
        descriptor_channels = 4 * modalities
        physics_channels = 9 if self.config.use_physics_features else 0
        in_channels = (
            self.config.in_steps * modalities
            + 2
            + descriptor_channels
            + physics_channels
        )
        self.lift = nn.Conv2d(in_channels, self.config.width, 1)
        self.blocks = nn.ModuleList(
            PhysicsSliceBlock(
                self.config.width,
                self.config.slices,
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
        self.interval_temperature = nn.Parameter(
            torch.tensor(1.0),
            requires_grad=False,
        )
        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)
        nn.init.zeros_(self.width_head[-1].weight)
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
        return (
            torch.stack((grid_x, grid_y), dim=0)
            .unsqueeze(0)
            .expand(batch, -1, -1, -1)
        )

    def _features(
        self,
        flow: torch.Tensor,
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
                (normalized_uv, (pressure - pressure_mean) / pressure_std),
                dim=-1,
            )
            selected = flow
        else:
            normalized = normalized_uv

        batch, _, height, width, _ = flow.shape
        flat = normalized.permute(0, 1, 4, 2, 3).reshape(
            batch,
            -1,
            height,
            width,
        )
        global_mean = selected.mean(dim=(1, 2, 3))
        global_std = selected.std(dim=(1, 2, 3))
        final_mean = selected[:, -1].mean(dim=(1, 2))
        final_std = selected[:, -1].std(dim=(1, 2))
        descriptors = torch.cat(
            (global_mean, global_std, final_mean, final_std),
            dim=-1,
        )
        descriptors = descriptors[:, :, None, None].expand(
            -1,
            -1,
            height,
            width,
        )
        coordinates = self._coordinates(
            batch,
            height,
            width,
            flow.device,
            flow.dtype,
        )
        parts = [flat, coordinates, descriptors]
        if self.config.use_physics_features:
            temporal_mean = uv.mean(dim=1)
            fluctuation = uv - temporal_mean[:, None]
            variance = fluctuation.square().mean(dim=1)
            temporal_std = variance.clamp_min(1e-8).sqrt()
            local_tke = 0.5 * variance.sum(dim=-1, keepdim=True)
            recent = (uv[:, -1] - uv[:, -5]) / 4.0
            final_fluctuation = uv[:, -1] - temporal_mean
            physics = torch.cat(
                (
                    temporal_mean,
                    temporal_std,
                    local_tke,
                    recent,
                    final_fluctuation,
                ),
                dim=-1,
            ).permute(0, 3, 1, 2)
            parts.append(physics)
        return torch.cat(parts, dim=1), uv_mean, uv_std

    def forward(
        self,
        flow: torch.Tensor,
        *,
        return_interval: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features, uv_mean, uv_std = self._features(flow)
        hidden = self.lift(features)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)

        batch, _, height, width = hidden.shape
        correction = self.velocity_head(hidden).reshape(
            batch,
            self.config.out_steps,
            2,
            height,
            width,
        )
        correction = correction.permute(0, 1, 3, 4, 2)
        base_normalized = (flow[:, -1:, ..., :2] - uv_mean) / uv_std
        recent_trend = (
            flow[:, -1:, ..., :2] - flow[:, -5:-4, ..., :2]
        ) / (4.0 * uv_std)
        trend = recent_trend * self.trend_scale[None, :, None, None, :]
        prediction_uv = (base_normalized + trend + correction) * uv_std + uv_mean
        prediction = torch.cat(
            (prediction_uv, torch.zeros_like(prediction_uv[..., :1])),
            dim=-1,
        )
        if not return_interval:
            return prediction

        raw_width = self.width_head(hidden).reshape(
            batch,
            self.config.out_steps,
            2,
            height,
            width,
        )
        raw_width = raw_width.permute(0, 1, 3, 4, 2)
        half_width_uv = (
            F.softplus(raw_width)
            * uv_std
            * self.interval_temperature.clamp_min(1e-3)
        )
        half_width = torch.cat(
            (half_width_uv, torch.zeros_like(half_width_uv[..., :1])),
            dim=-1,
        )
        return prediction, half_width

    def predict(
        self,
        flow: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del metadata
        return self(flow)
