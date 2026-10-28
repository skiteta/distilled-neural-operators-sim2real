from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TemporalCrossAttentionConfig:
    """Fixed C269 content-adaptive 20-to-20 operator configuration."""

    in_steps: int = 20
    out_steps: int = 20
    width: int = 48
    heads: int = 4
    observed_depth: int = 2
    future_depth: int = 2
    expansion: int = 2
    dropout: float = 0.02
    maximum_normalized_correction: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FrameSpatialBlock(nn.Module):
    """Light spatial mixing while retaining every observed frame as a token."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        channel_groups = 4 if width % 4 == 0 else 1
        self.norm = nn.GroupNorm(groups, width)
        self.spatial = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.channel = nn.Sequential(
            nn.Conv2d(width, width, 1, groups=channel_groups),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(width, width, 1),
        )
        self.scale = nn.Parameter(torch.full((1, width, 1, 1), 1e-2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mixed = self.channel(self.spatial(self.norm(value)))
        return value + self.scale * mixed


class FutureSpaceTimeBlock(nn.Module):
    """Local future-horizon and spatial interaction after cross-attention."""

    def __init__(self, width: int, expansion: int, dropout: float) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        channel_groups = 4 if width % 4 == 0 else 1
        hidden = width * expansion
        self.norm1 = nn.GroupNorm(groups, width)
        self.local = nn.Conv3d(
            width,
            width,
            kernel_size=(3, 3, 3),
            padding=1,
            groups=width,
        )
        self.channel = nn.Conv3d(width, width, 1, groups=channel_groups)
        self.norm2 = nn.GroupNorm(groups, width)
        self.ffn = nn.Sequential(
            nn.Conv3d(width, hidden, 1, groups=channel_groups),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(hidden, width, 1, groups=channel_groups),
        )
        self.local_scale = nn.Parameter(torch.full((1, width, 1, 1, 1), 1e-2))
        self.ffn_scale = nn.Parameter(torch.full((1, width, 1, 1, 1), 1e-2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.local_scale * self.channel(self.local(self.norm1(value)))
        return value + self.ffn_scale * self.ffn(self.norm2(value))


class TemporalCrossAttentionFlowOperator(nn.Module):
    """Window-local, content-adaptive temporal retrieval for parallel futures."""

    def __init__(self, config: TemporalCrossAttentionConfig | None = None) -> None:
        super().__init__()
        self.config = config or TemporalCrossAttentionConfig()
        if self.config.in_steps != 20 or self.config.out_steps != 20:
            raise ValueError("C269 is fixed to the official 20-to-20 contract")
        if self.config.width % self.config.heads:
            raise ValueError("width must be divisible by heads")
        width = self.config.width
        # Normalized u/v plus observed time and x/y coordinates.
        self.lift = nn.Conv2d(5, width, 1)
        self.observed_blocks = nn.ModuleList(
            FrameSpatialBlock(width, self.config.dropout)
            for _ in range(self.config.observed_depth)
        )
        self.token_norm = nn.LayerNorm(width)
        self.context_projection = nn.Linear(3 * width, width)
        self.future_queries = nn.Parameter(torch.empty(self.config.out_steps, width))
        self.query_projection = nn.Linear(width, width)
        self.key_projection = nn.Linear(width, width)
        self.value_projection = nn.Linear(width, width)
        self.attention_output = nn.Linear(width, width)
        self.attention_scale = nn.Parameter(torch.tensor(1e-2))
        self.future_blocks = nn.ModuleList(
            FutureSpaceTimeBlock(
                width,
                self.config.expansion,
                self.config.dropout,
            )
            for _ in range(self.config.future_depth)
        )
        groups = 8 if width % 8 == 0 else 1
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
        nn.init.normal_(self.future_queries, std=0.02)
        nn.init.zeros_(self.point_head[-1].weight)
        nn.init.zeros_(self.point_head[-1].bias)
        nn.init.zeros_(self.width_head[-1].weight)
        nn.init.constant_(self.width_head[-1].bias, -2.0)

    @staticmethod
    def _normalize(
        flow: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uv = flow[..., :2].float()
        mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        scale = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        return (uv - mean) / scale, mean, scale

    @staticmethod
    def _coordinates(
        batch: int,
        steps: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        time = torch.linspace(-1.0, 0.0, steps, device=device, dtype=dtype)
        yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        spatial = torch.stack((grid_x, grid_y), dim=-1)[None, None]
        spatial = spatial.expand(batch, steps, height, width, 2)
        temporal = time[None, :, None, None, None].expand(
            batch, steps, height, width, 1
        )
        return torch.cat((spatial, temporal), dim=-1)

    def _cross_attention(
        self, observed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # observed: [B,T,C,h,w] -> one independent temporal sequence per cell.
        batch, steps, channels, height, width = observed.shape
        tokens = observed.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width, steps, channels
        )
        tokens = self.token_norm(tokens)
        context = torch.cat(
            (tokens[:, -1], tokens.mean(dim=1), tokens[:, -1] - tokens[:, -5]),
            dim=-1,
        )
        context = self.context_projection(context)
        queries = context[:, None] + self.future_queries[None]
        queries = self.query_projection(queries)
        keys = self.key_projection(tokens)
        values = self.value_projection(tokens)
        heads = self.config.heads
        head_width = channels // heads
        queries = queries.reshape(-1, self.config.out_steps, heads, head_width).transpose(1, 2)
        keys = keys.reshape(-1, steps, heads, head_width).transpose(1, 2)
        values = values.reshape(-1, steps, heads, head_width).transpose(1, 2)
        logits = torch.matmul(queries.float(), keys.float().transpose(-1, -2))
        logits = logits * (head_width ** -0.5)
        weights = torch.softmax(logits, dim=-1).to(values.dtype)
        attended = torch.matmul(weights, values).transpose(1, 2).reshape(
            -1, self.config.out_steps, channels
        )
        attended = queries.transpose(1, 2).reshape(
            -1, self.config.out_steps, channels
        ) + self.attention_scale * self.attention_output(attended)
        future = attended.reshape(
            batch, height, width, self.config.out_steps, channels
        ).permute(0, 4, 3, 1, 2)
        weights = weights.reshape(
            batch, height * width, heads, self.config.out_steps, steps
        )
        return future, weights

    def forward(
        self,
        flow: torch.Tensor,
        *,
        return_interval: bool = False,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (self.config.in_steps, 32, 64, 3)
        if flow.ndim != 5 or tuple(flow.shape[1:]) != expected:
            raise ValueError(f"expected [B,{expected}], got {tuple(flow.shape)}")
        normalized, mean, scale = self._normalize(flow)
        coordinates = self._coordinates(
            flow.shape[0],
            self.config.in_steps,
            flow.shape[2],
            flow.shape[3],
            device=flow.device,
            dtype=normalized.dtype,
        )
        source = torch.cat((normalized, coordinates), dim=-1)
        source = source.permute(0, 1, 4, 2, 3).reshape(
            -1, 5, flow.shape[2], flow.shape[3]
        )
        source = F.avg_pool2d(source, kernel_size=2, stride=2)
        observed = self.lift(source)
        for block in self.observed_blocks:
            observed = block(observed)
        latent_height, latent_width = observed.shape[-2:]
        observed = observed.reshape(
            flow.shape[0], self.config.in_steps, self.config.width,
            latent_height, latent_width,
        )
        future, attention = self._cross_attention(observed)
        for block in self.future_blocks:
            future = block(future)
        decoded = self.decode_norm(future).permute(0, 2, 1, 3, 4).reshape(
            -1, self.config.width, latent_height, latent_width
        )
        raw_point = self.point_head(decoded)
        raw_width = self.width_head(decoded)
        raw_point = F.interpolate(
            raw_point,
            size=(flow.shape[2], flow.shape[3]),
            mode="bilinear",
            align_corners=False,
        ).reshape(flow.shape[0], self.config.out_steps, 2, flow.shape[2], flow.shape[3])
        raw_width = F.interpolate(
            raw_width,
            size=(flow.shape[2], flow.shape[3]),
            mode="bilinear",
            align_corners=False,
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
        if return_interval and return_attention:
            return point, half, attention
        if return_interval:
            return point, half
        if return_attention:
            return point, attention
        return point

    def predict(
        self,
        input_array: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del metadata
        return self.forward(input_array)
