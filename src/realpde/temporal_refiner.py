from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class TemporalRefinerConfig:
    width: int = 32
    depth: int = 5
    expansion: int = 2
    dropout: float = 0.02
    maximum_normalized_correction: float = 0.25

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TemporalRefinerConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TemporalTransitionBlock(nn.Module):
    """Dilated temporal transition plus local spatial transport."""

    def __init__(self, width: int, expansion: int, dropout: float, dilation: int) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        hidden = width * expansion
        self.norm1 = nn.GroupNorm(groups, width)
        self.temporal = nn.Conv3d(
            width,
            width,
            (3, 1, 1),
            padding=(dilation, 0, 0),
            dilation=(dilation, 1, 1),
            groups=width,
        )
        self.spatial = nn.Conv3d(
            width, width, (1, 3, 3), padding=(0, 1, 1), groups=width
        )
        self.channel = nn.Conv3d(width, width, 1)
        self.norm2 = nn.GroupNorm(groups, width)
        self.ffn = nn.Sequential(
            nn.Conv3d(width, hidden * 2, 1),
            nn.GLU(dim=1),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(hidden, width, 1),
        )
        self.scale = nn.Parameter(torch.full((1, width, 1, 1, 1), 1e-2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(features)
        update = self.temporal(normalized) + self.spatial(normalized) + self.channel(normalized)
        features = features + self.scale * update
        return features + self.scale * self.ffn(self.norm2(features))


class TemporalTrajectoryRefiner(nn.Module):
    """Zero-initialized residual correction over an existing 20-frame forecast."""

    input_channels = 18

    def __init__(self, config: TemporalRefinerConfig | None = None) -> None:
        super().__init__()
        self.config = config or TemporalRefinerConfig()
        width = self.config.width
        groups = 8 if width % 8 == 0 else 1
        self.lift = nn.Conv3d(self.input_channels, width, 1)
        dilations = (1, 2, 4, 8, 1)
        self.blocks = nn.ModuleList(
            TemporalTransitionBlock(
                width,
                self.config.expansion,
                self.config.dropout,
                dilations[index % len(dilations)],
            )
            for index in range(self.config.depth)
        )
        self.norm = nn.GroupNorm(groups, width)
        self.head = nn.Sequential(
            nn.Conv3d(width, width, (1, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(width, 2, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _coordinates(
        batch: int, steps: int, height: int, width: int, source: torch.Tensor
    ) -> torch.Tensor:
        time = torch.linspace(0.05, 1.0, steps, device=source.device, dtype=source.dtype)
        phase = 2.0 * torch.pi * time
        temporal = torch.stack((time, torch.sin(phase), torch.cos(phase)), dim=0)
        temporal = temporal[None, :, :, None, None].expand(batch, -1, -1, height, width)
        yy = torch.linspace(-1.0, 1.0, height, device=source.device, dtype=source.dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=source.device, dtype=source.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        spatial = torch.stack((grid_x, grid_y), dim=0)
        spatial = spatial[None, :, None].expand(batch, -1, steps, -1, -1)
        return torch.cat((temporal, spatial), dim=1)

    def forward(self, observed: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        if observed.ndim != 5 or observed.shape[1:] != (20, 32, 64, 3):
            raise ValueError(f"unexpected observed shape {tuple(observed.shape)}")
        if baseline.shape != observed.shape:
            raise ValueError(f"unexpected baseline shape {tuple(baseline.shape)}")
        uv = observed[..., :2].float()
        base = baseline[..., :2].float()
        mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        std = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized_observed = (uv - mean) / std
        normalized_base = (base - mean) / std
        steps = normalized_base.shape[1]
        last = normalized_observed[:, -1]
        recent = (normalized_observed[:, -1] - normalized_observed[:, -5]) / 4.0
        observed_mean = normalized_observed.mean(dim=1)
        observed_std = normalized_observed.std(dim=1)
        fluctuation = normalized_observed - observed_mean[:, None]
        local_tke = 0.5 * fluctuation.square().sum(dim=-1).mean(dim=1, keepdim=False)[..., None]
        previous = torch.cat((last[:, None], normalized_base[:, :-1]), dim=1)
        transition = normalized_base - previous

        def repeat(value: torch.Tensor) -> torch.Tensor:
            return value[:, None].expand(-1, steps, -1, -1, -1)

        features = torch.cat(
            (
                normalized_base,
                transition,
                repeat(last),
                repeat(recent),
                repeat(observed_mean),
                repeat(observed_std),
                repeat(local_tke),
            ),
            dim=-1,
        ).permute(0, 4, 1, 2, 3)
        coordinates = self._coordinates(
            observed.shape[0], steps, observed.shape[2], observed.shape[3], features
        )
        features = torch.cat((features, coordinates), dim=1)
        if features.shape[1] != self.input_channels:
            raise RuntimeError(f"unexpected refiner channels: {features.shape[1]}")
        hidden = self.lift(features)
        for block in self.blocks:
            hidden = block(hidden)
        raw = self.head(self.norm(hidden)).permute(0, 2, 3, 4, 1)
        correction = self.config.maximum_normalized_correction * torch.tanh(raw)
        refined_uv = base + correction * std
        return torch.cat((refined_uv, torch.zeros_like(refined_uv[..., :1])), dim=-1)

