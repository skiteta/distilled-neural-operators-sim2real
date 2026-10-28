from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .model import HybridFlowOperator, OperatorBlock


@dataclass(frozen=True)
class MeanFlowRefinerConfig:
    width: int = 64
    depth: int = 6
    spectral_rank: int = 16
    modes_y: int = 12
    modes_x: int = 20
    expansion: int = 2
    dropout: float = 0.02
    maximum_normalized_correction: float = 0.25

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "MeanFlowRefinerConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MeanFlowRefiner(nn.Module):
    """Input-conditioned spatial mean-flow correction that preserves TKE exactly."""

    input_channels = 18

    def __init__(self, config: MeanFlowRefinerConfig | None = None) -> None:
        super().__init__()
        self.config = config or MeanFlowRefinerConfig()
        self.lift = nn.Conv2d(self.input_channels, self.config.width, 1)
        self.blocks = nn.ModuleList(
            OperatorBlock(
                self.config.width,
                self.config.spectral_rank,
                self.config.modes_y,
                self.config.modes_x,
                self.config.expansion,
                self.config.dropout,
            )
            for _ in range(self.config.depth)
        )
        groups = 8 if self.config.width % 8 == 0 else 1
        self.norm = nn.GroupNorm(groups, self.config.width)
        self.head = nn.Sequential(
            nn.Conv2d(self.config.width, self.config.width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.config.width, 2, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, observed: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        if observed.ndim != 5 or observed.shape[1:] != (20, 32, 64, 3):
            raise ValueError(f"unexpected observed shape {tuple(observed.shape)}")
        if baseline.shape != observed.shape:
            raise ValueError(f"unexpected baseline shape {tuple(baseline.shape)}")
        uv = observed[..., :2].float()
        base = baseline[..., :2].float()
        mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        std = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        observed_normalized = (uv - mean) / std
        base_normalized = (base - mean) / std

        observed_mean = observed_normalized.mean(dim=1)
        observed_fluctuation = observed_normalized - observed_mean[:, None]
        observed_variance = observed_fluctuation.square().mean(dim=1)
        observed_std = observed_variance.clamp_min(1e-8).sqrt()
        observed_tke = 0.5 * observed_variance.sum(dim=-1, keepdim=True)
        recent = (observed_normalized[:, -1] - observed_normalized[:, -5]) / 4.0
        final_fluctuation = observed_normalized[:, -1] - observed_mean

        base_mean = base_normalized.mean(dim=1)
        base_fluctuation = base_normalized - base_mean[:, None]
        base_variance = base_fluctuation.square().mean(dim=1)
        base_std = base_variance.clamp_min(1e-8).sqrt()
        base_tke = 0.5 * base_variance.sum(dim=-1, keepdim=True)
        base_last_delta = base_normalized[:, -1] - observed_normalized[:, -1]
        coordinates = HybridFlowOperator._coordinates(
            observed.shape[0], observed.shape[2], observed.shape[3], observed.device, uv.dtype
        ).permute(0, 2, 3, 1)
        features = torch.cat(
            (
                observed_mean,
                observed_std,
                observed_tke,
                recent,
                final_fluctuation,
                base_mean,
                base_std,
                base_tke,
                base_last_delta,
                coordinates,
            ),
            dim=-1,
        ).permute(0, 3, 1, 2)
        if features.shape[1] != self.input_channels:
            raise RuntimeError(f"unexpected mean-flow channels: {features.shape[1]}")
        hidden = self.lift(features)
        for block in self.blocks:
            hidden = block(hidden)
        raw = self.head(self.norm(hidden)).permute(0, 2, 3, 1)
        correction = self.config.maximum_normalized_correction * torch.tanh(raw) * std[:, 0]
        refined_uv = base + correction[:, None]
        return torch.cat((refined_uv, torch.zeros_like(refined_uv[..., :1])), dim=-1)

