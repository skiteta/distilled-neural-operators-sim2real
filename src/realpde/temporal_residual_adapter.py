from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from realpde.temporal_spectral_flow import project_local_tke


@dataclass(frozen=True)
class TemporalResidualAdapterConfig:
    in_steps: int = 20
    out_steps: int = 20
    width: int = 24
    depth: int = 3
    maximum_normalized_correction: float = 0.20

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TemporalResidualAdapterConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AdapterBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        self.norm = nn.GroupNorm(groups, width)
        self.spatial = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.channel = nn.Conv2d(width, width, 1)
        self.scale = nn.Parameter(torch.full((1, width, 1, 1), 1e-2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = F.silu(self.norm(features))
        return features + self.scale * (
            self.spatial(normalized) + self.channel(normalized)
        )


class TemporalResidualAdapter(nn.Module):
    """Cheap state-dependent correction distilled from a generative teacher.

    Temporal frames are explicit channels.  The 1x1 maps therefore learn full
    temporal/phase mixing, while depthwise spatial kernels only model the local
    transport missing from a pointwise temporal map.
    """

    def __init__(self, config: TemporalResidualAdapterConfig | None = None) -> None:
        super().__init__()
        self.config = config or TemporalResidualAdapterConfig()
        temporal_channels = 2 * (
            self.config.in_steps + 2 * self.config.out_steps
        )
        # Normalized observed history, normalized baseline, normalized student
        # correction, x/y coordinates, and four global state descriptors.
        input_channels = temporal_channels + 2 + 4
        self.lift = nn.Conv2d(input_channels, self.config.width, 1)
        self.blocks = nn.ModuleList(
            AdapterBlock(self.config.width) for _ in range(self.config.depth)
        )
        groups = 8 if self.config.width % 8 == 0 else 1
        self.norm = nn.GroupNorm(groups, self.config.width)
        self.head = nn.Conv2d(
            self.config.width, self.config.out_steps * 2, 1
        )
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

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
        return torch.stack((grid_x, grid_y), dim=0)[None].expand(
            batch, -1, -1, -1
        )

    def normalized_correction(
        self,
        observed: torch.Tensor,
        student: torch.Tensor,
        baseline: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed_uv = observed[..., :2].float()
        mean = observed_uv.mean(dim=(1, 2, 3), keepdim=True)
        scale = observed_uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        observed_normalized = (observed_uv - mean) / scale
        baseline_normalized = (baseline[..., :2].float() - mean) / scale
        student_delta = (
            student[..., :2].float() - baseline[..., :2].float()
        ) / scale
        batch, _, height, width, _ = observed.shape
        temporal = torch.cat(
            (observed_normalized, baseline_normalized, student_delta), dim=1
        ).permute(0, 1, 4, 2, 3).reshape(batch, -1, height, width)
        final_mean = observed_normalized[:, -1].mean(dim=(1, 2))
        recent_trend = (
            observed_normalized[:, -1] - observed_normalized[:, -5]
        ).mean(dim=(1, 2)) / 4.0
        descriptors = torch.cat((final_mean, recent_trend), dim=1)
        descriptors = descriptors[:, :, None, None].expand(
            -1, -1, height, width
        )
        coordinates = self._coordinates(
            batch, height, width, observed.device, observed.dtype
        )
        hidden = self.lift(torch.cat((temporal, coordinates, descriptors), dim=1))
        for block in self.blocks:
            hidden = block(hidden)
        raw = self.head(F.silu(self.norm(hidden))).reshape(
            batch, self.config.out_steps, 2, height, width
        )
        raw = raw.permute(0, 1, 3, 4, 2)
        correction = self.config.maximum_normalized_correction * torch.tanh(raw)
        return correction, scale

    def forward(
        self,
        observed: torch.Tensor,
        student: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        correction, scale = self.normalized_correction(
            observed, student, baseline
        )
        candidate_uv = student[..., :2].float() + correction * scale
        candidate = torch.cat(
            (candidate_uv, torch.zeros_like(candidate_uv[..., :1])), dim=-1
        )
        return project_local_tke(candidate, baseline)

