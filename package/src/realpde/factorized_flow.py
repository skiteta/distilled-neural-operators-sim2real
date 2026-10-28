from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from realpde.model import HybridFlowOperator


@dataclass(frozen=True)
class FactorizedFlowConfig:
    blend_weight: float = 0.65
    energy_width: int = 64
    energy_depth: int = 3
    maximum_log_amplitude: float = 0.7
    use_trajectory_features: bool = False
    energy_downsample: int = 1

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FactorizedFlowConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EnergyBlock(nn.Module):
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


def flow_prediction_and_hidden(
    model: HybridFlowOperator, flow: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one Hybrid backbone once and expose its final shared feature map."""

    raw = flow
    features, uv_mean, uv_std = model._features(raw)
    hidden = model.lift(features)
    for index, block in enumerate(model.blocks):
        hidden = block(hidden)
        if index + 1 == max(1, len(model.blocks) // 2):
            hidden = model.multiscale(hidden)
    hidden = model.final_norm(hidden)
    batch, _, height, width = hidden.shape
    correction = model.velocity_head(hidden).reshape(
        batch, model.config.out_steps, 2, height, width
    )
    correction = correction.permute(0, 1, 3, 4, 2)
    base_normalized = (raw[:, -1:, ..., :2] - uv_mean) / uv_std
    recent_trend = (
        raw[:, -1:, ..., :2] - raw[:, -5:-4, ..., :2]
    ) / (4 * uv_std)
    trend = recent_trend * model.trend_scale[None, :, None, None, :]
    prediction_uv = (base_normalized + trend + correction) * uv_std + uv_mean
    prediction = torch.cat(
        (prediction_uv, torch.zeros_like(prediction_uv[..., :1])), dim=-1
    )
    return prediction, hidden


def local_energy(flow: torch.Tensor) -> torch.Tensor:
    uv = flow[..., :2].float()
    fluctuation = uv - uv.mean(dim=1, keepdim=True)
    return fluctuation.square().sum(dim=-1).mean(dim=1)


class FactorizedTwoBackboneFlow(nn.Module):
    """Forecast mean/phase with frozen experts and energy with a shared head."""

    def __init__(
        self,
        unet: HybridFlowOperator,
        specialist: HybridFlowOperator,
        config: FactorizedFlowConfig | None = None,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.specialist = specialist
        self.config = config or FactorizedFlowConfig()
        self.unet.requires_grad_(False)
        self.specialist.requires_grad_(False)
        self.unet.eval()
        self.specialist.eval()
        trajectory_channels = 60 if self.config.use_trajectory_features else 0
        diagnostic_channels = (
            unet.config.width + specialist.config.width + 7 + trajectory_channels
        )
        self.energy_lift = nn.Conv2d(
            diagnostic_channels, self.config.energy_width, 1
        )
        self.energy_blocks = nn.ModuleList(
            EnergyBlock(self.config.energy_width)
            for _ in range(self.config.energy_depth)
        )
        groups = 8 if self.config.energy_width % 8 == 0 else 1
        self.energy_norm = nn.GroupNorm(groups, self.config.energy_width)
        self.energy_head = nn.Conv2d(self.config.energy_width, 1, 1)
        nn.init.zeros_(self.energy_head.weight)
        nn.init.zeros_(self.energy_head.bias)

    def train(self, mode: bool = True) -> "FactorizedTwoBackboneFlow":
        super().train(mode)
        self.unet.eval()
        self.specialist.eval()
        return self

    @staticmethod
    def _log_energy(energy: torch.Tensor) -> torch.Tensor:
        return torch.log(energy.clamp_min(1e-8))[:, None]

    def components(
        self, observed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        unet_point, unet_hidden = flow_prediction_and_hidden(self.unet, observed)
        specialist_point, specialist_hidden = flow_prediction_and_hidden(
            self.specialist, observed
        )
        raw = torch.lerp(
            unet_point,
            specialist_point,
            float(self.config.blend_weight),
        )
        observed_uv = observed[..., :2].float()
        observed_first = local_energy(
            torch.cat(
                (
                    observed_uv[:, :10],
                    torch.zeros_like(observed_uv[:, :10, ..., :1]),
                ),
                dim=-1,
            )
        )
        observed_recent = local_energy(
            torch.cat(
                (
                    observed_uv[:, 10:],
                    torch.zeros_like(observed_uv[:, 10:, ..., :1]),
                ),
                dim=-1,
            )
        )
        observed_full = local_energy(observed)
        raw_energy = local_energy(raw)
        unet_energy = local_energy(unet_point)
        specialist_energy = local_energy(specialist_point)
        trend = torch.log(
            observed_recent.clamp_min(1e-8)
            / observed_first.clamp_min(1e-8)
        )[:, None]
        energy_features = torch.cat(
            (
                self._log_energy(observed_full),
                self._log_energy(observed_recent),
                trend,
                self._log_energy(unet_energy),
                self._log_energy(specialist_energy),
                self._log_energy(raw_energy),
                self._log_energy(
                    0.5 * (unet_energy + specialist_energy)
                ),
            ),
            dim=1,
        )
        if self.config.use_trajectory_features:
            centered_uv = observed_uv - observed_uv.mean(dim=1, keepdim=True)
            local_scale = centered_uv.square().sum(dim=-1).mean(
                dim=1, keepdim=True
            ).sqrt().clamp_min(1e-4)
            normalized_trajectory = (
                centered_uv / local_scale[..., None]
            ).clamp(-5.0, 5.0)
            normalized_trajectory = normalized_trajectory.permute(
                0, 1, 4, 2, 3
            ).flatten(1, 2)
            instantaneous_energy = centered_uv.square().sum(dim=-1)
            relative_log_energy = torch.log(
                instantaneous_energy.clamp_min(1e-8)
                / local_scale.square().clamp_min(1e-8)
            ).clamp(-5.0, 5.0)
            energy_features = torch.cat(
                (
                    energy_features,
                    normalized_trajectory,
                    relative_log_energy,
                ),
                dim=1,
            )
        hidden = self.energy_lift(
            torch.cat((unet_hidden, specialist_hidden, energy_features), dim=1)
        )
        for block in self.energy_blocks:
            hidden = block(hidden)
        head_features = F.silu(self.energy_norm(hidden))
        if self.config.energy_downsample > 1:
            factor = int(self.config.energy_downsample)
            coarse_features = F.avg_pool2d(
                head_features, kernel_size=factor, stride=factor
            )
            coarse_log_amplitude = self.energy_head(coarse_features)
            raw_log_amplitude = F.interpolate(
                coarse_log_amplitude,
                size=hidden.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        else:
            raw_log_amplitude = self.energy_head(head_features)[:, 0]
        log_amplitude = self.config.maximum_log_amplitude * torch.tanh(
            raw_log_amplitude
        )
        return raw, log_amplitude, {
            "raw_energy": raw_energy,
            "unet_energy": unet_energy,
            "specialist_energy": specialist_energy,
        }

    @staticmethod
    def reconstruct(
        raw: torch.Tensor, log_amplitude: torch.Tensor
    ) -> torch.Tensor:
        uv = raw[..., :2].float()
        temporal_mean = uv.mean(dim=1, keepdim=True)
        direction = uv - temporal_mean
        scaled_uv = temporal_mean + direction * torch.exp(log_amplitude)[
            :, None, :, :, None
        ]
        return torch.cat(
            (scaled_uv, torch.zeros_like(scaled_uv[..., :1])), dim=-1
        )

    def forward(self, observed: torch.Tensor) -> torch.Tensor:
        raw, log_amplitude, _ = self.components(observed)
        return self.reconstruct(raw, log_amplitude)

