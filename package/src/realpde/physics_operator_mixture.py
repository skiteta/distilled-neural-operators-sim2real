from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .differential_integral_operator import FixedDerivativeBank
from .model import HybridFlowOperator, LowRankSpectralMix2d, ModelConfig


@dataclass(frozen=True)
class PhysicsOperatorMixtureConfig:
    """Frozen C402 heterogeneous operator-mixture configuration."""

    base: dict[str, Any]
    residual_cap: float = 0.20
    spectral_rank: int = 16
    gate_hidden: int = 96
    gate_temperature: float = 0.7

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PhysicsOperatorMixtureConfig:
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PhysicsOperatorMixture(nn.Module):
    """Full-width Hybrid with local, spectral, and temporal operator experts."""

    expert_names = ("local_differential", "global_spectral", "temporal_transport")

    def __init__(self, config: PhysicsOperatorMixtureConfig) -> None:
        super().__init__()
        self.mixture_config = config
        base_config = ModelConfig.from_dict(config.base)
        self.config = base_config
        self.base = HybridFlowOperator(base_config)
        width = base_config.width
        groups = 8 if width % 8 == 0 else 1

        self.local_norm = nn.GroupNorm(groups, width)
        self.local_operator = FixedDerivativeBank(width)
        self.local_head = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(width, width // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, base_config.out_steps * 2, 1),
        )

        self.spectral_norm = nn.GroupNorm(groups, width)
        self.spectral_operator = LowRankSpectralMix2d(
            width,
            config.spectral_rank,
            base_config.modes_y,
            base_config.modes_x,
        )
        self.spectral_head = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(width, width // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, base_config.out_steps * 2, 1),
        )

        self.transport_coefficients = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, config.gate_hidden),
            nn.GELU(),
            nn.Linear(config.gate_hidden, base_config.out_steps * 2 * 2),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, config.gate_hidden),
            nn.GELU(),
            nn.Linear(config.gate_hidden, base_config.out_steps * 3),
        )

        for head in (self.local_head, self.spectral_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.zeros_(self.transport_coefficients[-1].weight)
        nn.init.zeros_(self.transport_coefficients[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def _shared(
        self, flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, uv_mean, uv_std = self.base._features(flow)
        hidden = self.base.lift(features)
        for index, block in enumerate(self.base.blocks):
            hidden = block(hidden)
            if index + 1 == max(1, len(self.base.blocks) // 2):
                hidden = self.base.multiscale(hidden)
        hidden = self.base.final_norm(hidden)
        batch, _, height, width = hidden.shape
        correction = self.base.velocity_head(hidden).reshape(
            batch, self.config.out_steps, 2, height, width
        ).permute(0, 1, 3, 4, 2)
        normalized_last = (flow[:, -1:, ..., :2] - uv_mean) / uv_std
        normalized_recent = (flow[:, -1:, ..., :2] - flow[:, -5:-4, ..., :2]) / (
            4.0 * uv_std
        )
        trend = normalized_recent * self.base.trend_scale[None, :, None, None, :]
        common = (normalized_last + trend + correction) * uv_std + uv_mean
        return hidden, common, uv_mean, uv_std

    def _reshape_head(self, raw: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = raw.shape
        return raw.reshape(batch, self.config.out_steps, 2, height, width).permute(
            0, 1, 3, 4, 2
        )

    def forward(
        self,
        flow: torch.Tensor,
        *,
        return_interval: bool = False,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        hidden, common, _uv_mean, uv_std = self._shared(flow)
        local_latent = self.local_operator(self.local_norm(hidden))
        local_raw = self._reshape_head(self.local_head(local_latent))
        spectral_latent = self.spectral_operator(self.spectral_norm(hidden))
        spectral_raw = self._reshape_head(self.spectral_head(spectral_latent))

        batch = len(flow)
        coefficients = self.transport_coefficients(hidden).reshape(
            batch, self.config.out_steps, 2, 2
        )
        short = (flow[:, -1:, ..., :2] - flow[:, -5:-4, ..., :2]) / (4.0 * uv_std)
        long = (flow[:, -1:, ..., :2] - flow[:, :1, ..., :2]) / (19.0 * uv_std)
        temporal_raw = (
            coefficients[:, :, 0, :, None, None, :] * short
            + coefficients[:, :, 1, :, None, None, :] * long
        )

        cap = float(self.mixture_config.residual_cap)
        residuals = torch.stack(
            (
                cap * torch.tanh(local_raw) * uv_std,
                cap * torch.tanh(spectral_raw) * uv_std,
                cap * torch.tanh(temporal_raw) * uv_std,
            ),
            dim=1,
        )
        expert_uv = common[:, None] + residuals
        logits = self.gate(hidden).reshape(batch, self.config.out_steps, 3)
        weights = F.softmax(logits / float(self.mixture_config.gate_temperature), dim=-1)
        prediction_uv = (
            expert_uv * weights.permute(0, 2, 1)[:, :, :, None, None, None]
        ).sum(dim=1)
        pressure = torch.zeros_like(prediction_uv[..., :1])
        prediction = torch.cat((prediction_uv, pressure), dim=-1)

        _, _, height, width = hidden.shape
        raw_width = self.base.width_head(hidden).reshape(
            batch, self.config.out_steps, 2, height, width
        ).permute(0, 1, 3, 4, 2)
        half_uv = (
            F.softplus(raw_width)
            * uv_std
            * self.base.interval_temperature.clamp_min(1e-3)
        )
        half = torch.cat((half_uv, torch.zeros_like(half_uv[..., :1])), dim=-1)
        if return_details:
            expert_pressure = torch.zeros_like(expert_uv[..., :1])
            return {
                "prediction": prediction,
                "half_width": half,
                "experts": torch.cat((expert_uv, expert_pressure), dim=-1),
                "residuals": residuals,
                "logits": logits,
                "weights": weights,
                "common": torch.cat((common, torch.zeros_like(common[..., :1])), dim=-1),
            }
        if return_interval:
            return prediction, half
        return prediction

    @torch.inference_mode()
    def diagnostics(self, flow: torch.Tensor) -> dict[str, Any]:
        details = self(flow, return_details=True)
        residuals = details["residuals"].float()
        weights = details["weights"].float()
        return {
            "residual_rms": {
                name: float(residuals[:, index].square().mean().sqrt())
                for index, name in enumerate(self.expert_names)
            },
            "mean_weights": {
                name: float(weights[..., index].mean())
                for index, name in enumerate(self.expert_names)
            },
            "maximum_branch_spread": float(
                (details["experts"].float().amax(dim=1) - details["experts"].float().amin(dim=1)).abs().max()
            ),
        }

    def predict(
        self, input_array: torch.Tensor, metadata: dict[str, Any] | None = None
    ) -> torch.Tensor:
        del metadata
        return self(input_array)

