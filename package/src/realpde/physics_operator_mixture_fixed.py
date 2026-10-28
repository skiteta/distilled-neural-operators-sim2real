from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from .physics_operator_mixture import (
    PhysicsOperatorMixture,
    PhysicsOperatorMixtureConfig,
)


class FixedPhysicsOperatorMixture(PhysicsOperatorMixture):
    """C402R: shape-only repair of the temporal transport broadcast."""

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
        short_coeff = coefficients[:, :, 0, :][:, :, None, None, :]
        long_coeff = coefficients[:, :, 1, :][:, :, None, None, :]
        temporal_raw = short_coeff * short + long_coeff * long

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
        prediction = torch.cat(
            (prediction_uv, torch.zeros_like(prediction_uv[..., :1])), dim=-1
        )

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


__all__ = ["FixedPhysicsOperatorMixture", "PhysicsOperatorMixtureConfig"]

