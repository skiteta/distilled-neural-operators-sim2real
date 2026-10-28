from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .metrics import profile_relative_l2, relative_l2, tke_relative_l2


def spectral_relative_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_spec = torch.fft.rfft2(prediction[..., :2].float(), dim=(2, 3), norm="ortho").abs()
    true_spec = torch.fft.rfft2(target[..., :2].float(), dim=(2, 3), norm="ortho").abs()
    return (pred_spec - true_spec).abs().mean() / true_spec.abs().mean().clamp_min(1e-8)


def divergence_loss(prediction: torch.Tensor) -> torch.Tensor:
    u = prediction[..., 0]
    v = prediction[..., 1]
    du_dx = u[..., 1:] - u[..., :-1]
    dv_dy = v[..., 1:, :] - v[..., :-1, :]
    return (du_dx[..., 1:, :] + dv_dy[..., :, 1:]).square().mean().sqrt()


def interval_quantile_loss(
    prediction: torch.Tensor, target: torch.Tensor, half_width: torch.Tensor, quantile: float = 0.9
) -> torch.Tensor:
    residual = (prediction[..., :2] - target[..., :2]).abs()
    width = half_width[..., :2].clamp_min(1e-8)
    delta = residual - width
    calibration = torch.maximum(quantile * delta, (quantile - 1.0) * delta).mean()
    sharpness = width.mean()
    return calibration + 0.03 * sharpness


@dataclass(frozen=True)
class LossWeights:
    data: float = 1.0
    tke: float = 0.25
    profile: float = 0.15
    spectral: float = 0.08
    divergence: float = 0.02
    interval: float = 0.03

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> LossWeights:
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})


class MetricAwareLoss(nn.Module):
    def __init__(self, weights: LossWeights) -> None:
        super().__init__()
        self.weights = weights

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        half_width: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        components = {
            "data": relative_l2(prediction, target),
            "tke": tke_relative_l2(prediction, target),
            "profile": profile_relative_l2(prediction, target),
            "spectral": spectral_relative_l1(prediction, target),
            "divergence": divergence_loss(prediction),
        }
        if half_width is not None:
            components["interval"] = interval_quantile_loss(prediction, target, half_width)
        total = sum(getattr(self.weights, name) * value for name, value in components.items())
        detached = {name: float(value.detach()) for name, value in components.items()}
        detached["total"] = float(total.detach())
        return total, detached
