from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .metrics import SIGMA_GLOBAL
from .sps_ordinal import SPSOrdinalConfig, SPSOrdinalUncertaintyOperator


class SPSResidualMagnitudeOperator(SPSOrdinalUncertaintyOperator):
    """Regress the conditional absolute point error without CDF discretization."""

    def __init__(self, config: SPSOrdinalConfig) -> None:
        super().__init__(config)
        self.residual_head = nn.Sequential(
            nn.Conv2d(config.width, config.width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.width, self.out_steps * 2, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.constant_(self.residual_head[-1].bias, -1.0)

    def normalized_width(
        self, observed: torch.Tensor, point: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, scale = self.features(observed, point)
        raw = self.residual_head(hidden)
        normalized = raw.reshape(
            observed.shape[0],
            self.out_steps,
            2,
            observed.shape[2],
            observed.shape[3],
        ).permute(0, 1, 3, 4, 2)
        return F.softplus(normalized.float()) + 1e-4, scale

    def width_for_point(
        self, observed: torch.Tensor, point: torch.Tensor
    ) -> torch.Tensor:
        normalized, scale = self.normalized_width(observed, point)
        width_uv = normalized * scale
        return torch.cat((width_uv, torch.zeros_like(width_uv[..., :1])), dim=-1)


def residual_magnitude_loss(
    model: SPSResidualMagnitudeOperator,
    observed: torch.Tensor,
    point: torch.Tensor,
    target: torch.Tensor,
    values: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    predicted, scale = model.normalized_width(observed, point.detach())
    error = (point[..., :2] - target[..., :2]).abs().float()
    normalized_error = error / scale
    measured = target[..., :2] != 0.0
    epsilon = float(values.get("log_epsilon", 0.01))
    log_regression = F.smooth_l1_loss(
        torch.log(predicted[measured] + epsilon),
        torch.log(normalized_error[measured] + epsilon),
        beta=float(values.get("log_huber_beta", 0.25)),
    )
    residual = normalized_error - predicted
    quantile = float(values.get("quantile", 0.88))
    pinball = torch.where(
        residual >= 0.0, quantile * residual, (quantile - 1.0) * residual
    )[measured].mean()

    physical_width = predicted * scale
    temperature = float(values.get("soft_coverage_temperature", 0.003))
    soft_inside = torch.sigmoid((physical_width - error) / max(temperature, 1e-5))
    soft_reward = soft_inside * torch.exp(-2.0 * physical_width / SIGMA_GLOBAL)
    reward = soft_reward[measured].mean()
    loss = (
        float(values.get("log_regression_weight", 1.0)) * log_regression
        + float(values.get("quantile_weight", 0.5)) * pinball
        - float(values.get("soft_reward_weight", 0.5)) * reward
    )
    return loss, {
        "loss": float(loss.detach()),
        "log_regression": float(log_regression.detach()),
        "pinball": float(pinball.detach()),
        "soft_reward": float(reward.detach()),
        "mean_width": float(physical_width[measured].mean().detach()),
    }


def initialize_residual_from_ordinal(
    checkpoint_path: str | Path, device: torch.device
) -> SPSResidualMagnitudeOperator:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("model_type") != "sps_ordinal_uncertainty":
        raise ValueError(f"not an SPS ordinal checkpoint: {checkpoint_path}")
    model = SPSResidualMagnitudeOperator(
        SPSOrdinalConfig.from_dict(checkpoint["model_config"])
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if set(missing) != {
        "residual_head.0.weight",
        "residual_head.0.bias",
        "residual_head.2.weight",
        "residual_head.2.bias",
    } or unexpected:
        raise RuntimeError(
            f"unexpected ordinal initialization keys: missing={missing}, unexpected={unexpected}"
        )
    model.point.requires_grad_(False)
    model.cdf_head.requires_grad_(False)
    return model


def save_residual_checkpoint(
    path: str | Path,
    model: SPSResidualMagnitudeOperator,
    config: dict[str, Any],
    *,
    step: int,
    metrics: dict[str, Any],
    deployment: dict[str, Any],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "sps_residual_magnitude",
            "model_config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "stage": "fno_tangent_residual",
            "step": int(step),
            "metrics": metrics,
            "deployment": deployment,
            "source_config": {
                key: value for key, value in config.items() if not key.startswith("_")
            },
        },
        output,
    )
    return output


def load_residual_checkpoint(
    path: str | Path, device: torch.device
) -> SPSResidualMagnitudeOperator:
    started = time.perf_counter()
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model_type") != "sps_residual_magnitude":
        raise ValueError(f"not an SPS residual checkpoint: {path}")
    model = SPSResidualMagnitudeOperator(
        SPSOrdinalConfig.from_dict(checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model._checkpoint_load_seconds = time.perf_counter() - started
    return model
