from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class SignedIntervalConfig:
    feature_channels: int = 96
    hidden_channels: int = 96
    out_steps: int = 20
    maximum_normalized_shift: float = 0.5

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SignedIntervalConfig:
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SignedIntervalCenterHead(nn.Module):
    """Predict the signed interval-center residual without moving the point."""

    def __init__(self, config: SignedIntervalConfig) -> None:
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Conv2d(
                config.feature_channels,
                config.hidden_channels,
                3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(config.hidden_channels, config.out_steps * 2, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("features must have shape (N,C,H,W)")
        if features.shape[1] != self.config.feature_channels:
            raise ValueError(
                f"expected {self.config.feature_channels} feature channels, "
                f"got {features.shape[1]}"
            )
        raw = self.network(features)
        batch, _, height, width = raw.shape
        shift = raw.reshape(
            batch, self.config.out_steps, 2, height, width
        ).permute(0, 1, 3, 4, 2)
        return torch.tanh(shift) * float(self.config.maximum_normalized_shift)


@dataclass(frozen=True)
class SignedIntervalDecisionConfig(SignedIntervalConfig):
    maximum_log_width_scale: float = 0.6931471805599453


class SignedIntervalDecisionHead(nn.Module):
    """Joint Bayes decision for signed center and positive interval radius."""

    def __init__(self, config: SignedIntervalDecisionConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = nn.Sequential(
            nn.Conv2d(
                config.feature_channels,
                config.hidden_channels,
                3,
                padding=1,
            ),
            nn.GELU(),
        )
        channels = config.out_steps * 2
        self.center_head = nn.Conv2d(config.hidden_channels, channels, 1)
        self.log_width_head = nn.Conv2d(config.hidden_channels, channels, 1)
        for layer in (self.center_head, self.log_width_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _reshape(self, raw: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = raw.shape
        return raw.reshape(
            batch, self.config.out_steps, 2, height, width
        ).permute(0, 1, 3, 4, 2)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError("features must have shape (N,C,H,W)")
        if features.shape[1] != self.config.feature_channels:
            raise ValueError(
                f"expected {self.config.feature_channels} feature channels, "
                f"got {features.shape[1]}"
            )
        hidden = self.trunk(features)
        center = torch.tanh(self._reshape(self.center_head(hidden)))
        center = center * float(self.config.maximum_normalized_shift)
        log_width = torch.tanh(self._reshape(self.log_width_head(hidden)))
        log_width = log_width * float(self.config.maximum_log_width_scale)
        return center, log_width.exp()


@dataclass(frozen=True)
class SignedQuantileConfig(SignedIntervalConfig):
    quantiles: tuple[float, ...] = (
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
        0.70,
        0.80,
        0.90,
        0.95,
    )
    maximum_normalized_residual: float = 3.0

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SignedQuantileConfig:
        allowed = cls.__dataclass_fields__.keys()
        filtered = {key: value for key, value in values.items() if key in allowed}
        if "quantiles" in filtered:
            filtered["quantiles"] = tuple(float(item) for item in filtered["quantiles"])
        return cls(**filtered)


class SignedQuantileIntervalHead(nn.Module):
    """Conditional signed residual quantiles for asymmetric SPS intervals."""

    def __init__(self, config: SignedQuantileConfig) -> None:
        super().__init__()
        self.config = config
        if len(config.quantiles) < 2 or any(
            not 0.0 < value < 1.0 for value in config.quantiles
        ):
            raise ValueError("quantiles must contain at least two values in (0,1)")
        if tuple(sorted(config.quantiles)) != config.quantiles:
            raise ValueError("quantiles must be increasing")
        self.network = nn.Sequential(
            nn.Conv2d(
                config.feature_channels,
                config.hidden_channels,
                3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                config.hidden_channels,
                config.out_steps * 2 * len(config.quantiles),
                1,
            ),
        )
        nn.init.zeros_(self.network[-1].weight)
        with torch.no_grad():
            initial = torch.tensor(config.quantiles).mul(2.0).sub(1.0).mul(0.25)
            maximum = float(config.maximum_normalized_residual)
            bias = torch.atanh((initial / maximum).clamp(-0.99, 0.99))
            repeated = bias.repeat(config.out_steps * 2)
            self.network[-1].bias.copy_(repeated)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("features must have shape (N,C,H,W)")
        if features.shape[1] != self.config.feature_channels:
            raise ValueError(
                f"expected {self.config.feature_channels} feature channels, "
                f"got {features.shape[1]}"
            )
        raw = self.network(features)
        batch, _, height, width = raw.shape
        values = raw.reshape(
            batch,
            self.config.out_steps,
            2,
            len(self.config.quantiles),
            height,
            width,
        ).permute(0, 1, 4, 5, 2, 3)
        values = torch.tanh(values) * float(self.config.maximum_normalized_residual)
        return values.sort(dim=-1).values
