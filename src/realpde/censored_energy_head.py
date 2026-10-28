from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CensoredEnergyHeadConfig:
    in_channels: int = 34
    width: int = 32
    dropout: float = 0.02
    central_bound: float = 0.7
    excess_bound: float = 1.3
    output_bound: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Residual2D(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        groups = min(8, channels)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, values: Tensor) -> Tensor:
        return values + self.block(values)


class CensoredHighFrequencyEnergyHead(nn.Module):
    """Full-resolution plus U-Net generator with an explicit censored decomposition."""

    def __init__(self, config: CensoredEnergyHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or CensoredEnergyHeadConfig()
        w = self.config.width
        self.full = nn.Sequential(
            nn.Conv2d(self.config.in_channels, w, 3, padding=1),
            Residual2D(w, self.config.dropout),
            Residual2D(w, self.config.dropout),
        )
        self.stem = nn.Conv2d(self.config.in_channels, w, 3, padding=1)
        self.enc0 = nn.Sequential(Residual2D(w, self.config.dropout), Residual2D(w, self.config.dropout))
        self.down0 = nn.Conv2d(w, 2 * w, 4, stride=2, padding=1)
        self.enc1 = nn.Sequential(Residual2D(2 * w, self.config.dropout), Residual2D(2 * w, self.config.dropout))
        self.down1 = nn.Conv2d(2 * w, 4 * w, 4, stride=2, padding=1)
        self.middle = nn.Sequential(Residual2D(4 * w, self.config.dropout), Residual2D(4 * w, self.config.dropout))
        self.up1 = nn.Conv2d(6 * w, 2 * w, 3, padding=1)
        self.dec1 = nn.Sequential(Residual2D(2 * w, self.config.dropout), Residual2D(2 * w, self.config.dropout))
        self.up0 = nn.Conv2d(3 * w, w, 3, padding=1)
        self.dec0 = nn.Sequential(Residual2D(w, self.config.dropout), Residual2D(w, self.config.dropout))
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * w, w, 3, padding=1),
            Residual2D(w, self.config.dropout),
        )
        self.out = nn.Conv2d(w, 6, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        full = self.full(features)
        x0 = self.enc0(self.stem(features))
        x1 = self.enc1(self.down0(x0))
        middle = self.middle(self.down1(x1))
        up1 = F.interpolate(middle, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.dec1(self.up1(torch.cat((up1, x1), dim=1)))
        up0 = F.interpolate(up1, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        up0 = self.dec0(self.up0(torch.cat((up0, x0), dim=1)))
        raw = self.out(self.fuse(torch.cat((up0, full), dim=1)))
        central = self.config.central_bound * torch.tanh(raw[:, 0])
        class_logits = raw[:, 1:4]
        class_probability = class_logits.softmax(dim=1)
        positive_excess = self.config.excess_bound * torch.sigmoid(raw[:, 4])
        negative_excess = self.config.excess_bound * torch.sigmoid(raw[:, 5])
        log_amplitude = central + class_probability[:, 2] * positive_excess
        log_amplitude = log_amplitude - class_probability[:, 0] * negative_excess
        log_amplitude = log_amplitude.clamp(-self.config.output_bound, self.config.output_bound)
        return {
            "log_amplitude": log_amplitude,
            "central": central,
            "class_logits": class_logits,
            "class_probability": class_probability,
            "positive_excess": positive_excess,
            "negative_excess": negative_excess,
        }
