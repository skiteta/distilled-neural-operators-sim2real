from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RawHistoryEnergyHeadConfig:
    summary_channels: int = 34
    history_width: int = 24
    width: int = 48
    dropout: float = 0.02
    output_bound: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Residual3D(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, values: Tensor) -> Tensor:
        return values + self.block(values)


class Residual2D(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, values: Tensor) -> Tensor:
        return values + self.block(values)


class RawHistoryHighFrequencyEnergyHead(nn.Module):
    def __init__(self, config: RawHistoryEnergyHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or RawHistoryEnergyHeadConfig()
        h = self.config.history_width
        w = self.config.width
        self.history_stem = nn.Conv3d(4, h, 3, padding=1)
        self.history_encoder = nn.Sequential(
            Residual3D(h, self.config.dropout),
            nn.Conv3d(h, h, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
            Residual3D(h, self.config.dropout),
            Residual3D(h, self.config.dropout),
        )
        # Three temporal summaries, plus two fixed spatial high-pass views of the final summary.
        combined_channels = self.config.summary_channels + 5 * h
        self.stem = nn.Conv2d(combined_channels, w, 3, padding=1)
        self.enc0 = nn.Sequential(Residual2D(w, self.config.dropout), Residual2D(w, self.config.dropout))
        self.down0 = nn.Conv2d(w, 2 * w, 4, stride=2, padding=1)
        self.enc1 = nn.Sequential(Residual2D(2 * w, self.config.dropout), Residual2D(2 * w, self.config.dropout))
        self.down1 = nn.Conv2d(2 * w, 4 * w, 4, stride=2, padding=1)
        self.middle = nn.Sequential(Residual2D(4 * w, self.config.dropout), Residual2D(4 * w, self.config.dropout))
        self.up1 = nn.Conv2d(6 * w, 2 * w, 3, padding=1)
        self.dec1 = nn.Sequential(Residual2D(2 * w, self.config.dropout), Residual2D(2 * w, self.config.dropout))
        self.up0 = nn.Conv2d(3 * w, w, 3, padding=1)
        self.dec0 = nn.Sequential(Residual2D(w, self.config.dropout), Residual2D(w, self.config.dropout))
        self.out = nn.Conv2d(w, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _highpass(values: Tensor, kernel: int) -> Tensor:
        smooth = F.avg_pool2d(
            values, kernel, stride=1, padding=kernel // 2, count_include_pad=False
        )
        return values - smooth

    def forward(self, summary: Tensor, history: Tensor) -> Tensor:
        # history is [batch,time,height,width,2], already normalized in velocity units.
        velocity = history.permute(0, 4, 1, 2, 3)
        difference = torch.zeros_like(velocity)
        difference[:, :, 1:] = velocity[:, :, 1:] - velocity[:, :, :-1]
        latent = self.history_encoder(self.history_stem(torch.cat((velocity, difference), dim=1)))
        temporal_mean = latent.mean(dim=2)
        final = latent[:, :, -1]
        width = max(1, latent.shape[2] // 3)
        trend = latent[:, :, -width:].mean(dim=2) - latent[:, :, :width].mean(dim=2)
        combined = torch.cat(
            (
                summary,
                temporal_mean,
                final,
                trend,
                self._highpass(final, 3),
                self._highpass(final, 7),
            ),
            dim=1,
        )
        x0 = self.enc0(self.stem(combined))
        x1 = self.enc1(self.down0(x0))
        middle = self.middle(self.down1(x1))
        up1 = F.interpolate(middle, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.dec1(self.up1(torch.cat((up1, x1), dim=1)))
        up0 = F.interpolate(up1, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        up0 = self.dec0(self.up0(torch.cat((up0, x0), dim=1)))
        return self.config.output_bound * torch.tanh(self.out(up0)[:, 0])
