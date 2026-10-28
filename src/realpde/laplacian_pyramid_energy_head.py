"""Temporal-phase Laplacian-pyramid log-energy residual generator for C489."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LaplacianPyramidEnergyConfig:
    summary_channels: int = 34
    block_state_channels: int = 20
    temporal_modes: int = 6
    temporal_mode_channels: int = 24
    input_channels: int = 78
    full_width: int = 64
    mid_width: int = 128
    low_width: int = 192
    dropout: float = 0.02
    low_bound: float = 0.7
    mid_bound: float = 0.7
    high_bound: float = 1.0
    total_bound: float = 2.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class PyramidResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, width), width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(min(8, width), width),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(width, width, 3, padding=1, padding_mode="circular"),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.net(value)


def circular_mean(value: Tensor, kernel: int) -> Tensor:
    padding = kernel // 2
    return F.avg_pool2d(F.pad(value, (padding, padding, padding, padding), mode="circular"), kernel, stride=1)


class LaplacianPyramidEnergyHead(nn.Module):
    def __init__(self, config: LaplacianPyramidEnergyConfig = LaplacianPyramidEnergyConfig()):
        super().__init__()
        self.config = config
        self.fourier_lift = nn.Conv2d(config.temporal_mode_channels, 32, 1)
        combined = config.summary_channels + config.block_state_channels + 32
        self.full_lift = nn.Conv2d(combined, config.full_width, 5, padding=2, padding_mode="circular")
        self.full_blocks = nn.Sequential(
            PyramidResidualBlock(config.full_width, config.dropout),
            PyramidResidualBlock(config.full_width, config.dropout),
        )
        self.down_mid = nn.Conv2d(
            config.full_width, config.mid_width, 4, stride=2, padding=1, padding_mode="circular"
        )
        self.mid_blocks = nn.Sequential(
            PyramidResidualBlock(config.mid_width, config.dropout),
            PyramidResidualBlock(config.mid_width, config.dropout),
        )
        self.down_low = nn.Conv2d(
            config.mid_width, config.low_width, 4, stride=2, padding=1, padding_mode="circular"
        )
        self.low_blocks = nn.Sequential(
            PyramidResidualBlock(config.low_width, config.dropout),
            PyramidResidualBlock(config.low_width, config.dropout),
            PyramidResidualBlock(config.low_width, config.dropout),
        )
        self.low_head = nn.Conv2d(config.low_width, 1, 1)
        self.mid_head = nn.Conv2d(config.mid_width, 1, 1)
        self.high_head = nn.Conv2d(config.full_width, 1, 1)
        for head in (self.low_head, self.mid_head, self.high_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def temporal_phase_features(self, history: Tensor) -> Tensor:
        if history.ndim != 5 or history.shape[1:] != (20, 32, 64, 2):
            raise ValueError(f"unexpected C489 history {list(history.shape)}")
        spectrum = torch.fft.rfft(history.float(), dim=1, norm="ortho")[:, : self.config.temporal_modes]
        modes = torch.view_as_real(spectrum).permute(0, 1, 4, 5, 2, 3).contiguous()
        return modes.flatten(1, 3)

    def forward_components(self, summary: Tensor, block_states: Tensor, history: Tensor) -> dict[str, Tensor]:
        if summary.shape[1:] != (self.config.summary_channels, 32, 64):
            raise ValueError(f"unexpected C489 summary {list(summary.shape)}")
        if block_states.shape[1:] != (4, 5, 32, 64):
            raise ValueError(f"unexpected C489 block states {list(block_states.shape)}")
        temporal = self.temporal_phase_features(history)
        fourier = F.silu(self.fourier_lift(temporal))
        block = block_states.flatten(1, 2)
        full = self.full_blocks(self.full_lift(torch.cat((summary, block, fourier), dim=1)))
        mid = self.mid_blocks(self.down_mid(full))
        low = self.low_blocks(self.down_low(mid))
        low_raw = F.interpolate(self.low_head(low), size=(32, 64), mode="bilinear", align_corners=False)
        mid_raw = F.interpolate(self.mid_head(mid), size=(32, 64), mode="bilinear", align_corners=False)
        high_raw = self.high_head(full)
        low_band = self.config.low_bound * torch.tanh(low_raw)
        mid_band = self.config.mid_bound * torch.tanh(mid_raw - circular_mean(mid_raw, 15))
        high_band = self.config.high_bound * torch.tanh(high_raw - circular_mean(high_raw, 3))
        total = (low_band + mid_band + high_band).clamp(-self.config.total_bound, self.config.total_bound)
        return {
            "low": low_band[:, 0],
            "mid": mid_band[:, 0],
            "high": high_band[:, 0],
            "total": total[:, 0],
            "temporal_modes": temporal,
            "fourier_latent": fourier,
            "full_latent": full,
            "mid_latent": mid,
            "low_latent": low,
        }

    def forward(self, summary: Tensor, block_states: Tensor, history: Tensor) -> Tensor:
        return self.forward_components(summary, block_states, history)["total"]
