from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from realpde.model import HybridFlowOperator, ModelConfig


class PatchHighFrequencyScaling2d(nn.Module):
    """Patch DC/HFC residual scaling with an exact identity initialization.

    This is a clean reimplementation of the HFS equation described in
    https://arxiv.org/abs/2503.13695 and the MIT-licensed reference at
    https://github.com/SiaK4/HFS_ResUNet (commit 613740c).  For each fixed
    within-patch coordinate, ``dc`` is the mean over all spatial patches and
    ``hfc`` is the corresponding per-patch residual.
    """

    def __init__(self, channels: int, patch_size: int = 4) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.channels = int(channels)
        self.patch_size = int(patch_size)
        parameter_shape = (1, self.channels, 1, 1, 1, 1)
        self.lambda_dc = nn.Parameter(torch.zeros(parameter_shape))
        self.lambda_hfc = nn.Parameter(torch.zeros(parameter_shape))

    def split(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(
                f"expected [B,{self.channels},H,W], got {tuple(features.shape)}"
            )
        batch, channels, height, width = features.shape
        patch = self.patch_size
        if height % patch or width % patch:
            raise ValueError(
                f"height and width must be divisible by patch_size={patch}, "
                f"got {(height, width)}"
            )
        patches = features.reshape(
            batch, channels, height // patch, patch, width // patch, patch
        ).permute(0, 1, 2, 4, 3, 5)
        dc = patches.mean(dim=(2, 3), keepdim=True)
        hfc = patches - dc
        return patches, dc, hfc

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        patches, dc, hfc = self.split(features)
        scaled = patches + self.lambda_dc * dc + self.lambda_hfc * hfc
        batch, channels, patch_rows, patch_columns, patch_y, patch_x = scaled.shape
        return scaled.permute(0, 1, 2, 4, 3, 5).reshape(
            batch,
            channels,
            patch_rows * patch_y,
            patch_columns * patch_x,
        )


class HFSHybridFlowOperator(HybridFlowOperator):
    """HybridFlowOperator challenger with one midpoint HFS module."""

    def __init__(self, config: ModelConfig, patch_size: int = 4) -> None:
        super().__init__(config)
        self.hfs = PatchHighFrequencyScaling2d(config.width, patch_size=patch_size)

    def forward(
        self, flow: torch.Tensor, *, return_interval: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raw = flow
        features, uv_mean, uv_std = self._features(raw)
        hidden = self.lift(features)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden)
            if index + 1 == max(1, len(self.blocks) // 2):
                hidden = self.hfs(self.multiscale(hidden))
        hidden = self.final_norm(hidden)

        batch, _, height, width = hidden.shape
        correction = self.velocity_head(hidden).reshape(
            batch, self.config.out_steps, 2, height, width
        )
        correction = correction.permute(0, 1, 3, 4, 2)
        base_normalized = (raw[:, -1:, ..., :2] - uv_mean) / uv_std
        recent_trend = (
            raw[:, -1:, ..., :2] - raw[:, -5:-4, ..., :2]
        ) / (4 * uv_std)
        trend = recent_trend * self.trend_scale[None, :, None, None, :]
        prediction_uv = (base_normalized + trend + correction) * uv_std + uv_mean
        prediction = torch.cat(
            (prediction_uv, torch.zeros_like(prediction_uv[..., :1])), dim=-1
        )

        if not return_interval:
            return prediction
        raw_width = self.width_head(hidden).reshape(
            batch, self.config.out_steps, 2, height, width
        )
        raw_width = raw_width.permute(0, 1, 3, 4, 2)
        half_width_uv = (
            F.softplus(raw_width)
            * uv_std
            * self.interval_temperature.clamp_min(1e-3)
        )
        half_width = torch.cat(
            (half_width_uv, torch.zeros_like(half_width_uv[..., :1])), dim=-1
        )
        return prediction, half_width

    def predict(
        self, input_array: torch.Tensor, metadata: dict[str, Any] | None = None
    ) -> torch.Tensor:
        del metadata
        return self.forward(input_array)
