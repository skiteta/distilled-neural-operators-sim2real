from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from realpde.model import HybridFlowOperator


@dataclass(frozen=True)
class TKEFoldedSpecialistConfig:
    head_width: int = 32
    maximum_log_amplitude: float = 0.7

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TKEFoldedSpecialistConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TKEFoldedSpecialist(nn.Module):
    """Reuse a frozen specialist backbone and predict only a TKE amplitude map."""

    def __init__(
        self,
        specialist: HybridFlowOperator,
        config: TKEFoldedSpecialistConfig | None = None,
    ) -> None:
        super().__init__()
        self.specialist = specialist
        self.config = config or TKEFoldedSpecialistConfig()
        width = specialist.config.width
        self.tke_head = nn.Sequential(
            nn.Conv2d(width, self.config.head_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.config.head_width, 1, 1),
        )
        nn.init.zeros_(self.tke_head[-1].weight)
        nn.init.zeros_(self.tke_head[-1].bias)
        self.specialist.requires_grad_(False)
        self.specialist.eval()

    def train(self, mode: bool = True) -> "TKEFoldedSpecialist":
        super().train(mode)
        self.specialist.eval()
        return self

    def components(
        self, flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.specialist
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
        log_amplitude = self.config.maximum_log_amplitude * torch.tanh(
            self.tke_head(hidden)[:, 0]
        )
        return prediction, log_amplitude

    @staticmethod
    def apply_amplitude(
        raw_blend: torch.Tensor, log_amplitude: torch.Tensor
    ) -> torch.Tensor:
        uv = raw_blend[..., :2].float()
        temporal_mean = uv.mean(dim=1, keepdim=True)
        fluctuation = uv - temporal_mean
        scaled_uv = temporal_mean + fluctuation * torch.exp(log_amplitude)[
            :, None, :, :, None
        ]
        return torch.cat(
            (scaled_uv, torch.zeros_like(scaled_uv[..., :1])), dim=-1
        )

