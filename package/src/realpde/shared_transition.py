from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .latent_state_refiner import ConvGRUCell
from .model import HybridFlowOperator


@dataclass(frozen=True)
class SharedTransitionConfig:
    width: int = 48
    maximum_normalized_correction: float = 0.35

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SharedTransitionConfig:
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SharedTransitionHybrid(nn.Module):
    """Joint direct proposal and recursive shared-transition correction.

    The direct anchor supplies a strong initial trajectory. A single recurrent
    transition is then applied at every horizon and consumes its own preceding
    corrected state. The correction head is zero initialized, so construction
    preserves the anchor prediction exactly while allowing end-to-end training.
    """

    def __init__(
        self,
        anchor: HybridFlowOperator,
        config: SharedTransitionConfig | None = None,
    ) -> None:
        super().__init__()
        self.anchor = anchor
        self.config = config or SharedTransitionConfig()
        width = self.config.width
        groups = 8 if width % 8 == 0 else 1
        self.initial_encoder = nn.Sequential(
            nn.Conv2d(16, width, 5, stride=2, padding=2),
            nn.GroupNorm(groups, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.Conv2d(width, width, 1),
            nn.GELU(),
        )
        self.source_encoder = nn.Sequential(
            nn.Conv2d(11, width, 5, stride=2, padding=2),
            nn.GroupNorm(groups, width),
            nn.GELU(),
        )
        self.transition = ConvGRUCell(width, width)
        self.state_norm = nn.GroupNorm(groups, width)
        self.correction_head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 2, 1),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    @staticmethod
    def _coordinates(
        batch: int,
        height: int,
        width: int,
        source: torch.Tensor,
    ) -> torch.Tensor:
        yy = torch.linspace(-1.0, 1.0, height, device=source.device, dtype=source.dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=source.device, dtype=source.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=0)[None].expand(batch, -1, -1, -1)

    def forward(
        self,
        flow: torch.Tensor,
        *,
        return_interval: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        anchor_point, half_width = self.anchor(flow, return_interval=True)
        observed = flow[..., :2].float()
        anchor_uv = anchor_point[..., :2].float()
        mean = observed.mean(dim=(1, 2, 3), keepdim=True)
        std = observed.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized_observed = (observed - mean) / std
        normalized_anchor = (anchor_uv - mean) / std

        previous_observed = torch.cat(
            (normalized_observed[:, :1], normalized_observed[:, :-1]),
            dim=1,
        )
        observed_delta = normalized_observed - previous_observed
        initial = torch.cat(
            (
                normalized_observed[:, -4:].reshape(flow.shape[0], flow.shape[2], flow.shape[3], 8),
                observed_delta[:, -4:].reshape(flow.shape[0], flow.shape[2], flow.shape[3], 8),
            ),
            dim=-1,
        ).permute(0, 3, 1, 2)
        state = self.initial_encoder(initial)
        coordinates = self._coordinates(flow.shape[0], flow.shape[2], flow.shape[3], observed)

        previous = normalized_observed[:, -1]
        previous_anchor = previous
        corrected = []
        latent_height, latent_width = state.shape[-2:]
        for index in range(normalized_anchor.shape[1]):
            time = (index + 1.0) / normalized_anchor.shape[1]
            phase = 2.0 * math.pi * time
            time_features = observed.new_tensor((time, math.sin(phase), math.cos(phase)))
            time_features = time_features[None, :, None, None].expand(
                flow.shape[0], -1, flow.shape[2], flow.shape[3]
            )
            proposal = normalized_anchor[:, index]
            source = torch.cat(
                (
                    proposal.permute(0, 3, 1, 2),
                    previous.permute(0, 3, 1, 2),
                    (proposal - previous_anchor).permute(0, 3, 1, 2),
                    coordinates,
                    time_features,
                ),
                dim=1,
            )
            encoded = self.source_encoder(source)
            if encoded.shape[-2:] != (latent_height, latent_width):
                raise RuntimeError("transition source and recurrent state shapes differ")
            state = self.transition(encoded, state)
            correction = self.correction_head(self.state_norm(state))
            correction = F.interpolate(
                correction,
                size=(flow.shape[2], flow.shape[3]),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            correction = self.config.maximum_normalized_correction * torch.tanh(correction)
            prediction = proposal + correction
            corrected.append(prediction)
            previous = prediction
            previous_anchor = proposal

        normalized_prediction = torch.stack(corrected, dim=1)
        prediction_uv = normalized_prediction * std + mean
        pressure = torch.zeros_like(prediction_uv[..., :1])
        prediction = torch.cat((prediction_uv, pressure), dim=-1)
        if return_interval:
            return prediction, half_width
        return prediction
