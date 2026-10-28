from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LatentStateRefinerConfig:
    latent_width: int = 64
    out_steps: int = 20
    dropout: float = 0.01
    maximum_mean_correction: float = 0.25
    maximum_fluctuation_correction: float = 0.25

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "LatentStateRefinerConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConvGRUCell(nn.Module):
    def __init__(self, input_width: int, hidden_width: int) -> None:
        super().__init__()
        total = input_width + hidden_width
        self.gates = nn.Conv2d(total, 2 * hidden_width, 3, padding=1)
        self.candidate = nn.Conv2d(total, hidden_width, 3, padding=1)

    def forward(self, source: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        joined = torch.cat((source, state), dim=1)
        reset, update = self.gates(joined).chunk(2, dim=1)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update)
        candidate = torch.tanh(self.candidate(torch.cat((source, reset * state), dim=1)))
        return torch.lerp(state, candidate, update)


class LatentStateRefiner(nn.Module):
    """Assimilate observed frames, autonomously evolve state, and refine an anchor."""

    def __init__(self, config: LatentStateRefinerConfig | None = None) -> None:
        super().__init__()
        self.config = config or LatentStateRefinerConfig()
        width = self.config.latent_width
        groups32 = 8 if 32 % 8 == 0 else 1
        groups = 8 if width % 8 == 0 else 1
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 5, stride=2, padding=2),
            nn.GroupNorm(groups32, 32),
            nn.GELU(),
            nn.Conv2d(32, width, 3, stride=2, padding=1),
            nn.GroupNorm(groups, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.Conv2d(width, width, 1),
            nn.GELU(),
        )
        self.assimilator = ConvGRUCell(width, width)
        self.transition = ConvGRUCell(3, width)
        self.state_norm = nn.GroupNorm(groups, width)
        self.mean_decoder = self._decoder(width)
        self.fluctuation_decoder = self._decoder(width)
        nn.init.zeros_(self.mean_decoder[-1].weight)
        nn.init.zeros_(self.mean_decoder[-1].bias)
        nn.init.zeros_(self.fluctuation_decoder[-1].weight)
        nn.init.zeros_(self.fluctuation_decoder[-1].bias)

    @staticmethod
    def _decoder(width: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 2, 1),
        )

    @staticmethod
    def _coordinates(batch: int, height: int, width: int, source: torch.Tensor):
        yy = torch.linspace(-1.0, 1.0, height, device=source.device, dtype=source.dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=source.device, dtype=source.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=0)[None].expand(batch, -1, -1, -1)

    def _decode(self, decoder: nn.Module, state: torch.Tensor, size: tuple[int, int]):
        value = decoder(self.state_norm(state))
        return F.interpolate(value, size=size, mode="bilinear", align_corners=False)

    def forward(self, observed: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        if observed.ndim != 5 or observed.shape[1:] != (20, 32, 64, 3):
            raise ValueError(f"unexpected observed shape {tuple(observed.shape)}")
        if baseline.shape != observed.shape:
            raise ValueError(f"unexpected baseline shape {tuple(baseline.shape)}")
        uv = observed[..., :2].float()
        base = baseline[..., :2].float()
        mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        std = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized = (uv - mean) / std
        previous = torch.cat((normalized[:, :1], normalized[:, :-1]), dim=1)
        temporal_delta = normalized - previous
        batch, steps, height, width, _ = normalized.shape
        coordinates = self._coordinates(batch, height, width, normalized)
        encoded = []
        for index in range(steps):
            frame = torch.cat(
                (
                    normalized[:, index].permute(0, 3, 1, 2),
                    temporal_delta[:, index].permute(0, 3, 1, 2),
                    coordinates,
                ),
                dim=1,
            )
            encoded.append(self.encoder(frame))
        state = torch.zeros_like(encoded[0])
        for frame in encoded:
            state = self.assimilator(frame, state)

        mean_raw = self._decode(self.mean_decoder, state, (height, width))
        mean_correction = self.config.maximum_mean_correction * torch.tanh(mean_raw)
        fluctuation_raw = []
        latent_height, latent_width = state.shape[-2:]
        for index in range(self.config.out_steps):
            time = (index + 1.0) / self.config.out_steps
            phase = 2.0 * math.pi * time
            time_features = state.new_tensor((time, math.sin(phase), math.cos(phase)))
            time_features = time_features[None, :, None, None].expand(
                batch, -1, latent_height, latent_width
            )
            state = self.transition(time_features, state)
            fluctuation_raw.append(
                self._decode(self.fluctuation_decoder, state, (height, width))
            )
        fluctuation = torch.stack(fluctuation_raw, dim=1)
        fluctuation = self.config.maximum_fluctuation_correction * torch.tanh(fluctuation)
        fluctuation = fluctuation - fluctuation.mean(dim=1, keepdim=True)
        correction = mean_correction[:, None] + fluctuation
        refined_uv = base + correction.permute(0, 1, 3, 4, 2) * std
        return torch.cat((refined_uv, torch.zeros_like(refined_uv[..., :1])), dim=-1)

