from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FNOFrequencyReadoutConfig:
    steps: int = 20
    hidden_channels: int = 128
    cutoff: int = 5

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> FNOFrequencyReadoutConfig:
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fno_forward_with_hidden(
    model: nn.Module, inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the organizer FNO once and expose its existing 128-D readout features."""
    grid = model.get_grid(inputs.shape, inputs.device)
    hidden = model.fc0(torch.cat((inputs, grid), dim=-1))
    hidden = hidden.permute(0, 4, 1, 2, 3)
    padding = int(model.padding)
    hidden = F.pad(hidden, [0, padding, 0, padding, 0, padding])
    for index in range(int(model.n_layers)):
        spectral = model.spectral_convs[index](hidden)
        local = model.convs[index](hidden)
        hidden = model.bns[index](spectral + local)
        if index < int(model.n_layers) - 1:
            hidden = F.gelu(hidden)
    hidden = hidden[..., :-padding, :-padding, :-padding]
    hidden = hidden.permute(0, 2, 3, 4, 1)
    readout_features = F.gelu(model.fc1(hidden))
    projected = model.fc2(readout_features)
    projected = projected.reshape(
        *projected.shape[:-1],
        model.shape_out[-1],
        model.shape_out[0] // model.shape_in[0],
    )
    prediction = projected.permute(0, 1, 5, 2, 3, 4).reshape(
        projected.shape[0], *model.shape_out
    )
    return prediction, readout_features


class FNOFrequencyReadout(nn.Module):
    """A 258-parameter band-limited residual head on frozen FNO features."""

    def __init__(self, config: FNOFrequencyReadoutConfig) -> None:
        super().__init__()
        if config.cutoff < 0 or config.cutoff >= config.steps // 2:
            raise ValueError(
                f"cutoff must be in [0,{config.steps // 2 - 1}], got {config.cutoff}"
            )
        self.config = config
        self.output = nn.Linear(config.hidden_channels, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        time = torch.arange(config.steps, dtype=torch.float32)
        difference = time[:, None] - time[None, :]
        projector = torch.full(
            (config.steps, config.steps), 1.0 / config.steps, dtype=torch.float32
        )
        for mode in range(1, config.cutoff + 1):
            projector += (2.0 / config.steps) * torch.cos(
                2.0 * math.pi * mode * difference / config.steps
            )
        self.register_buffer("temporal_projector", projector, persistent=False)

    def project(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 5 or field.shape[1] != self.config.steps:
            raise ValueError(
                f"expected [B,{self.config.steps},H,W,C], got {tuple(field.shape)}"
            )
        projector = self.temporal_projector.to(dtype=field.dtype)
        return torch.einsum("ts,bshwc->bthwc", projector, field)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.project(self.output(features))


def save_fno_frequency_readout(
    path: str,
    model: FNOFrequencyReadout,
    *,
    ridge_ratio: float,
    correction_scale: float,
    source_config: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    torch.save(
        {
            "format_version": 1,
            "model_type": "fno_frequency_readout",
            "model_config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "ridge_ratio": float(ridge_ratio),
            "correction_scale": float(correction_scale),
            "source_config": source_config,
            "validation": validation,
        },
        path,
    )


def load_fno_frequency_readout(
    path: str, device: torch.device
) -> FNOFrequencyReadout:
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model_type") != "fno_frequency_readout":
        raise ValueError(f"not an FNO frequency readout checkpoint: {path}")
    model = FNOFrequencyReadout(
        FNOFrequencyReadoutConfig.from_dict(checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model
