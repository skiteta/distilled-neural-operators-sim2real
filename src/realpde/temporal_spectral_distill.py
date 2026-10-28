from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from realpde.temporal_spectral_flow import project_local_tke


@dataclass(frozen=True)
class TemporalSpectralDistillConfig:
    """Fixed low-cost temporal-frequency transfer learned from a teacher."""

    out_steps: int = 20
    tile_y: int = 4
    tile_x: int = 4
    feature_count: int = 5

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TemporalSpectralDistillConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def temporal_spectral_features(
    observed: torch.Tensor,
    student: torch.Tensor,
    baseline: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized temporal spectra and the physical velocity scale.

    The five complex predictors are observed u/v, the student's u/v correction
    from the baseline, and a frequency-specific intercept.  Keeping the feature
    family this small makes the fitted transfer identifiable across regimes.
    """

    observed_uv = observed[..., :2].float()
    mean = observed_uv.mean(dim=(1, 2, 3), keepdim=True)
    scale = observed_uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
    observed_normalized = (observed_uv - mean) / scale
    student_delta = (student[..., :2].float() - baseline[..., :2].float()) / scale
    inputs = torch.cat((observed_normalized, student_delta), dim=-1)
    spectrum = torch.fft.rfft(inputs, dim=1, norm="ortho")
    intercept = torch.ones_like(spectrum[..., :1])
    return torch.cat((spectrum, intercept), dim=-1), scale


class TiledTemporalSpectralTransfer(nn.Module):
    """Piecewise-spatial complex linear map over temporal Fourier modes."""

    def __init__(
        self,
        coefficients: torch.Tensor,
        config: TemporalSpectralDistillConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or TemporalSpectralDistillConfig()
        expected = (
            self.config.out_steps // 2 + 1,
            32 // self.config.tile_y,
            64 // self.config.tile_x,
            self.config.feature_count,
            2,
        )
        if tuple(coefficients.shape) != expected or not coefficients.is_complex():
            raise ValueError(
                f"expected complex coefficients {expected}, got {tuple(coefficients.shape)}"
            )
        self.register_buffer("coefficients", coefficients.to(torch.complex64))

    def forward(
        self,
        observed: torch.Tensor,
        student: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        features, scale = temporal_spectral_features(observed, student, baseline)
        batch, frequencies, height, width, predictors = features.shape
        tile_y = self.config.tile_y
        tile_x = self.config.tile_x
        tiled = features.reshape(
            batch,
            frequencies,
            height // tile_y,
            tile_y,
            width // tile_x,
            tile_x,
            predictors,
        )
        correction_spectrum = torch.einsum(
            "bfiajcp,fijpo->bfiajco", tiled, self.coefficients
        ).reshape(batch, frequencies, height, width, 2)
        correction = torch.fft.irfft(
            correction_spectrum,
            n=self.config.out_steps,
            dim=1,
            norm="ortho",
        )
        candidate_uv = student[..., :2].float() + correction * scale
        candidate = torch.cat(
            (candidate_uv, torch.zeros_like(candidate_uv[..., :1])), dim=-1
        )
        return project_local_tke(candidate, baseline)

