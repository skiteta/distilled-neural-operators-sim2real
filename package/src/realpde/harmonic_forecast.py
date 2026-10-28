from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HarmonicForecastConfig:
    """Configuration for input-only harmonic phase extrapolation."""

    minimum_cycles: float = 0.5
    maximum_cycles: float = 5.0
    cycle_count: int = 91
    harmonic_count: int = 1
    ridge: float = 1e-5

    def __post_init__(self) -> None:
        if self.minimum_cycles <= 0.0:
            raise ValueError("minimum_cycles must be positive")
        if self.maximum_cycles <= self.minimum_cycles:
            raise ValueError("maximum_cycles must exceed minimum_cycles")
        if self.cycle_count < 2:
            raise ValueError("cycle_count must be at least two")
        if self.harmonic_count < 1:
            raise ValueError("harmonic_count must be positive")
        if self.ridge < 0.0:
            raise ValueError("ridge must be non-negative")


@dataclass(frozen=True)
class SpectralKoopmanConfig:
    """Diagonal spatial-Fourier Koopman forecast configuration."""

    modes_y: int = 16
    modes_x: int = 24
    coherence_threshold: float = 0.0
    coherence_power: float = 0.0
    phase_gain: float = 1.0

    def __post_init__(self) -> None:
        if self.modes_y < 1 or self.modes_x < 1:
            raise ValueError("spatial mode counts must be positive")
        if not 0.0 <= self.coherence_threshold <= 1.0:
            raise ValueError("coherence_threshold must lie in [0, 1]")
        if self.coherence_power < 0.0:
            raise ValueError("coherence_power must be non-negative")
        if self.phase_gain <= 0.0:
            raise ValueError("phase_gain must be positive")


def _design(
    times: torch.Tensor,
    cycles: torch.Tensor,
    observation_steps: int,
    harmonic_count: int,
) -> torch.Tensor:
    phase = (
        2.0
        * torch.pi
        * cycles[:, None]
        * times[None, :]
        / float(observation_steps)
    )
    columns = []
    for harmonic in range(1, harmonic_count + 1):
        columns.extend((torch.cos(harmonic * phase), torch.sin(harmonic * phase)))
    return torch.stack(columns, dim=-1)


def _ridge_inverse(design: torch.Tensor, ridge: float) -> torch.Tensor:
    """Return (X'X + ridge I)^-1 X' for each frequency grid point."""
    transpose = design.transpose(-2, -1)
    gram = transpose @ design
    identity = torch.eye(
        gram.shape[-1], device=gram.device, dtype=gram.dtype
    ).expand_as(gram)
    return torch.linalg.solve(gram + float(ridge) * identity, transpose)


@torch.inference_mode()
def harmonic_phase_forecast(
    observed: torch.Tensor,
    *,
    out_steps: int,
    config: HarmonicForecastConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Extrapolate the dominant observed temporal frequency into the future.

    The shared frequency for all spatial locations is chosen independently for
    every sample by minimizing reconstruction error on the observed frames only.
    No future target or regime metadata participates in the decision.
    """
    if observed.ndim != 5 or observed.shape[-1] < 2:
        raise ValueError("observed must have shape (B,T,H,W,C>=2)")
    if observed.shape[1] < 4:
        raise ValueError("at least four observed frames are required")
    if out_steps < 1:
        raise ValueError("out_steps must be positive")

    settings = config or HarmonicForecastConfig()
    velocity = observed[..., :2].float()
    batch, in_steps, height, width, channels = velocity.shape
    flattened = velocity.reshape(batch, in_steps, -1)
    temporal_mean = flattened.mean(dim=1, keepdim=True)
    centered = flattened - temporal_mean

    cycles = torch.linspace(
        settings.minimum_cycles,
        settings.maximum_cycles,
        settings.cycle_count,
        device=velocity.device,
        dtype=velocity.dtype,
    )
    observed_times = torch.arange(
        in_steps, device=velocity.device, dtype=velocity.dtype
    )
    observed_design = _design(
        observed_times, cycles, in_steps, settings.harmonic_count
    )
    inverse = _ridge_inverse(observed_design, settings.ridge)
    coefficients = torch.einsum("kpt,btd->bkpd", inverse, centered)
    reconstruction = torch.einsum(
        "ktp,bkpd->bktd", observed_design, coefficients
    )
    squared_error = (reconstruction - centered[:, None]).square().mean(dim=(2, 3))
    selected_index = squared_error.argmin(dim=1)
    batch_index = torch.arange(batch, device=velocity.device)
    selected_coefficients = coefficients[batch_index, selected_index]

    future_times = torch.arange(
        in_steps, in_steps + out_steps, device=velocity.device, dtype=velocity.dtype
    )
    future_design_grid = _design(
        future_times, cycles, in_steps, settings.harmonic_count
    )
    future_design = future_design_grid[selected_index]
    future_centered = torch.einsum(
        "btp,bpd->btd", future_design, selected_coefficients
    )
    forecast = (future_centered + temporal_mean).reshape(
        batch, out_steps, height, width, channels
    )

    total_variance = centered.square().mean(dim=(1, 2)).clamp_min(1e-12)
    selected_error = squared_error[batch_index, selected_index]
    diagnostics = {
        "selected_cycles": cycles[selected_index],
        "observed_explained_fraction": (1.0 - selected_error / total_variance).clamp(
            max=1.0
        ),
    }
    return forecast.to(dtype=observed.dtype), diagnostics


def blend_harmonic_fluctuation(
    neural_prediction: torch.Tensor,
    harmonic_prediction: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Blend temporal fluctuations while retaining the neural temporal mean."""
    if neural_prediction.shape != harmonic_prediction.shape:
        raise ValueError("neural and harmonic predictions must have equal shapes")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must lie in [0, 1]")
    neural_mean = neural_prediction.mean(dim=1, keepdim=True)
    neural_fluctuation = neural_prediction - neural_mean
    harmonic_fluctuation = harmonic_prediction - harmonic_prediction.mean(
        dim=1, keepdim=True
    )
    return neural_mean + torch.lerp(
        neural_fluctuation, harmonic_fluctuation, float(weight)
    )


def _spatial_mode_mask(
    height: int,
    half_width: int,
    modes_y: int,
    modes_x: int,
    device: torch.device,
) -> torch.Tensor:
    y = torch.arange(height, device=device)
    retained_y = (y < modes_y) | (y >= height - modes_y)
    x = torch.arange(half_width, device=device)
    return retained_y[:, None] & (x[None, :] < modes_x)


@torch.inference_mode()
def spectral_koopman_forecast(
    observed: torch.Tensor,
    *,
    out_steps: int,
    config: SpectralKoopmanConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Forecast transported waves through reversible spatial-mode phase rotation.

    A unit-modulus eigenvalue is estimated from the complex lag-one correlation
    of each spatial Fourier mode. Unit modulus makes the estimate symmetric:
    backward evolution is exactly the inverse of forward evolution.
    """
    if observed.ndim != 5 or observed.shape[-1] < 2:
        raise ValueError("observed must have shape (B,T,H,W,C>=2)")
    if observed.shape[1] < 3:
        raise ValueError("at least three observed frames are required")
    if out_steps < 1:
        raise ValueError("out_steps must be positive")
    settings = config or SpectralKoopmanConfig()

    velocity = observed[..., :2].float()
    batch, _, height, width, channels = velocity.shape
    spectrum = torch.fft.rfft2(velocity, dim=(2, 3), norm="ortho")
    temporal_mean = spectrum.mean(dim=1, keepdim=True)
    centered = spectrum - temporal_mean
    previous = centered[:, :-1]
    following = centered[:, 1:]
    cross = (previous.conj() * following).sum(dim=(1, 4))
    previous_energy = previous.abs().square().sum(dim=(1, 4))
    following_energy = following.abs().square().sum(dim=(1, 4))
    coherence = cross.abs() / (
        previous_energy * following_energy
    ).clamp_min(1e-20).sqrt()
    phase = torch.polar(
        torch.ones_like(cross.real), torch.angle(cross) * settings.phase_gain
    )

    mask = _spatial_mode_mask(
        height,
        spectrum.shape[3],
        min(settings.modes_y, height // 2 + 1),
        min(settings.modes_x, spectrum.shape[3]),
        velocity.device,
    )
    mask = mask[None] & (coherence >= settings.coherence_threshold)
    if settings.coherence_power == 0.0:
        gate = mask.to(velocity.dtype)
    else:
        gate = mask.to(velocity.dtype) * coherence.pow(settings.coherence_power)

    steps = torch.arange(
        1, out_steps + 1, device=velocity.device, dtype=velocity.dtype
    )
    evolution = phase[:, None].pow(steps[None, :, None, None])
    future_centered = (
        centered[:, -1:, :, :, :]
        * evolution[..., None]
        * gate[:, None, :, :, None]
    )
    future_spectrum = temporal_mean + future_centered
    forecast = torch.fft.irfft2(
        future_spectrum,
        s=(height, width),
        dim=(2, 3),
        norm="ortho",
    ).reshape(batch, out_steps, height, width, channels)
    diagnostics = {
        "coherence_mean": coherence.mean(dim=(1, 2)),
        "retained_fraction": gate.gt(0.0).float().mean(dim=(1, 2)),
        "retained_coherence_mean": (
            (coherence * mask).sum(dim=(1, 2))
            / mask.sum(dim=(1, 2)).clamp_min(1)
        ),
    }
    return forecast.to(observed.dtype), diagnostics
