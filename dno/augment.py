"""Data augmentation used in stages S8-S15 (all samples are exact sub-samplings or symmetries of released data).

* decimation phase: the released real fields are 64 x 128; a training window takes every 2nd pixel starting at one of
  the four 2 x 2 phases (the competition inputs use phase (0, 0)).
* temporal cadence: with probability 0.25 the 40 frames are taken every 2nd frame (if the case is long enough).
* vertical reflection: with probability 0.5 the field is flipped vertically and v is negated. Losses are computed
  after mapping the forecast and the interval back to the native geometry, because the MVPE probes are not
  reflection-symmetric.
* reflection-averaged teacher (S13-S15): the distillation target is 0.5 * (T(x) + R^-1 T(R x)).
"""

from __future__ import annotations

import torch

SIGN = (1.0, -1.0, 1.0)  # u, v, p under a vertical flip


def sample_window(flow: torch.Tensor, start: int, phase: int, slow: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Input and target (20, 32, 64, 3) from a released case `flow` (T, 64, 128, 2) of real u, v (pressure = 0)."""
    step = 2 if slow else 1
    if slow and start + 79 > flow.shape[0]:
        raise ValueError("case too short for cadence 2")
    dy, dx = divmod(phase, 2)
    uv = flow[start : start + 40 * step : step, dy::2, dx::2]
    window = torch.cat((uv, torch.zeros_like(uv[..., :1])), dim=-1)
    return window[:20], window[20:]


def reflect(flow: torch.Tensor) -> torch.Tensor:
    """Vertical flip of (N, T, H, W, 3) with v negated (an involution: reflect(reflect(x)) == x)."""
    return flow.flip(2) * flow.new_tensor(SIGN)


def to_native_geometry(prediction, shift, radius, scale, native_input, reflected):
    """Map a forecast and its interval (shift, radius in units of `scale`) from reflected back to native geometry."""
    choose = reflected[:, None, None, None, None]
    native_scale = native_input[..., :2].std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
    shift_native = (shift * scale).flip(2) * shift.new_tensor(SIGN[:2]) / native_scale
    radius_native = (radius * scale).flip(2) / native_scale
    return (
        torch.where(choose, reflect(prediction), prediction),
        torch.where(choose, shift_native, shift),
        torch.where(choose, radius_native, radius),
        torch.where(choose, native_scale, scale),
    )


def reflection_averaged(model, x: torch.Tensor) -> torch.Tensor:
    """0.5 * (f(x) + R^-1 f(R x)) for a forecaster f returning (N, 20, H, W, 3) as its first output."""
    first = model(x)[0]
    return 0.5 * (first + reflect(model(reflect(x))[0]))


def reflection_averaged_interval(center, half_width, center_reflected, half_width_reflected):
    """Average an interval (center vectors, unsigned half-widths) predicted on x and on R x, in the geometry of x."""
    return (
        0.5 * (center + center_reflected.flip(2) * center.new_tensor(SIGN[:2])),
        0.5 * (half_width + half_width_reflected.flip(2)),
    )
