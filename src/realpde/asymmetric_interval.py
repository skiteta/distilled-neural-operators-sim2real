from __future__ import annotations

import torch


def shifted_interval_bounds(
    point: torch.Tensor,
    half_width: torch.Tensor,
    center_shift: torch.Tensor,
    *,
    width_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an interval whose center may differ from the point forecast.

    The official v9 SPS contract only requires finite ordered bounds.  It does
    not require the deterministic point prediction to be the interval center
    (or even to lie inside it), so keeping these two decisions separate avoids
    an unnecessary restriction on the Bayes interval decision.
    """
    if point.shape != half_width.shape:
        raise ValueError("point and half_width must have identical shapes")
    if center_shift.shape != point.shape and not all(
        shift == 1 or shift == value
        for shift, value in zip(center_shift.shape, point.shape, strict=True)
    ):
        raise ValueError("center_shift must match or broadcast over point")
    if not torch.isfinite(point).all() or not torch.isfinite(half_width).all():
        raise ValueError("point and half_width must be finite")
    if not torch.isfinite(center_shift).all():
        raise ValueError("center_shift must be finite")
    if torch.any(half_width < 0.0):
        raise ValueError("half_width must be non-negative")
    if width_scale < 0.0:
        raise ValueError("width_scale must be non-negative")
    radius = half_width * float(width_scale)
    center = point + center_shift
    return center - radius, center + radius


def _finite_median(values: torch.Tensor, dim: int) -> torch.Tensor:
    result = torch.nanmedian(values, dim=dim).values
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def fit_residual_location(
    signed_residual: torch.Tensor,
    measured: torch.Tensor,
    family: str,
) -> torch.Tensor:
    """Estimate a robust signed residual location without future metadata.

    Inputs have shape ``(sample, time, y, x, channel)``.  The returned tensor
    has a leading singleton sample axis and is directly broadcastable at
    inference.  Unmeasured airfoil cells are ignored and receive zero shift.
    """
    if signed_residual.ndim != 5 or signed_residual.shape != measured.shape:
        raise ValueError("residual and measured must share shape (N,T,H,W,C)")
    if signed_residual.shape[-1] != 2:
        raise ValueError("signed residual must contain exactly u/v channels")
    values = signed_residual.masked_fill(~measured, float("nan"))
    samples, steps, height, width, channels = values.shape

    global_location = _finite_median(values.reshape(-1, channels), dim=0).reshape(
        1, 1, 1, 1, channels
    )
    if family == "global":
        return global_location

    horizon = _finite_median(
        values.permute(1, 0, 2, 3, 4).reshape(steps, -1, channels), dim=1
    ).reshape(1, steps, 1, 1, channels)
    if family == "horizon":
        return horizon

    spatial = _finite_median(
        values.permute(2, 3, 0, 1, 4).reshape(height, width, -1, channels),
        dim=2,
    ).reshape(1, 1, height, width, channels)
    if family == "spatial":
        return spatial

    additive = horizon + spatial - global_location
    if family == "additive":
        return additive

    element = _finite_median(values, dim=0).unsqueeze(0)
    if family == "element":
        return element
    if family == "hierarchical":
        return 0.5 * element + 0.5 * additive
    raise ValueError(f"unknown residual-location family: {family}")
