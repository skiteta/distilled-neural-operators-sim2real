from __future__ import annotations

from dataclasses import dataclass

import torch

EPS = 1e-8
SIGMA_GLOBAL = 0.0563870259
SOLVER_REFERENCE_SECONDS = 0.72896
INFERRED_COMPOSITE_POWER = 0.6854
INFERRED_COMPOSITE_WEIGHTS = (0.5, 0.1, 0.1, 0.1, 0.2)


def _uv(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[-1] < 2:
        raise ValueError("last dimension must contain u and v")
    return tensor[..., :2]


def _relative_l2_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = _uv(prediction)
    target = _uv(target)
    dimensions = tuple(range(1, prediction.ndim))
    numerator = torch.linalg.vector_norm(prediction - target, dim=dimensions)
    denominator = torch.linalg.vector_norm(target, dim=dimensions).clamp_min(EPS)
    return numerator / denominator


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _relative_l2_per_sample(prediction, target).mean()


def turbulent_kinetic_energy(flow: torch.Tensor) -> torch.Tensor:
    velocity = _uv(flow)
    fluctuation = velocity - velocity.mean(dim=1, keepdim=True)
    return 0.5 * fluctuation.square().sum(dim=-1).mean(dim=1)


def _tke_relative_l2_per_sample(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    pred_tke = turbulent_kinetic_energy(prediction)
    true_tke = turbulent_kinetic_energy(target)
    numerator = torch.linalg.vector_norm(pred_tke - true_tke, dim=(1, 2))
    denominator = torch.linalg.vector_norm(true_tke, dim=(1, 2)).clamp_min(EPS)
    return numerator / denominator


def tke_relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _tke_relative_l2_per_sample(prediction, target).mean()


def _mvpe_relative_l2_per_sample(
    prediction: torch.Tensor, target: torch.Tensor, sub_s_real: int = 2
) -> torch.Tensor:
    """Official v9 mean-velocity-profile error at the four wake probes."""
    if prediction.ndim != 5 or target.shape != prediction.shape:
        raise ValueError("prediction and target must share shape (N,T,H,W,C)")

    d = 16
    center_x = 10
    center_y = 32
    n_probe = 9
    batch, _, height, width, _ = prediction.shape
    probe_center_y = int(center_y / sub_s_real)
    interval_y = min(2, int(height / (n_probe + 1)))
    probe_y = [
        probe_center_y + interval_y * j
        for j in range(-(n_probe - 1) // 2, n_probe - (n_probe - 1) // 2)
    ]
    probe_y = [y for y in probe_y if 0 <= y < height]
    if not probe_y:
        return prediction.new_zeros((batch,))

    errors = []
    for index in range(4):
        if int((2 * d + center_x) / sub_s_real) < width:
            probe_x = int(((index + 1) * d + center_x) / sub_s_real)
        else:
            probe_x = int((0.5 * (index + 2) * d + center_x) / sub_s_real)
        if not 0 <= probe_x < width:
            continue
        pred_profile = _uv(prediction)[:, :, probe_y, probe_x].mean(dim=1).flatten(1)
        true_profile = _uv(target)[:, :, probe_y, probe_x].mean(dim=1).flatten(1)
        numerator = torch.linalg.vector_norm(pred_profile - true_profile, dim=1)
        denominator = torch.linalg.vector_norm(true_profile, dim=1).clamp_min(EPS)
        errors.append(numerator / denominator)
    if not errors:
        return prediction.new_zeros((batch,))
    return torch.stack(errors).mean(dim=0)


def profile_relative_l2(
    prediction: torch.Tensor, target: torch.Tensor, sub_s_real: int = 2
) -> torch.Tensor:
    """Official v9 MVPE; the historical function name is kept for callers."""
    return _mvpe_relative_l2_per_sample(prediction, target, sub_s_real).mean()


def error_to_score(error: torch.Tensor | float) -> torch.Tensor:
    value = torch.as_tensor(error)
    mapped = 100.0 / (1.0 + 0.5 * value.clamp_min(0.0))
    return torch.where(torch.isfinite(value), mapped, torch.zeros_like(mapped))


def time_score(seconds: torch.Tensor | float) -> torch.Tensor:
    value = torch.as_tensor(seconds)
    if not value.is_floating_point():
        value = value.float()
    valid = torch.isfinite(value) & (value > 0.0)
    safe = value.clamp_min(torch.finfo(value.dtype).tiny)
    mapped = 100.0 / (1.0 + torch.sqrt(safe / SOLVER_REFERENCE_SECONDS))
    return torch.where(valid, mapped, torch.zeros_like(mapped))


def inferred_composite_score(
    rel_l2_score: torch.Tensor | float,
    tke_score: torch.Tensor | float,
    mvpe_score: torch.Tensor | float,
    runtime_score: torch.Tensor | float,
    sps_score: torch.Tensor | float,
) -> torch.Tensor:
    """Composite inferred exactly from 92 public rows, not organizer-published."""
    # Public score components are scalars and may originate on different devices
    # (for example, measured runtime on CPU and SPS on CUDA). Composite scoring
    # is a tiny host-side reduction, so normalize all components to CPU first.
    values = torch.stack(
        [
            torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
            for value in (rel_l2_score, tke_score, mvpe_score, runtime_score, sps_score)
        ]
    ).clamp_min(0.0)
    weights = values.new_tensor(INFERRED_COMPOSITE_WEIGHTS)
    power = INFERRED_COMPOSITE_POWER
    return torch.sum(weights * values.pow(power)).pow(1.0 / power)


def safe_prediction_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
    sigma_global: float = SIGMA_GLOBAL,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the official v9 SPS subscore (0-100) and measured-element coverage."""
    pred = _uv(prediction)
    true = _uv(target)
    if lower is None and upper is None:
        interval = 0.1 * pred.abs()
        lo = pred - interval / 2.0
        hi = pred + interval / 2.0
    elif lower is None or upper is None:
        raise ValueError("SPS bounds require both lower and upper, or neither")
    else:
        lo = _uv(lower)
        hi = _uv(upper)
        if lo.shape != pred.shape or hi.shape != pred.shape:
            raise ValueError("SPS bounds must have the prediction shape")
        if not torch.isfinite(lo).all() or not torch.isfinite(hi).all():
            raise ValueError("SPS bounds must be finite")
        if torch.any(lo > hi):
            raise ValueError("lower bound exceeds upper bound")

    inside = (true >= lo) & (true <= hi)
    normalized_length = (hi - lo) / sigma_global
    scored = true != 0.0
    n_scored = scored.count_nonzero()
    if int(n_scored) == 0:
        raise ValueError("target has no non-zero measured element for SPS")

    branches = (
        (0.5, _relative_l2_per_sample(prediction, target)),
        (0.3, _tke_relative_l2_per_sample(prediction, target)),
        (0.2, _mvpe_relative_l2_per_sample(prediction, target)),
    )
    weighted = prediction.new_zeros((), dtype=torch.float64)
    for weight, metric in branches:
        mapped = metric / (0.5 + metric)
        shaped = mapped.reshape((mapped.shape[0],) + (1,) * (pred.ndim - 1))
        element = (1.0 - shaped) * torch.exp(-normalized_length)
        element = torch.where(inside, element, torch.zeros_like(element))
        element = torch.where(torch.isfinite(shaped), element, torch.zeros_like(element))
        branch = element.masked_fill(~scored, 0.0).sum(dtype=torch.float64) / n_scored
        weighted = weighted + weight * branch

    coverage = (inside & scored).count_nonzero().to(torch.float64) / n_scored
    return (100.0 * weighted).clamp(0.0, 100.0), coverage


@dataclass(frozen=True)
class PublicScores:
    rel_l2_score: float
    tke_score: float
    mvpe_score: float
    time_score: float
    sps_score: float
    sps_coverage: float
    inferred_final_score: float


@torch.inference_mode()
def score_batch(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    elapsed_seconds: float,
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
) -> PublicScores:
    rel = error_to_score(relative_l2(prediction, target)).item()
    tke = error_to_score(tke_relative_l2(prediction, target)).item()
    profile = error_to_score(profile_relative_l2(prediction, target)).item()
    timing = time_score(elapsed_seconds).item()
    uncertainty, coverage = safe_prediction_score(
        prediction, target, lower=lower, upper=upper
    )
    composite = inferred_composite_score(rel, tke, profile, timing, uncertainty)
    return PublicScores(
        rel_l2_score=rel,
        tke_score=tke,
        mvpe_score=profile,
        time_score=timing,
        sps_score=uncertainty.item(),
        sps_coverage=coverage.item(),
        inferred_final_score=composite.item(),
    )
