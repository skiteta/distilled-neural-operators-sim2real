"""Training objectives of the lineage, written out readably (the trainers in package/ use the same formulas inline).

Shapes: forecasts and targets are (N, 20, H, W, 3) with channels (u, v, p); only u and v are scored.

Competition errors (per window, then averaged over the batch)
    relative_l2      ||p - t|| / ||t|| over all frames, pixels and components
    tke_relative_l2  same on the per-pixel turbulent kinetic energy 0.5 * sum_{u,v} var_t
    mvpe_relative_l2 same on the time-mean velocity at 4 x 9 wake probe pixels (official probe geometry)

Interval objective
    soft_sps_reward  smooth surrogate of the Safe Prediction Score: an accuracy factor of the forecast times a sigmoid
                     "inside the interval" indicator times a tightness term, averaged over measured elements

Point objectives
    composite_score_loss  (100 - C) / 25, C = a differentiable copy of the competition composite (stages S8-S14)
    final_stage_loss      the stage S15 real-data loss (relL2 + TKE + 0.1 MVPE + 0.25 early + 2 phase + 0.2 late
                          - 6 soft SPS)
    distillation_loss     point KD to the teacher forecast + 0.1 normalized hidden-feature KD
    interval_distillation_loss  center relL2 + 0.1 mean |log half-width difference| to the teacher interval
"""

from __future__ import annotations

import torch

EPS = 1e-8
SIGMA_GLOBAL = 0.0563870259  # official SPS width scale
COMPOSITE_POWER = 0.6854  # composite = weighted power mean of the five 0-100 subscores
COMPOSITE_WEIGHTS = (0.5, 0.1, 0.1, 0.1, 0.2)  # RelL2, TKE, MVPE, Time, SPS
FIXED_TIME_SCORE = 90.0  # the Time subscore does not depend on the weights


def _norm_ratio(difference: torch.Tensor, reference: torch.Tensor, dims) -> torch.Tensor:
    return torch.linalg.vector_norm(difference, dim=dims) / torch.linalg.vector_norm(reference, dim=dims).clamp_min(EPS)


def relative_l2_per_window(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p, t = prediction[..., :2], target[..., :2]
    return _norm_ratio(p - t, t, tuple(range(1, p.ndim)))


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return relative_l2_per_window(prediction, target).mean()


def turbulent_kinetic_energy(flow: torch.Tensor) -> torch.Tensor:
    """(N, H, W): 0.5 * sum over u, v of the temporal variance."""
    velocity = flow[..., :2]
    return 0.5 * (velocity - velocity.mean(dim=1, keepdim=True)).square().sum(dim=-1).mean(dim=1)


def tke_relative_l2_per_window(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p, t = turbulent_kinetic_energy(prediction), turbulent_kinetic_energy(target)
    return _norm_ratio(p - t, t, (1, 2))


def tke_relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return tke_relative_l2_per_window(prediction, target).mean()


def wake_probes(height: int, width: int, subsample: int = 2) -> tuple[list[int], list[int]]:
    """Official probe pixels: 9 rows around the wake center line, 4 columns downstream (at the 2x-subsampled grid)."""
    center_y, center_x, spacing, rows = 32, 10, 16, 9
    step = min(2, int(height / (rows + 1)))
    ys = [int(center_y / subsample) + step * j for j in range(-(rows - 1) // 2, rows - (rows - 1) // 2)]
    xs = [
        int(((k + 1) * spacing + center_x) / subsample)
        if int((2 * spacing + center_x) / subsample) < width
        else int((0.5 * (k + 2) * spacing + center_x) / subsample)
        for k in range(4)
    ]
    return [y for y in ys if 0 <= y < height], [x for x in xs if 0 <= x < width]


def mvpe_relative_l2_per_window(prediction: torch.Tensor, target: torch.Tensor, subsample: int = 2) -> torch.Tensor:
    ys, xs = wake_probes(prediction.shape[2], prediction.shape[3], subsample)
    if not ys or not xs:
        return prediction.new_zeros(prediction.shape[0])
    errors = []
    for x in xs:
        p = prediction[..., :2][:, :, ys, x].mean(dim=1).flatten(1)
        t = target[..., :2][:, :, ys, x].mean(dim=1).flatten(1)
        errors.append(_norm_ratio(p - t, t, 1))
    return torch.stack(errors).mean(dim=0)


def mvpe_relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mvpe_relative_l2_per_window(prediction, target).mean()


def accuracy_factor(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(N,) the SPS accuracy factor sum_k w_k (1 - e_k / (0.5 + e_k)) with weights 0.5/0.3/0.2 on RelL2/TKE/MVPE."""
    rel = relative_l2_per_window(prediction, target)
    tke = tke_relative_l2_per_window(prediction, target)
    mvpe = mvpe_relative_l2_per_window(prediction, target)
    return 0.5 * (1.0 - rel / (0.5 + rel)) + 0.3 * (1.0 - tke / (0.5 + tke)) + 0.2 * (1.0 - mvpe / (0.5 + mvpe))


def soft_sps_reward(prediction, target, shift, radius, scale, tau: float = 0.01) -> torch.Tensor:
    """Mean over measured elements (target != 0) of  factor * sigmoid((r - |e - s|) / tau) * exp(-2 r scale / sigma).

    e = (target - forecast) / scale, s = shift, r = radius (interval half-width), all in units of the window scale.
    The official SPS uses a hard inside/outside indicator; the sigmoid makes it differentiable.
    """
    error = (target[..., :2] - prediction[..., :2]) / scale
    measured = target[..., :2] != 0
    inside = torch.sigmoid((radius - (error - shift).abs()) / tau)
    tightness = torch.exp(-2 * radius * scale / SIGMA_GLOBAL)
    factor = accuracy_factor(prediction, target)[:, None, None, None, None]
    return (factor * inside * tightness)[measured].mean()


def composite_score_loss(rel_l2, tke, mvpe, soft_sps) -> torch.Tensor:
    """(100 - C) / 25, C = weighted power mean of the subscores 100 / (1 + 0.5 e); Time fixed, SPS = 100 x reward."""
    quality = 100.0 / (1.0 + 0.5 * torch.stack((rel_l2, tke, mvpe)).to(torch.float64))
    scores = torch.cat(
        (quality, quality.new_tensor([FIXED_TIME_SCORE]), (100.0 * soft_sps).to(torch.float64).reshape(1))
    )
    composite = (
        (scores.new_tensor(COMPOSITE_WEIGHTS) * scores.clamp_min(1e-12).pow(COMPOSITE_POWER))
        .sum()
        .pow(1.0 / COMPOSITE_POWER)
    )
    return (100.0 - composite) / 25.0


def energy_normalized_phase_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RelL2 of the time-centered fluctuations after rescaling the forecast to the target's per-pixel energy.

    Measures *where and when* fluctuations happen, independently of how strong they are.
    """
    p = prediction[..., :2] - prediction[..., :2].mean(1, keepdim=True)
    t = target[..., :2] - target[..., :2].mean(1, keepdim=True)
    p_norm = torch.linalg.vector_norm(p, dim=(1, 4), keepdim=True)
    t_norm = torch.linalg.vector_norm(t, dim=(1, 4), keepdim=True)
    return relative_l2(p * (t_norm / p_norm.clamp_min(1e-8)), t)


def final_stage_loss(prediction: torch.Tensor, target: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
    """Stage S15 real-data loss: full TKE weight plus frame-range and phase terms, minus 6 x the soft SPS reward."""
    return (
        relative_l2(prediction, target)
        + 1.0 * tke_relative_l2(prediction, target)
        + 0.1 * mvpe_relative_l2(prediction, target)
        + 0.25 * relative_l2(prediction[:, :3], target[:, :3])
        - 6.0 * reward
        + 2.0 * energy_normalized_phase_error(prediction, target)
        + 0.2 * relative_l2(prediction[:, 10:], target[:, 10:])
    )


def distillation_loss(prediction, hidden, teacher_prediction, teacher_hidden) -> torch.Tensor:
    feature = (hidden - teacher_hidden).square().mean() / teacher_hidden.square().mean().clamp_min(1e-8)
    return relative_l2(prediction, teacher_prediction) + 0.1 * feature


def interval_distillation_loss(center, half_width, teacher_center, teacher_half_width) -> torch.Tensor:
    log_gap = torch.log(half_width.clamp_min(1e-6)) - torch.log(teacher_half_width.clamp_min(1e-6))
    return 1.0 * relative_l2(center, teacher_center) + 0.1 * log_gap.abs().mean()
