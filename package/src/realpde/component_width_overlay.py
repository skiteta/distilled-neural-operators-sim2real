"""C550 pure component-separated radius decisions and official reward accounting.

No model loading, data access, scorer import, process-global settings or CUDA
initialization. Callers inject the pinned official scorer. Features are exactly
the frozen C546/C548 input-only CPU FP32 features.
"""
from __future__ import annotations

import numpy as np
import torch

from .spatial_width_overlay import BANDS, FACTORS, INPUT_ONLY_CHANNELS, RIDGE_ALPHA, input_features

WEIGHTS = np.asarray((.5, .3, .2), np.float64)


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def routed_bounds(center, physical_radius, actions):
    """N4 shared or N4x2 separate decisions; raw radius*factor then center±radius."""
    if (not isinstance(center, torch.Tensor) or not isinstance(physical_radius, torch.Tensor)
            or center.shape != physical_radius.shape or center.ndim != 5
            or tuple(center.shape[1:]) != (20, 32, 64, 2) or not len(center)
            or center.dtype != torch.float32 or physical_radius.dtype != torch.float32
            or center.device != physical_radius.device
            or not bool(torch.isfinite(center).all() and torch.isfinite(physical_radius).all())
            or bool((physical_radius < 0).any())):
        raise ValueError("finite FP32 raw center/radius on the same device are required")
    choice = _numpy(actions)
    if choice.shape == (len(center), 4):
        choice = np.repeat(choice[..., None], 2, axis=-1)
    if (choice.shape != (len(center), 4, 2) or not np.issubdtype(choice.dtype, np.integer)
            or not np.isin(choice, (0, 1, 2)).all()):
        raise ValueError("integer actions must be N4 or N4x2 and use indices0/1/2")
    factors = np.ones((len(center), 64, 2), np.float32)
    for band, (left, right) in enumerate(BANDS):
        factors[:, left:right, :] = FACTORS[choice[:, band, :]][:, None, :]
    factor = torch.from_numpy(factors).to(center.device).reshape(len(center), 1, 1, 64, 2)
    radius = physical_radius * factor
    return center - radius, center + radius


def component_statistics(point, target, lower, upper, *, official):
    """Original full20 UV point penalties; FP32 products before FP64 sums.

    All spatial bands and components retain their individual numerators/counts;
    callers normalize by the original full-window/surface measured UV count.
    ``inside`` counts measured inclusive hits; ``width_sums`` includes only
    measured components. Neither component is scored as an independent c=1 task.
    """
    point, target, lower, upper = map(_numpy, (point, target, lower, upper))
    if (point.shape != target.shape or point.ndim != 5 or not len(point)
            or point.shape[1:4] != (20, 32, 64) or point.shape[-1] not in (2, 3)):
        raise ValueError("original full20 two-component point/target arrays are required")
    if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in (point, target, lower, upper)):
        raise ValueError("finite FP32 arrays are required")
    if any(value.ndim != 5 or value.shape[-1] not in (2, 3) for value in (lower, upper)):
        raise ValueError("bounds must have two UV channels or UV plus zero pressure")
    if any(value.shape[-1] == 3 and not np.all(value[..., 2] == 0) for value in (point, target, lower, upper)):
        raise ValueError("third channel must remain exactly zero")
    measured_target, lo, hi = target[..., :2], lower[..., :2], upper[..., :2]
    if lo.shape != measured_target.shape or hi.shape != measured_target.shape or np.any(lo > hi):
        raise ValueError("bounds must match original UV geometry and be ordered")
    errors = np.stack((official.rel_l2_per_sample(point, target, 2),
                       official.tke_rel_l2_per_sample(point, target, 2),
                       official.mvpe_rel_l2_per_sample(point, target)), axis=-1)
    if errors.shape != (len(point), 3) or errors.dtype != np.float32:
        raise ValueError("pinned official full-window error vectors must be FP32 N3")
    measured = measured_target != 0
    inside = (measured_target >= lo) & (measured_target <= hi)
    width = hi - lo
    decay = np.exp(-width / official.SIGMA_GLOBAL)
    branches = []
    for branch in range(3):
        penalty = errors[:, branch] / (.5 + errors[:, branch])
        product = (1. - penalty[:, None, None, None, None]) * decay
        product = np.where(inside, product, 0.)
        product = np.where(np.isfinite(penalty)[:, None, None, None, None], product, 0.)
        if product.dtype != np.float32:
            raise RuntimeError("official branch products must remain FP32 before summation")
        branches.append(np.stack([
            np.sum(product[:, :, :, left:right], axis=(2, 3),
                   where=measured[:, :, :, left:right], dtype=np.float64)
            for left, right in BANDS], axis=2))
    counts = np.stack([measured[:, :, :, left:right].sum((2, 3), dtype=np.int64) for left, right in BANDS], axis=2)
    hits = np.stack([(inside & measured)[:, :, :, left:right].sum((2, 3), dtype=np.int64) for left, right in BANDS], axis=2)
    widths = np.stack([np.sum(width[:, :, :, left:right], axis=(2, 3),
                             where=measured[:, :, :, left:right], dtype=np.float64)
                       for left, right in BANDS], axis=2)
    return {"reward_sums": np.stack(branches, axis=-1), "counts": counts,
            "errors": errors, "inside": hits, "width_sums": widths}


def component_reward_targets(point, target, center, physical_radius, *, official):
    """FP64 labels N4band x2component x2(low/high), original full UV denominator.

    Spatial/component contributions are additive with unchanged full-window
    point penalties. Thus each fixed action can be computed simultaneously in
    every band; labels equal individual band/component interventions versus
    identity, not a globally normalized per-component score.
    """
    identity = np.ones((len(center), 4, 2), np.int8)
    lower, upper = routed_bounds(center, physical_radius, identity)
    if not (torch.equal(lower, center - physical_radius) and torch.equal(upper, center + physical_radius)):
        raise RuntimeError("identity must exactly replay raw center/radius")
    baseline = component_statistics(point, target, lower, upper, official=official)
    denominator = baseline["counts"].sum((1, 2, 3))
    if np.any(denominator <= 0):
        raise ValueError("empty scored training window")
    baseline_num = np.tensordot(baseline["reward_sums"].sum(1), WEIGHTS, axes=(-1, 0))
    labels = []
    for action in (0, 2):
        choice = np.full_like(identity, action)
        lower, upper = routed_bounds(center, physical_radius, choice)
        actual = component_statistics(point, target, lower, upper, official=official)
        if not (np.array_equal(actual["counts"], baseline["counts"])
                and np.array_equal(actual["errors"], baseline["errors"])):
            raise RuntimeError("component intervention changed original masks/point factors")
        numerator = np.tensordot(actual["reward_sums"].sum(1), WEIGHTS, axes=(-1, 0))
        labels.append(100 * (numerator - baseline_num) / denominator[:, None, None])
    result = np.stack(labels, axis=-1)
    if result.shape != (len(center), 4, 2, 2) or result.dtype != np.float64 or not np.isfinite(result).all():
        raise RuntimeError("invalid sixteen component contribution labels")
    return result


def fit_ridge(features, labels):
    """Same alpha10 FP64 train solve; exactly one sixteen-output fitted model."""
    x, y = np.asarray(features, np.float64), np.asarray(labels, np.float64)
    if y.ndim == 4 and y.shape[1:] == (4, 2, 2):
        y = y.reshape(len(y), 16)
    if (x.ndim != 2 or not len(x) or x.shape[1] == 0 or y.shape != (len(x), 16)
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("finite train-only features and N16/N4x2x2 labels required")
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-8] = 1.
    normalized, y_mean = (x - mean) / std, y.mean(0)
    gram = normalized.T @ normalized
    gram.flat[::len(gram) + 1] += RIDGE_ALPHA
    weight = np.linalg.solve(gram, normalized.T @ (y - y_mean))
    result = {key: value.astype(np.float32) for key, value in
              (("mean", mean), ("std", std), ("y_mean", y_mean), ("weight", weight))}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("nonfinite frozen sixteen-output ridge")
    return result


def predicted_gains(features, parameters):
    """FP32 N4band x2component x2(low/high) gains, fixed unpadded B4 GEMM."""
    x = np.asarray(features, np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("finite feature matrix required")
    shapes = {"mean": (x.shape[1],), "std": (x.shape[1],), "y_mean": (16,), "weight": (x.shape[1], 16)}
    if set(parameters) != set(shapes) or any(np.asarray(parameters[key]).shape != shape
            or np.asarray(parameters[key]).dtype != np.float32 or not np.isfinite(parameters[key]).all()
            for key, shape in shapes.items()) or not np.all(parameters["std"] > 0):
        raise ValueError("exact frozen FP32 sixteen-output ridge parameters required")
    result = np.empty((len(x), 16), np.float32)
    for start in range(0, len(x), 4):
        normalized = (x[start:start + 4] - parameters["mean"]) / parameters["std"]
        result[start:start + 4] = parameters["y_mean"] + normalized @ parameters["weight"]
    if not np.isfinite(result).all():
        raise ValueError("nonfinite predicted component gains")
    return result.reshape(len(x), 4, 2, 2)


def choose_actions(gains, *, shared=False):
    """Same16 shared control sums u/v gains; ties/nonpositive always identity.

    This is the deployed C546 strict winner rule, not a target-only oracle's
    identity-first argmax ordering. In particular equal POSITIVE low/high gains
    also choose identity.
    """
    values = np.asarray(gains, np.float32)
    if values.ndim == 2 and values.shape[1] == 16:
        values = values.reshape(len(values), 4, 2, 2)
    if values.ndim != 4 or values.shape[1:] != (4, 2, 2) or not np.isfinite(values).all():
        raise ValueError("finite N16/N4x2x2 component gains required")
    if shared:
        values = values.sum(axis=2, dtype=np.float32)
    low, high = values[..., 0], values[..., 1]
    actions = np.ones(low.shape, np.int8)
    actions[(low > 0) & (low > high)] = 0
    actions[(high > 0) & (high > low)] = 2
    return actions
