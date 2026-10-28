"""Pure NumPy train-prediction limits; no fitting, labels, models or settings."""
from __future__ import annotations

import numpy as np


def _predictions(values):
    values = np.asarray(values)
    if values.dtype != np.float32 or values.ndim != 4 or values.shape[1:] != (4, 2, 2) or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty FP32 N x 4 x 2 x 2 predictions required")
    return values


def clip_limits(train_predictions):
    """All sixteen min/max columns, derived only from supplied training predictions."""
    values = _predictions(train_predictions)
    return {"lower": values.min(axis=0), "upper": values.max(axis=0)}


def clip_gains(predictions, limits):
    """Two-sided FP32 elementwise clip, preserving every within-range input bit."""
    values = _predictions(predictions)
    if set(limits) != {"lower", "upper"}:
        raise ValueError("exact lower/upper limits required")
    lower, upper = (np.asarray(limits[key]) for key in ("lower", "upper"))
    if any(item.dtype != np.float32 or item.shape != (4, 2, 2) or not np.isfinite(item).all() for item in (lower, upper)) or np.any(lower > upper):
        raise ValueError("finite ordered FP32 4 x 2 x 2 limits required")
    result = np.minimum(np.maximum(values, lower), upper)
    # Equal signed zeros are within support: retain their input bits as required.
    return np.where((values >= lower) & (values <= upper), values, result)
