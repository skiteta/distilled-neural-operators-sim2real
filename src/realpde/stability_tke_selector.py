"""Input-half TKE stability features and a case-balanced fixed ridge selector."""
import numpy as np
from .learned_trajectory_gate import case_weights

FEATURE_NAMES = ('predicted_gain_per_action', 'past_proxy_per_action', 'action_scale',
                 'half_TKE_disagreement', 'early_proxy_per_action', 'late_proxy_per_action',
                 'stability_weighted_proxy', 'stability_weighted_predicted_gain')


def energy(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 4 or x.shape[0] < 2 or x.shape[-1] != 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite T,H,W,2 input')
    return .5 * np.mean(np.sum((x - x.mean(0)) ** 2, axis=-1), axis=0)


def past_maps(observed):
    x = np.asarray(observed)
    if x.shape[0] != 20:
        raise ValueError('Only the current 20-frame input is accepted')
    return np.stack([energy(x), energy(x[:10]), energy(x[10:])])


def action_features(maps, baseline, candidate, predicted_gain):
    k = np.asarray(maps, dtype=np.float64)
    kb, kc = energy(baseline), energy(candidate)
    if k.shape != (3, *kb.shape) or not np.isfinite(k).all() or not np.isfinite(predicted_gain):
        raise ValueError('Invalid observed maps or gain')
    scale = float(np.linalg.norm(kc - kb) / max(np.linalg.norm(kb), 1e-16))
    den = max(scale, 1e-8)
    proxy = np.array([(np.linalg.norm(kb - v) - np.linalg.norm(kc - v)) /
                      max(np.linalg.norm(v), 1e-16) / den for v in k])
    disagreement = float(np.linalg.norm(k[1] - k[2]) / max(np.linalg.norm(k[0]), 1e-16))
    p = float(predicted_gain) / den
    return np.array([p, proxy[0], scale, disagreement, proxy[1], proxy[2],
                     proxy[0] / (1 + disagreement), p / (1 + disagreement)]), den


def fit_selector(x, y, rows, columns):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    cols = list(columns)
    if x.ndim != 2 or y.shape != (len(x),) or len(rows) != len(x) or not len(x):
        raise ValueError('Aligned nonempty training rows required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite training values')
    x = x[:, cols]
    w = case_weights(rows)
    mu = np.sum(w[:, None] * x, axis=0)
    sd = np.sqrt(np.sum(w[:, None] * (x - mu) ** 2, axis=0)).clip(1e-6)
    z = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    penalty = np.eye(z.shape[1]); penalty[0, 0] = 0
    coef = np.linalg.solve(z.T @ (w[:, None] * z) + penalty, z.T @ (w * y))
    return dict(columns=cols, mean=mu.tolist(), scale=sd.tolist(), coefficient=coef.tolist(),
                train_cases=sorted({r['case'] for r in rows}), train_rows=len(rows), ridge_lambda=1.)


def predict_selector(model, x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Finite feature matrix required')
    mu, sd, coef = [np.asarray(model[k]) for k in ('mean', 'scale', 'coefficient')]
    return np.array([np.r_[1., (r[model['columns']] - mu) / sd] @ coef for r in x])
