"""Differentiable empirical Public composite; not an organizer-published loss.

Evaluation keeps all original official per-case/whole-case aggregations intact.
The fixed time component is a counterfactual for identical inference architectures.
"""
POWER = 0.6854
WEIGHTS = (0.5, 0.1, 0.1, 0.1, 0.2)
FIXED_TIME = 90.0


def training_loss(rel_l2, tke, mvpe, soft_sps):
    import torch
    errors = torch.stack((rel_l2, tke, mvpe)).to(torch.float64)
    quality = 100.0 / (1.0 + 0.5 * errors)
    scores = torch.cat((quality, quality.new_tensor([FIXED_TIME]),
                        (100.0 * soft_sps).to(torch.float64).reshape(1)))
    weights = scores.new_tensor(WEIGHTS)
    composite = (weights * scores.clamp_min(1e-12).pow(POWER)).sum().pow(1.0 / POWER)
    # A fixed scale keeps RelL2 gradients near the old training loss scale.
    return (100.0 - composite) / 25.0


def composite(scores):
    import numpy as np
    return np.sum(np.asarray(scores, dtype=np.float64) ** POWER * WEIGHTS,
                  axis=-1) ** (1.0 / POWER)


def compare_rows(previous, candidate, seed=20260922983):
    """Case-cluster bootstrap of whole-case aggregation, never mean case scores."""
    import numpy as np
    assert len(previous) == len(candidate) > 0
    for a, b in zip(previous, candidate):
        assert all(a[k] == b[k] for k in (
            'case', 'windows', 'scored_elements', 'input_sha256',
            'target_sha256', 'window_starts'))
    n = len(previous)
    sample = np.random.default_rng(seed).integers(n, size=(10000, n))

    def aggregate(rows, indices):
        windows = np.array([r['windows'] for r in rows])[indices]
        elements = np.array([r['scored_elements'] for r in rows])[indices]
        scores = []
        for name in ['rel_l2', 'tke', 'mvpe']:
            errors = np.array([r['mean_errors'][name] for r in rows])[indices]
            mean = (windows * errors).sum(axis=-1) / windows.sum(axis=-1)
            scores.append(100.0 / (1.0 + 0.5 * mean))
        scores.append(np.full(indices.shape[:-1], FIXED_TIME))
        sps = np.array([r['sps_score'] for r in rows])[indices]
        scores.append((elements * sps).sum(axis=-1) / elements.sum(axis=-1))
        return composite(np.stack(scores, axis=-1))

    original = np.arange(n)
    a, b = float(aggregate(previous, original)), float(aggregate(candidate, original))
    bootstrap = aggregate(candidate, sample) - aggregate(previous, sample)
    return dict(baseline=a, candidate=b, delta=b-a,
                CI90=np.quantile(bootstrap, [.05, .95]).tolist(),
                formula='Empirical power0.6854, weights0.5/0.1/0.1/0.1/0.2; Time fixed90',
                scope='Derived from the same official component rows, not a new inference or official final/Public measurement. Reused case clusters; not selection-adjusted.')
