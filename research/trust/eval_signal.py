"""Shared error detection metric. Larger signal means more suspicious."""

import numpy as np
from sklearn.metrics import roc_auc_score


def signal_error_auc(signal, wrong, n_boot=500, seed=42):
    s = np.asarray(signal, dtype=float)
    y = np.asarray(wrong)

    if s.ndim != 1 or y.shape != s.shape or s.size == 0:
        raise ValueError("Need equal, nonempty 1D signal and wrong arrays")

    if not np.isfinite(s).all():
        raise ValueError("Signals must be finite")

    # Validate before converting: 0.5 must not silently become 0.
    if not np.isin(y, [0, 1]).all():
        raise ValueError("wrong must contain only 0 or 1")
    y = y.astype(int)

    if (
        isinstance(n_boot, (bool, np.bool_))
        or not isinstance(n_boot, (int, np.integer))
        or n_boot < 1
    ):
        raise ValueError("n_boot must be a positive integer")

    if np.unique(y).size < 2:
        return None, None, None

    auc = float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)
    values = []

    for _ in range(n_boot):
        ids = rng.integers(0, len(y), len(y))
        if np.unique(y[ids]).size == 2:
            values.append(roc_auc_score(y[ids], s[ids]))

    if not values:
        return auc, None, None

    lo, hi = np.percentile(values, [2.5, 97.5])
    return auc, float(lo), float(hi)