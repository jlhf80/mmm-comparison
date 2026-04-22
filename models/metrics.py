"""Evaluation metrics shared across the fitters.

- `mape` / `rmse`: standard fit-quality on y.
- `allocation_error`: L1 distance between the optimizer's implied budget
  shares and the ground-truth optimal shares.  This is the decision-relevant
  error — the whole project hinges on showing that fit-quality error and
  allocation error are not the same thing.
- `waic`: widely applicable information criterion from a (S, T) matrix of
  per-sample, per-observation log-likelihoods.  Generic over source; the
  first consumer is the PyMC fitter's posterior.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error: mean(|y - ŷ| / |y|).

    Undefined when `y_true` contains zeros; callers are responsible for
    filtering those out if relevant.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return float(np.mean(np.abs(y_true - y_pred) / np.abs(y_true)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-square error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def allocation_error(
    allocated: dict[str, float],
    optimal: dict[str, float],
) -> float:
    """L1 distance between allocation shares.

    Each dict maps channel name → absolute spend.  Shares are computed by
    normalizing each dict by its own total, so scale cancels out; what we
    are measuring is the *pattern* of the allocation, not the total budget.
    Maximum value is 2.0 (two disjoint all-in-one-channel allocations).
    """
    if set(allocated) != set(optimal):
        raise ValueError(
            f"key mismatch: {sorted(allocated)} vs {sorted(optimal)}"
        )
    total_a = sum(allocated.values())
    total_o = sum(optimal.values())
    if total_a <= 0 or total_o <= 0:
        raise ValueError("allocation totals must be positive")
    return float(
        sum(abs(allocated[k] / total_a - optimal[k] / total_o) for k in allocated)
    )


def waic(loglik: np.ndarray) -> float:
    """Widely Applicable Information Criterion on (S, T) log-likelihoods.

    WAIC = -2 · (lppd - p_waic), where
        lppd_t   = log mean_s exp(loglik[s, t])
        p_waic_t = var_s(loglik[s, t])   (unbiased, ddof=1)

    `loglik[s, t]` is the log-likelihood of the t-th observation under the
    s-th posterior draw.
    """
    loglik = np.asarray(loglik, dtype=float)
    if loglik.ndim != 2:
        raise ValueError(f"loglik must be 2-D (S, T); got shape {loglik.shape}")
    n_samples = loglik.shape[0]
    if n_samples < 2:
        raise ValueError(f"need at least 2 posterior samples; got {n_samples}")
    lppd = float(np.sum(logsumexp(loglik, axis=0) - np.log(n_samples)))
    p_waic = float(np.sum(np.var(loglik, axis=0, ddof=1)))
    return -2.0 * (lppd - p_waic)
