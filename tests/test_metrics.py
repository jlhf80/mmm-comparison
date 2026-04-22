"""Tests for MAPE, RMSE, allocation_error, and WAIC."""

from __future__ import annotations

import numpy as np
import pytest

from models.metrics import allocation_error, mape, rmse, waic

# --- mape / rmse ---------------------------------------------------------


def test_mape_identical_is_zero() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert mape(y, y) == pytest.approx(0.0)


def test_mape_known_value() -> None:
    """mean(|[0.1, 0.1]| / |[1, 1]|) == 0.1."""
    y_true = np.array([1.0, 1.0])
    y_pred = np.array([0.9, 1.1])
    assert mape(y_true, y_pred) == pytest.approx(0.1)


def test_rmse_identical_is_zero() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_known_value() -> None:
    """RMSE([0,0,0], [1,2,3]) = sqrt(14/3)."""
    y_true = np.zeros(3)
    y_pred = np.array([1.0, 2.0, 3.0])
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(14.0 / 3.0))


def test_mape_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        mape(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_rmse_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        rmse(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


# --- allocation_error ----------------------------------------------------


def test_allocation_error_identical_is_zero() -> None:
    a = {"tv": 100.0, "digital": 50.0, "search": 25.0}
    assert allocation_error(a, a) == pytest.approx(0.0)


def test_allocation_error_disjoint_extreme_is_two() -> None:
    """All-in-one-channel vs all-in-other is the L1-shares maximum, 2.0."""
    a = {"tv": 10.0, "digital": 0.0}
    b = {"tv": 0.0, "digital": 10.0}
    assert allocation_error(a, b) == pytest.approx(2.0)


def test_allocation_error_is_scale_invariant() -> None:
    """Same shares at different scales ⇒ zero error."""
    a = {"tv": 1.0, "digital": 1.0}
    b = {"tv": 100.0, "digital": 100.0}
    assert allocation_error(a, b) == pytest.approx(0.0)


def test_allocation_error_known_value() -> None:
    """Shares (0.75, 0.25) vs (0.5, 0.5) ⇒ L1 = 0.25 + 0.25 = 0.5."""
    a = {"tv": 75.0, "digital": 25.0}
    b = {"tv": 50.0, "digital": 50.0}
    assert allocation_error(a, b) == pytest.approx(0.5)


def test_allocation_error_key_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="key mismatch"):
        allocation_error({"tv": 1.0}, {"digital": 1.0})


def test_allocation_error_nonpositive_total_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        allocation_error({"tv": 0.0, "digital": 0.0}, {"tv": 1.0, "digital": 1.0})


# --- waic ---------------------------------------------------------------


def test_waic_constant_loglik_has_zero_penalty() -> None:
    """If every posterior draw yields identical loglik, p_waic = 0 and
    WAIC = -2·sum(loglik_t)."""
    n_samples = 50
    per_obs = np.array([-1.0, -2.0, -0.5, -1.5, -3.0, -0.1, -0.2, -0.8, -1.1, -0.3])
    loglik = np.tile(per_obs, (n_samples, 1))
    expected = -2.0 * float(np.sum(per_obs))
    assert waic(loglik) == pytest.approx(expected)


def test_waic_matches_manual_formula_on_small_array() -> None:
    """Direct computation of WAIC = -2(lppd - p_waic) on a 3x2 array."""
    loglik = np.array(
        [
            [-1.0, -2.0],
            [-1.5, -1.0],
            [-0.5, -1.5],
        ]
    )
    lppd = float(np.sum(np.log(np.mean(np.exp(loglik), axis=0))))
    p_waic = float(np.sum(np.var(loglik, axis=0, ddof=1)))
    expected = -2.0 * (lppd - p_waic)
    assert waic(loglik) == pytest.approx(expected)


def test_waic_rejects_1d_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        waic(np.array([-1.0, -2.0, -3.0]))


def test_waic_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        waic(np.array([[-1.0, -2.0]]))
