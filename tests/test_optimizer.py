"""Tests for SLSQP budget allocation."""

from __future__ import annotations

import numpy as np
import pytest

from dgp.config import ChannelConfig
from models.optimizer import allocate_budget, expected_revenue


def _channel(name: str, hill_alpha: float = 2.0, hill_gamma: float = 0.5,
             hill_scale: float = 100.0) -> ChannelConfig:
    """Test fixture: a channel whose RW/adstock/spend params are irrelevant here.

    The optimizer only reads hill_alpha / hill_gamma / hill_scale; the other
    fields are required by ChannelConfig but do not affect allocate_budget.
    """
    return ChannelConfig(
        name=name,
        beta_0=1.0,
        beta_drift=0.0,
        beta_innovation_std=0.0,
        adstock_decay=0.0,
        hill_alpha=hill_alpha,
        hill_gamma=hill_gamma,
        hill_scale=hill_scale,
        spend_mean=50.0,
        spend_log_sigma=0.1,
    )


# --- symmetry: equal β + identical channels → equal split ---------------


def test_identical_channels_equal_beta_yields_equal_split() -> None:
    channels = (_channel("a"), _channel("b"))
    beta_hat = np.array([1.0, 1.0])
    result = allocate_budget(beta_hat, channels, total_budget=100.0)
    assert result["a"] == pytest.approx(50.0, abs=1e-4)
    assert result["b"] == pytest.approx(50.0, abs=1e-4)


# --- budget conservation and non-negativity -----------------------------


def test_allocation_sums_to_total_budget() -> None:
    channels = (_channel("a"), _channel("b"), _channel("c"))
    beta_hat = np.array([2.0, 1.0, 3.0])
    total = 250.0
    result = allocate_budget(beta_hat, channels, total_budget=total)
    assert sum(result.values()) == pytest.approx(total, abs=1e-4)


def test_allocation_is_non_negative() -> None:
    channels = (_channel("a"), _channel("b"), _channel("c"))
    beta_hat = np.array([1.0, 0.01, 1.0])  # near-zero β on b should push spend low
    result = allocate_budget(beta_hat, channels, total_budget=200.0)
    for v in result.values():
        assert v >= -1e-8


# --- directionality: higher β gets more -------------------------------


def test_higher_beta_gets_more_spend_when_channels_are_identical() -> None:
    channels = (_channel("a"), _channel("b"))
    beta_hat = np.array([3.0, 1.0])
    result = allocate_budget(beta_hat, channels, total_budget=200.0)
    assert result["a"] > result["b"]


# --- KKT first-order condition -----------------------------------------


def test_kkt_marginal_revenues_equal_at_interior_optimum() -> None:
    """At an interior optimum with sum-to-budget, ∂R/∂s_c must be equal
    across channels (the common Lagrange multiplier).  Check with finite
    differences."""
    channels = (
        _channel("a", hill_alpha=2.0, hill_gamma=0.5, hill_scale=100.0),
        _channel("b", hill_alpha=1.5, hill_gamma=0.4, hill_scale=80.0),
        _channel("c", hill_alpha=2.5, hill_gamma=0.6, hill_scale=120.0),
    )
    beta_hat = np.array([2.0, 1.2, 1.8])
    total = 300.0
    result = allocate_budget(beta_hat, channels, total_budget=total)
    spend = np.array([result[c.name] for c in channels])

    eps = 1e-4
    marginals = []
    for i in range(len(channels)):
        s_up = spend.copy()
        s_down = spend.copy()
        s_up[i] += eps
        s_down[i] -= eps
        mr = (
            expected_revenue(s_up, beta_hat, channels)
            - expected_revenue(s_down, beta_hat, channels)
        ) / (2 * eps)
        marginals.append(mr)

    # Interior solution (all spends well above 0), so all marginals equal.
    assert min(spend) > 1e-2
    np.testing.assert_allclose(marginals, marginals[0], atol=1e-4)


# --- validation --------------------------------------------------------


def test_allocate_budget_rejects_wrong_beta_shape() -> None:
    channels = (_channel("a"), _channel("b"))
    with pytest.raises(ValueError, match="beta_hat shape"):
        allocate_budget(np.array([1.0, 2.0, 3.0]), channels, total_budget=100.0)


def test_allocate_budget_rejects_nonpositive_budget() -> None:
    channels = (_channel("a"), _channel("b"))
    with pytest.raises(ValueError, match="total_budget"):
        allocate_budget(np.array([1.0, 1.0]), channels, total_budget=0.0)


# --- expected_revenue sanity ------------------------------------------


def test_expected_revenue_zero_spend_is_zero() -> None:
    channels = (_channel("a"), _channel("b"))
    beta_hat = np.array([2.0, 3.0])
    assert expected_revenue(np.zeros(2), beta_hat, channels) == pytest.approx(0.0)


def test_expected_revenue_is_sum_of_per_channel_contributions() -> None:
    """R(s) with one channel active matches β̂_c · Hill(s_c / scale_c)."""
    channels = (_channel("a", hill_alpha=2.0, hill_gamma=0.5, hill_scale=100.0),)
    beta_hat = np.array([3.0])
    s = np.array([50.0])
    # Hill(0.5; α=2, γ=0.5) = 0.5.
    assert expected_revenue(s, beta_hat, channels) == pytest.approx(3.0 * 0.5)
