"""Tests for geometric_adstock, hill_saturation, and precompute_features."""

from __future__ import annotations

import numpy as np
import pytest

from dgp.config import ChannelConfig
from dgp.transforms import geometric_adstock, hill_saturation, precompute_features

# --- geometric_adstock ---------------------------------------------------


def test_geometric_adstock_initial_condition() -> None:
    """a*_0 = s_0, since a*_{-1} is defined to be 0."""
    out = geometric_adstock(np.array([3.0, 0.0, 0.0]), decay=0.5)
    assert out[0] == pytest.approx(3.0)


def test_geometric_adstock_impulse_response_is_geometric_decay() -> None:
    """A single unit impulse at t=0 produces [1, λ, λ², …]."""
    decay = 0.6
    x = np.zeros(5)
    x[0] = 1.0
    expected = np.array([decay**i for i in range(5)])
    out = geometric_adstock(x, decay=decay)
    np.testing.assert_allclose(out, expected)


def test_geometric_adstock_zero_decay_is_identity() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(geometric_adstock(x, decay=0.0), x)


def test_geometric_adstock_is_linear() -> None:
    """The filter is linear: f(a·x + b·y) = a·f(x) + b·f(y)."""
    decay = 0.4
    x = np.array([1.0, 0.0, 2.0, 0.5])
    y = np.array([0.5, 1.0, 0.0, 0.0])
    a, b = 2.0, 3.0
    lhs = geometric_adstock(a * x + b * y, decay=decay)
    rhs = a * geometric_adstock(x, decay=decay) + b * geometric_adstock(y, decay=decay)
    np.testing.assert_allclose(lhs, rhs)


@pytest.mark.parametrize("bad_decay", [-0.1, 1.0, 1.5])
def test_geometric_adstock_rejects_out_of_range_decay(bad_decay: float) -> None:
    with pytest.raises(ValueError, match="decay"):
        geometric_adstock(np.array([1.0, 2.0]), decay=bad_decay)


def test_geometric_adstock_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        geometric_adstock(np.ones((2, 3)), decay=0.5)


# --- hill_saturation -----------------------------------------------------


def test_hill_saturation_at_gamma_is_half() -> None:
    """By construction, Hill(gamma) = 1/2 for any alpha, gamma > 0."""
    for alpha in (0.5, 1.0, 2.0, 3.7):
        for gamma in (0.1, 0.5, 1.0, 3.0):
            value = hill_saturation(np.array([gamma]), alpha=alpha, gamma=gamma)[0]
            assert value == pytest.approx(0.5)


def test_hill_saturation_at_zero_is_zero() -> None:
    np.testing.assert_allclose(
        hill_saturation(np.array([0.0]), alpha=2.0, gamma=0.5),
        np.array([0.0]),
    )


def test_hill_saturation_is_strictly_monotone() -> None:
    x = np.linspace(0.0, 10.0, 50)
    y = hill_saturation(x, alpha=2.0, gamma=1.0)
    diffs = np.diff(y)
    assert np.all(diffs > 0)


def test_hill_saturation_is_bounded_in_unit_interval() -> None:
    x = np.logspace(-3, 3, 100)
    y = hill_saturation(x, alpha=2.0, gamma=1.0)
    assert y.min() >= 0.0
    assert y.max() < 1.0


def test_hill_saturation_approaches_one_for_large_x() -> None:
    y = hill_saturation(np.array([1e6]), alpha=2.0, gamma=1.0)
    assert y[0] > 0.999


@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_hill_saturation_rejects_nonpositive_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        hill_saturation(np.array([1.0]), alpha=alpha, gamma=1.0)


@pytest.mark.parametrize("gamma", [0.0, -0.5])
def test_hill_saturation_rejects_nonpositive_gamma(gamma: float) -> None:
    with pytest.raises(ValueError, match="gamma"):
        hill_saturation(np.array([1.0]), alpha=2.0, gamma=gamma)


def test_hill_saturation_rejects_negative_input() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        hill_saturation(np.array([-0.1]), alpha=2.0, gamma=1.0)


# --- precompute_features -------------------------------------------------


def test_precompute_features_equals_chain_of_primitives() -> None:
    """precompute_features = hill_saturation(adstocked/scale)."""
    channel = ChannelConfig(
        name="test",
        beta_0=1.0,
        beta_drift=0.0,
        beta_innovation_std=0.0,
        adstock_decay=0.5,
        hill_alpha=2.0,
        hill_gamma=0.4,
        hill_scale=100.0,
        spend_mean=50.0,
        spend_log_sigma=0.3,
    )
    spend = np.array([100.0, 50.0, 0.0, 25.0, 75.0])

    expected = hill_saturation(
        geometric_adstock(spend, decay=channel.adstock_decay) / channel.hill_scale,
        alpha=channel.hill_alpha,
        gamma=channel.hill_gamma,
    )
    np.testing.assert_allclose(precompute_features(spend, channel), expected)


def test_precompute_features_output_is_in_unit_interval() -> None:
    channel = ChannelConfig(
        name="test",
        beta_0=1.0,
        beta_drift=0.0,
        beta_innovation_std=0.0,
        adstock_decay=0.7,
        hill_alpha=2.0,
        hill_gamma=0.5,
        hill_scale=200.0,
        spend_mean=100.0,
        spend_log_sigma=0.3,
    )
    rng = np.random.default_rng(0)
    spend = rng.lognormal(mean=4.6, sigma=0.3, size=200)
    features = precompute_features(spend, channel)
    assert features.min() >= 0.0
    assert features.max() < 1.0
    assert features.shape == spend.shape
