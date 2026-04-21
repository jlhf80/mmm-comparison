"""Tests for the synthetic DGP pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from dgp.config import ChannelConfig, DGPConfig, default_channels
from dgp.generator import (
    generate_dataset,
    simulate_baseline,
    simulate_beta_trajectory,
    simulate_spend,
)

# --- shape / field contract ---------------------------------------------


def test_generate_dataset_field_shapes() -> None:
    cfg = DGPConfig(n_weeks=60, seed=0)
    data = generate_dataset(cfg)

    assert data.y.shape == (60,)
    assert data.baseline.shape == (60,)
    assert data.noise.shape == (60,)
    for name in data.channel_names:
        assert data.spend[name].shape == (60,)
        assert data.features[name].shape == (60,)
        assert data.beta[name].shape == (60,)


def test_feature_matrix_and_beta_matrix_column_order_matches_channels() -> None:
    cfg = DGPConfig(n_weeks=40, seed=1)
    data = generate_dataset(cfg)

    feat_mat = data.feature_matrix()
    beta_mat = data.beta_matrix()
    assert feat_mat.shape == (40, len(cfg.channels))
    assert beta_mat.shape == (40, len(cfg.channels))
    for i, name in enumerate(data.channel_names):
        np.testing.assert_allclose(feat_mat[:, i], data.features[name])
        np.testing.assert_allclose(beta_mat[:, i], data.beta[name])


# --- coefficient random walk --------------------------------------------


def test_beta_trajectory_starts_at_beta_zero() -> None:
    """β_{c,0} is deterministic and equals channel.beta_0."""
    channel = ChannelConfig(
        name="c",
        beta_0=7.5,
        beta_drift=0.1,
        beta_innovation_std=0.5,
        adstock_decay=0.3,
        hill_alpha=2.0,
        hill_gamma=0.5,
        hill_scale=100.0,
        spend_mean=50.0,
        spend_log_sigma=0.2,
    )
    rng = np.random.default_rng(123)
    traj = simulate_beta_trajectory(channel, n_weeks=100, rng=rng)
    assert traj[0] == pytest.approx(7.5)


def test_beta_trajectory_drift_dominates_long_run_mean() -> None:
    """Without innovation noise, the mean step is exactly beta_drift."""
    channel = ChannelConfig(
        name="c",
        beta_0=0.0,
        beta_drift=0.05,
        beta_innovation_std=0.0,  # deterministic RW
        adstock_decay=0.0,
        hill_alpha=2.0,
        hill_gamma=0.5,
        hill_scale=1.0,
        spend_mean=1.0,
        spend_log_sigma=0.0,
    )
    rng = np.random.default_rng(0)
    traj = simulate_beta_trajectory(channel, n_weeks=101, rng=rng)
    # With 100 steps each of size 0.05, expect beta_100 ≈ 5.0.
    assert traj[-1] == pytest.approx(5.0)


# --- baseline -----------------------------------------------------------


def test_baseline_has_expected_length_and_mean() -> None:
    cfg = DGPConfig(n_weeks=52, baseline_mean=8.0, baseline_trend=0.0,
                    seasonality_amplitude=0.0, seed=0)
    alpha = simulate_baseline(cfg)
    assert alpha.shape == (52,)
    # With no trend and no seasonality, α_t ≡ baseline_mean.
    np.testing.assert_allclose(alpha, 8.0)


def test_baseline_seasonality_is_sinusoid() -> None:
    """With period=52 and amplitude=A, α_t - mean - trend·t is A·sin(2πt/52)."""
    cfg = DGPConfig(n_weeks=104, baseline_mean=0.0, baseline_trend=0.0,
                    seasonality_amplitude=2.0, seasonality_period=52.0, seed=0)
    alpha = simulate_baseline(cfg)
    t = np.arange(104)
    expected = 2.0 * np.sin(2.0 * np.pi * t / 52.0)
    np.testing.assert_allclose(alpha, expected)


# --- observation equation: reconstruct y from ground truth ---------------


def test_observation_equation_reconstruction() -> None:
    """y ≡ baseline + Σ_c β_{c,t} · a_{c,t} + ε_t, exactly."""
    cfg = DGPConfig(n_weeks=80, seed=7)
    data = generate_dataset(cfg)

    contribution = np.zeros(cfg.n_weeks)
    for c in data.channel_names:
        contribution += data.beta[c] * data.features[c]
    reconstructed = data.baseline + contribution + data.noise

    np.testing.assert_allclose(reconstructed, data.y)


# --- reproducibility -----------------------------------------------------


def test_same_seed_produces_identical_output() -> None:
    cfg = DGPConfig(n_weeks=40, seed=42)
    a = generate_dataset(cfg)
    b = generate_dataset(cfg)
    np.testing.assert_array_equal(a.y, b.y)
    for c in a.channel_names:
        np.testing.assert_array_equal(a.beta[c], b.beta[c])
        np.testing.assert_array_equal(a.spend[c], b.spend[c])


def test_different_seeds_produce_different_output() -> None:
    a = generate_dataset(DGPConfig(n_weeks=40, seed=1))
    b = generate_dataset(DGPConfig(n_weeks=40, seed=2))
    assert not np.array_equal(a.y, b.y)


# --- default config sanity: encodes the narrative ------------------------


def test_default_channels_encode_tv_digital_search_narrative() -> None:
    """TV drift < 0 (declining), Digital drift > 0 (rising), Search drift == 0."""
    channels = {c.name: c for c in default_channels()}
    assert set(channels.keys()) == {"tv", "digital", "search"}
    assert channels["tv"].beta_drift < 0
    assert channels["digital"].beta_drift > 0
    assert channels["search"].beta_drift == 0.0


def test_channel_config_is_frozen() -> None:
    """Ground-truth config must be immutable once constructed."""
    channel = default_channels()[0]
    with pytest.raises((AttributeError, Exception)):
        channel.beta_0 = 999.0  # type: ignore[misc]


# --- spend generation ---------------------------------------------------


def test_spend_is_positive_and_mean_matches() -> None:
    """Log-normal with the calibration in config produces E[s] ≈ spend_mean."""
    channel = ChannelConfig(
        name="c",
        beta_0=1.0,
        beta_drift=0.0,
        beta_innovation_std=0.0,
        adstock_decay=0.0,
        hill_alpha=2.0,
        hill_gamma=0.5,
        hill_scale=1.0,
        spend_mean=100.0,
        spend_log_sigma=0.3,
    )
    rng = np.random.default_rng(0)
    s = simulate_spend(channel, n_weeks=10_000, rng=rng)
    assert (s > 0).all()
    # Law of large numbers: sample mean should be close to spend_mean.
    assert s.mean() == pytest.approx(100.0, rel=0.02)
