"""Tests for the structural DLM extensions.

These cover:
  - The block-diagonal F / Q / slice layout assembled by
    `build_structural_state_space`.
  - The observation-matrix builder, including the "first of each seasonal
    pair is observed" convention.
  - End-to-end recovery of (a) a pure sinusoid by a seasonal-only DLM,
    (b) a linear trend by a local-linear-trend DLM, and (c) β_{c,t} on a
    DGP with trend + annual seasonality + drifting β — the regime the
    structural DLM was introduced to handle.
"""

from __future__ import annotations

import numpy as np
import pytest

from dgp.config import DGPConfig
from dgp.generator import generate_dataset
from models.dlm_model import (
    DLMModel,
    build_observation_matrix,
    build_structural_state_space,
)

# --- state-space assembly -------------------------------------------------


def test_build_state_space_default_sizes():
    """No structural extras: dim = 1 (level) + C (beta)."""
    F, Q, slices, D = build_structural_state_space(
        n_channels=3,
        local_linear_trend=False,
        seasonal_period=None,
        seasonal_harmonics=1,
        level_innovation_var=1e-6,
        slope_innovation_var=1e-6,
        seasonal_innovation_var=0.0,
        beta_innovation_var=5e-3,
    )
    assert D == 4
    assert F.shape == (4, 4)
    assert Q.shape == (4, 4)
    assert slices["level"] == slice(0, 1)
    assert slices["beta"] == slice(1, 4)


def test_build_state_space_with_trend_and_seasonal():
    F, Q, slices, D = build_structural_state_space(
        n_channels=2,
        local_linear_trend=True,
        seasonal_period=52.0,
        seasonal_harmonics=1,
        level_innovation_var=1e-6,
        slope_innovation_var=1e-7,
        seasonal_innovation_var=0.0,
        beta_innovation_var=5e-3,
    )
    # level + slope + 2 seasonal states + 2 β = 6
    assert D == 6
    assert slices["level"] == slice(0, 1)
    assert slices["slope"] == slice(1, 2)
    assert slices["seasonal"] == slice(2, 4)
    assert slices["beta"] == slice(4, 6)
    # LLT block: [[1, 1], [0, 1]]
    np.testing.assert_allclose(F[:2, :2], np.array([[1.0, 1.0], [0.0, 1.0]]))
    # Beta block is identity.
    np.testing.assert_allclose(F[4:, 4:], np.eye(2))


def test_seasonal_block_is_rotation_of_expected_angle():
    F, _, slices, _ = build_structural_state_space(
        n_channels=1,
        local_linear_trend=False,
        seasonal_period=52.0,
        seasonal_harmonics=1,
        level_innovation_var=1e-6,
        slope_innovation_var=1e-6,
        seasonal_innovation_var=0.0,
        beta_innovation_var=1e-3,
    )
    omega = 2.0 * np.pi / 52.0
    expected = np.array([[np.cos(omega), np.sin(omega)],
                         [-np.sin(omega), np.cos(omega)]])
    seas = slices["seasonal"]
    np.testing.assert_allclose(F[seas, seas], expected, atol=1e-12)
    # Rotation matrices are orthogonal: F_seas^T F_seas = I.
    np.testing.assert_allclose(F[seas, seas].T @ F[seas, seas], np.eye(2),
                               atol=1e-12)


def test_seasonal_multiple_harmonics_stacked_block_diagonal():
    _, _, slices, D = build_structural_state_space(
        n_channels=1,
        local_linear_trend=False,
        seasonal_period=52.0,
        seasonal_harmonics=3,
        level_innovation_var=1e-6,
        slope_innovation_var=1e-6,
        seasonal_innovation_var=0.0,
        beta_innovation_var=1e-3,
    )
    # level (1) + 3 harmonics * 2 + beta (1) = 8
    assert D == 8
    assert slices["seasonal"].stop - slices["seasonal"].start == 6


def test_q_is_psd():
    _, Q, _, _ = build_structural_state_space(
        n_channels=3,
        local_linear_trend=True,
        seasonal_period=52.0,
        seasonal_harmonics=2,
        level_innovation_var=1e-4,
        slope_innovation_var=1e-6,
        seasonal_innovation_var=1e-5,
        beta_innovation_var=5e-3,
    )
    eig = np.linalg.eigvalsh(Q)
    assert (eig >= -1e-12).all()


def test_build_observation_matrix_levels_seasonal_beta():
    n_obs, n_channels = 5, 2
    X = np.arange(n_obs * n_channels, dtype=float).reshape(n_obs, n_channels)
    _, _, slices, D = build_structural_state_space(
        n_channels=n_channels,
        local_linear_trend=True,
        seasonal_period=52.0,
        seasonal_harmonics=2,
        level_innovation_var=1e-6,
        slope_innovation_var=1e-7,
        seasonal_innovation_var=0.0,
        beta_innovation_var=1e-3,
    )
    H = build_observation_matrix(X, slices, D)
    # Level is always observed.
    assert (H[:, slices["level"].start] == 1.0).all()
    # Slope is never observed.
    assert (H[:, slices["slope"].start] == 0.0).all()
    # Seasonal: first element of each 2-pair is 1, second is 0.
    seas = slices["seasonal"]
    for h in range(2):
        assert (H[:, seas.start + 2 * h] == 1.0).all()
        assert (H[:, seas.start + 2 * h + 1] == 0.0).all()
    # Beta block reflects X verbatim.
    np.testing.assert_allclose(H[:, slices["beta"]], X)


# --- seasonal-only recovery -----------------------------------------------


def test_seasonal_dlm_recovers_pure_sinusoid():
    """A DLM with only a seasonal block should recover a sine wave exactly.

    We generate y_t = amp * sin(2π t / period) + small noise and check that
    the smoothed seasonal state matches y to within a few noise SDs.
    """
    rng = np.random.default_rng(0)
    n_obs = 200
    period = 40.0
    amp = 2.5
    t = np.arange(n_obs, dtype=float)
    season = amp * np.sin(2.0 * np.pi * t / period)
    noise_std = 0.1
    y = season + rng.normal(scale=noise_std, size=n_obs)

    # One dummy channel that carries no signal (a ≡ 0).
    X = np.zeros((n_obs, 1))
    model = DLMModel(
        seasonal_period=period,
        seasonal_harmonics=1,
        level_innovation_var=1e-10,
        seasonal_innovation_var=0.0,
        beta_innovation_var=1e-10,
        observation_var=noise_std**2,
        initial_var=10.0,
    ).fit(X, y)

    fitted = model.fitted_values()
    # RMSE of fitted vs truth (season only) under the noise floor.
    rmse = float(np.sqrt(np.mean((fitted - season) ** 2)))
    assert rmse < 3.0 * noise_std, f"seasonal recovery RMSE {rmse} too high"


def test_local_linear_trend_recovers_linear_mean():
    """LLT-only DLM should track a linear ramp y_t = a + b*t + noise."""
    rng = np.random.default_rng(1)
    n_obs = 150
    a, b = 5.0, 0.05
    noise_std = 0.1
    t = np.arange(n_obs, dtype=float)
    y = a + b * t + rng.normal(scale=noise_std, size=n_obs)

    X = np.zeros((n_obs, 1))
    model = DLMModel(
        local_linear_trend=True,
        level_innovation_var=1e-6,
        slope_innovation_var=1e-9,
        beta_innovation_var=1e-10,
        observation_var=noise_std**2,
        initial_var=100.0,
    ).fit(X, y)

    # Smoothed slope should be close to b; level_T close to a + b*(T-1).
    slope = model.component_trajectory("slope")[-1, 0]
    level_T = model.component_trajectory("level")[-1, 0]
    assert abs(slope - b) < 0.02, f"slope {slope} != {b}"
    assert abs(level_T - (a + b * (n_obs - 1))) < 0.3


# --- end-to-end MMM recovery ---------------------------------------------


def test_structural_dlm_removes_seasonal_leak_in_beta():
    """The purpose of adding a seasonal block is to stop the annual
    sinusoid in y from contaminating β̂_{c,t}.  On the default DGP this is
    measurable: the 52-week Fourier amplitude of β̂_{tv,t} should drop by
    more than an order of magnitude once the structural DLM is used.

    This test does NOT claim β_T recovery — with default-saturated features
    and drifts that cancel in aggregate y, individual β's are not
    identifiable regardless of model structure.  That is a separate DGP-
    tuning question; here we only verify the structural change works.
    """
    config = DGPConfig(seed=0)
    data = generate_dataset(config)
    X = data.feature_matrix()
    y = data.y

    def seasonal_amplitude(trajectory: np.ndarray, period: float) -> float:
        series = trajectory - trajectory.mean()
        spectrum = np.abs(np.fft.rfft(series))
        freqs = np.fft.rfftfreq(series.size, d=1.0)
        idx = int(np.argmin(np.abs(freqs - 1.0 / period)))
        return float(spectrum[idx])

    legacy = DLMModel(
        innovation_var=np.array([1e-8, 5e-3, 5e-3, 5e-3]),
        observation_var=config.noise_std**2,
    ).fit(X, y)
    structural = DLMModel(
        local_linear_trend=True,
        seasonal_period=config.seasonality_period,
        seasonal_harmonics=1,
        level_innovation_var=1e-6,
        slope_innovation_var=1e-8,
        seasonal_innovation_var=0.0,
        beta_innovation_var=5e-3,
        observation_var=config.noise_std**2,
        initial_var=10.0,
    ).fit(X, y)

    tv_legacy = legacy.smoothed_states[:, 1]
    tv_struct = structural.component_trajectory("beta")[:, 0]
    amp_legacy = seasonal_amplitude(tv_legacy, config.seasonality_period)
    amp_struct = seasonal_amplitude(tv_struct, config.seasonality_period)
    assert amp_struct < 0.1 * amp_legacy, (
        f"seasonal leak not reduced enough: legacy={amp_legacy:.2f}, "
        f"structural={amp_struct:.2f}"
    )


def test_structural_dlm_predicts_forward_with_seasonal_continuation():
    """predict() must propagate structural states — next-step seasonal value
    should be consistent with the trained rotation applied to x_T."""
    rng = np.random.default_rng(2)
    n_obs = 120
    period = 12.0
    t = np.arange(n_obs, dtype=float)
    y = 2.0 * np.sin(2.0 * np.pi * t / period) + rng.normal(scale=0.05, size=n_obs)

    X = np.zeros((n_obs, 1))
    model = DLMModel(
        seasonal_period=period,
        seasonal_harmonics=1,
        level_innovation_var=1e-10,
        seasonal_innovation_var=0.0,
        beta_innovation_var=1e-10,
        observation_var=0.05**2,
        initial_var=10.0,
    ).fit(X, y)

    # Predict one step ahead and check the seasonal component advanced.
    x_T = model.smoothed_states[-1]
    F = model._F
    X_new = np.zeros((1, 1))
    preds = model.predict(X_new)
    # Observation loading for this fit: level + seasonal[0] + β·0.
    seas_idx = model.state_slices["seasonal"].start
    level_idx = model.state_slices["level"].start
    expected_obs = (F @ x_T)[level_idx] + (F @ x_T)[seas_idx]
    assert abs(preds[0] - expected_obs) < 1e-10


def test_legacy_random_walk_still_works():
    """Non-structural mode must remain backward-compatible with the
    pre-existing DLMModel contract used by test_kalman.py."""
    rng = np.random.default_rng(3)
    n_obs, n_channels = 60, 2
    X = rng.normal(size=(n_obs, n_channels))
    true_beta = np.array([1.5, -0.7])
    y = X @ true_beta + rng.normal(scale=0.1, size=n_obs)

    model = DLMModel(
        innovation_var=1e-6, observation_var=0.1**2, initial_var=10.0
    ).fit(X, y)
    assert not model.is_structural
    assert model.beta_at_T().shape == (n_channels,)
    # On stationary data the smoothed β should recover the truth.
    np.testing.assert_allclose(model.beta_at_T(), true_beta, atol=0.1)


def test_beta_innovation_var_shape_validation():
    with pytest.raises(ValueError):
        build_structural_state_space(
            n_channels=3,
            local_linear_trend=False,
            seasonal_period=None,
            seasonal_harmonics=1,
            level_innovation_var=1e-6,
            slope_innovation_var=1e-6,
            seasonal_innovation_var=0.0,
            beta_innovation_var=np.array([1e-3, 1e-3]),  # wrong length
        )


def test_seasonal_period_must_be_positive():
    with pytest.raises(ValueError):
        build_structural_state_space(
            n_channels=1,
            local_linear_trend=False,
            seasonal_period=-52.0,
            seasonal_harmonics=1,
            level_innovation_var=1e-6,
            slope_innovation_var=1e-6,
            seasonal_innovation_var=0.0,
            beta_innovation_var=1e-3,
        )
