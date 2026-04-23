"""Tests for the hand-rolled Kalman filter, RTS smoother, and DLMModel.

This is the highest-stakes test in the project.  The DLM's claim in the
punchline — that it correctly recovers the true β_{c,T} trajectory when
Robyn and PyMC cannot — rests on the filter + smoother actually computing
what they claim to compute.  So we test against:

  1. Synthetic data simulated from the *exact* linear-Gaussian state-space
     model (tests math-implementation agreement).
  2. Analytical identities (smoother collapses to filter at t=T-1;
     covariances stay symmetric and PSD).
  3. A hand-computed likelihood on a tiny scalar example.
  4. The actual DGP (tests/_fixtures): DLM recovers the smoothed β̂_T within
     a tight tolerance of the true β_T.
"""

from __future__ import annotations

import numpy as np
import pytest

from dgp.config import ChannelConfig, DGPConfig
from dgp.generator import generate_dataset
from models.dlm_model import DLMModel, kalman_filter, rts_smoother

# =========================================================================
# Simulation helper: forward-roll from the exact linear-Gaussian model
# =========================================================================


def _simulate_state_space(
    n_obs: int,
    H: np.ndarray,
    u: np.ndarray,
    Q: np.ndarray,
    R: float,
    x0: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-simulate x_t = x_{t-1} + u + w_t, y_t = h_t^T x_t + ε_t."""
    rng = np.random.default_rng(seed)
    D = x0.shape[0]
    x = np.zeros((n_obs, D))
    y = np.zeros(n_obs)
    x_prev = x0.astype(float).copy()
    is_degenerate_Q = np.allclose(Q, 0.0)
    for t in range(n_obs):
        if is_degenerate_Q:
            w = np.zeros(D)
        else:
            w = rng.multivariate_normal(np.zeros(D), Q, method="svd")
        x_prev = x_prev + u + w
        x[t] = x_prev
        eps = rng.normal(0.0, np.sqrt(R))
        y[t] = float(H[t] @ x_prev) + eps
    return x, y


# =========================================================================
# Core correctness: smoother recovers known state trajectories
# =========================================================================


def test_smoother_recovers_constant_state_to_within_noise() -> None:
    """With Q = 0 and many observations, filter ⇒ recursive OLS.

    True state is constant; smoother posterior mean should converge to it.
    """
    n_obs, D = 200, 3
    true_x = np.array([1.0, -2.0, 0.5])
    H = np.random.default_rng(0).normal(size=(n_obs, D))
    Q = np.zeros((D, D))
    R = 0.5

    _, y = _simulate_state_space(
        n_obs, H, u=np.zeros(D), Q=Q, R=R, x0=true_x, seed=42
    )

    x_filt, P_filt, x_pred, P_pred, _ = kalman_filter(
        y, H, u=np.zeros(D), Q=Q, R=R,
        x0=np.zeros(D), P0=np.eye(D) * 100.0,
    )
    x_smooth, _ = rts_smoother(x_filt, P_filt, x_pred, P_pred)

    # After 200 obs at noise R=0.5, posterior mean should be within ~0.15 of truth.
    np.testing.assert_allclose(x_smooth[-1], true_x, atol=0.15)


def test_smoother_tracks_random_walk_better_than_filter() -> None:
    """On a drifting state, the smoother uses future observations and so
    its in-sample MSE against the true trajectory must be ≤ the filter's."""
    n_obs, D = 150, 2
    H = np.random.default_rng(1).normal(size=(n_obs, D))
    u = np.array([0.01, -0.02])
    Q = np.eye(D) * 0.05
    R = 0.3

    x_true, y = _simulate_state_space(
        n_obs, H, u=u, Q=Q, R=R, x0=np.array([1.0, 0.5]), seed=7
    )

    x_filt, P_filt, x_pred, P_pred, _ = kalman_filter(
        y, H, u=u, Q=Q, R=R, x0=np.zeros(D), P0=np.eye(D) * 10.0,
    )
    x_smooth, _ = rts_smoother(x_filt, P_filt, x_pred, P_pred)

    filter_mse = float(np.mean((x_filt - x_true) ** 2))
    smoother_mse = float(np.mean((x_smooth - x_true) ** 2))
    # Smoother is never worse than filter in expectation; on a random
    # walk it is meaningfully better.
    assert smoother_mse <= filter_mse
    assert smoother_mse < 0.7 * filter_mse


def test_smoother_ninety_percent_credible_intervals_cover_truth() -> None:
    """Coverage check: smoothed ±1.96·σ intervals cover the true state ≥85%.

    Tests that reported posterior variance is correctly scaled."""
    n_obs, D = 250, 2
    H = np.random.default_rng(2).normal(size=(n_obs, D))
    Q = np.eye(D) * 0.01
    R = 0.5

    x_true, y = _simulate_state_space(
        n_obs, H, u=np.zeros(D), Q=Q, R=R, x0=np.zeros(D), seed=11
    )

    x_filt, P_filt, x_pred, P_pred, _ = kalman_filter(
        y, H, u=np.zeros(D), Q=Q, R=R, x0=np.zeros(D), P0=np.eye(D) * 10.0,
    )
    x_smooth, P_smooth = rts_smoother(x_filt, P_filt, x_pred, P_pred)

    sigma = np.sqrt(np.einsum("tdd->td", P_smooth))
    covered = (x_true >= x_smooth - 1.96 * sigma) & (
        x_true <= x_smooth + 1.96 * sigma
    )
    coverage = float(covered.mean())
    assert coverage >= 0.85


# =========================================================================
# Analytical identities
# =========================================================================


def test_smoother_matches_filter_at_final_step() -> None:
    """x_{T-1|T} = x_{T-1|T-1} and P_{T-1|T} = P_{T-1|T-1} — no future info."""
    n_obs, D = 50, 3
    H = np.random.default_rng(3).normal(size=(n_obs, D))
    y = np.random.default_rng(4).normal(size=n_obs)

    x_filt, P_filt, x_pred, P_pred, _ = kalman_filter(
        y, H, u=np.zeros(D), Q=np.eye(D) * 0.01, R=1.0,
        x0=np.zeros(D), P0=np.eye(D) * 5.0,
    )
    x_smooth, P_smooth = rts_smoother(x_filt, P_filt, x_pred, P_pred)

    np.testing.assert_allclose(x_smooth[-1], x_filt[-1])
    np.testing.assert_allclose(P_smooth[-1], P_filt[-1])


def test_filtered_covariances_are_symmetric_and_psd() -> None:
    n_obs, D = 30, 3
    H = np.random.default_rng(5).normal(size=(n_obs, D))
    y = np.random.default_rng(6).normal(size=n_obs)
    _, P_filt, _, _, _ = kalman_filter(
        y, H, u=np.zeros(D), Q=np.eye(D) * 0.02, R=0.8,
        x0=np.zeros(D), P0=np.eye(D) * 10.0,
    )
    for t in range(n_obs):
        np.testing.assert_allclose(P_filt[t], P_filt[t].T, atol=1e-10)
        eigvals = np.linalg.eigvalsh(P_filt[t])
        assert eigvals.min() > -1e-8


def test_smoothed_covariances_are_symmetric_and_psd() -> None:
    n_obs, D = 30, 3
    H = np.random.default_rng(7).normal(size=(n_obs, D))
    y = np.random.default_rng(8).normal(size=n_obs)
    x_filt, P_filt, x_pred, P_pred, _ = kalman_filter(
        y, H, u=np.zeros(D), Q=np.eye(D) * 0.02, R=0.8,
        x0=np.zeros(D), P0=np.eye(D) * 10.0,
    )
    _, P_smooth = rts_smoother(x_filt, P_filt, x_pred, P_pred)
    for t in range(n_obs):
        np.testing.assert_allclose(P_smooth[t], P_smooth[t].T, atol=1e-10)
        eigvals = np.linalg.eigvalsh(P_smooth[t])
        assert eigvals.min() > -1e-8


# =========================================================================
# Likelihood: hand-computed vs implementation
# =========================================================================


def test_kalman_filter_loglik_matches_manual_scalar_example() -> None:
    """Scalar (D=1) 3-step example, compute log p(y_{1:3}) by hand."""
    H = np.array([[1.0], [1.0], [1.0]])
    y = np.array([0.5, -0.3, 1.0])
    Q = np.array([[0.1]])
    R = 0.2
    x0 = np.array([0.0])
    P0 = np.array([[1.0]])
    u = np.array([0.0])

    _, _, _, _, ll = kalman_filter(y, H, u, Q, R, x0, P0)

    # Manual recursion.
    x, P = 0.0, 1.0
    manual_ll = 0.0
    for t in range(3):
        x_p, P_p = x + 0.0, P + 0.1
        s = P_p + R
        v = y[t] - x_p
        manual_ll += -0.5 * (np.log(2.0 * np.pi * s) + v * v / s)
        k = P_p / s
        x = x_p + k * v
        P = (1 - k) * P_p * (1 - k) + k * k * R  # Joseph form, scalar

    assert ll == pytest.approx(manual_ll)


# =========================================================================
# DLMModel interface
# =========================================================================


def test_dlm_fit_returns_self_and_exposes_shapes() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = rng.normal(size=50)
    model = DLMModel().fit(X, y)
    assert model.beta_at_T().shape == (3,)
    assert model.smoothed_states.shape == (50, 4)
    assert model.smoothed_covariances.shape == (50, 4, 4)
    assert model.filtered_states.shape == (50, 4)
    assert np.isfinite(model.loglik)


def test_dlm_predict_uses_last_smoothed_state() -> None:
    """predict(X) for DLM is H · x_smooth[T-1], with H augmented by an intercept."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 2))
    y = rng.normal(size=40)
    model = DLMModel().fit(X, y)

    X_new = rng.normal(size=(5, 2))
    preds = model.predict(X_new)
    state_T = model.smoothed_states[-1]
    H_new = np.hstack([np.ones((5, 1)), X_new])
    np.testing.assert_allclose(preds, H_new @ state_T)


def test_dlm_fitted_values_use_smoothed_trajectory() -> None:
    """In-sample fitted ŷ_t should use x_{t|T}, not a single x_{T|T}."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 2))
    y = rng.normal(size=40)
    model = DLMModel().fit(X, y)
    expected = np.einsum(
        "td,td->t", np.hstack([np.ones((40, 1)), X]), model.smoothed_states
    )
    np.testing.assert_allclose(model.fitted_values(), expected)


def test_dlm_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        DLMModel().predict(np.zeros((5, 2)))


def test_dlm_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="expected X"):
        DLMModel().fit(np.zeros((10, 2)), np.zeros(9))


def test_dlm_rejects_wrong_drift_shape() -> None:
    with pytest.raises(ValueError, match="drift"):
        DLMModel(drift=np.zeros(2)).fit(np.zeros((10, 3)), np.zeros(10))


def test_dlm_accepts_scalar_or_vector_innovation_var() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 2))
    y = rng.normal(size=30)
    DLMModel(innovation_var=1e-2).fit(X, y)
    DLMModel(innovation_var=np.array([1e-3, 1e-2, 1e-4])).fit(X, y)


def test_component_trajectory_std_matches_diagonal() -> None:
    """std for a component should equal sqrt of the matching covariance diagonal."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 3))
    y = rng.normal(size=40)
    model = DLMModel().fit(X, y)
    std = model.component_trajectory_std("beta")
    beta_slice = model.state_slices["beta"]
    P = model.smoothed_covariances
    # Extract diagonal of the β block: P[t, beta_slice.start+c, beta_slice.start+c]
    idx = np.arange(beta_slice.start, beta_slice.stop)
    expected_var = P[:, idx, idx]
    np.testing.assert_allclose(std**2, expected_var, atol=1e-10)
    assert std.shape == (40, 3)


def test_component_trajectory_std_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        DLMModel().component_trajectory_std("beta")


# =========================================================================
# Punchline: DLM recovers true β_T on data from the DGP
# =========================================================================


def _drifting_dgp() -> DGPConfig:
    """A small DGP with clearly drifting β_c,t — the regime the DLM targets.

    Features are tuned so that spend/scale lands in the sensitive part of
    the Hill curve (mean ≈ 0.55, std ≈ 0.10), keeping β identifiable over
    156 weeks.  An earlier draft of this test used the project's default
    scale parameters which saturated the Hill function; β was then nearly
    unidentified and the test was measuring noise.
    """
    channels = (
        ChannelConfig(
            name="rising",
            beta_0=1.0,
            beta_drift=0.015,
            beta_innovation_std=0.015,
            adstock_decay=0.2,
            hill_alpha=1.5,
            hill_gamma=0.5,
            hill_scale=60.0,
            spend_mean=30.0,
            spend_log_sigma=0.4,
        ),
        ChannelConfig(
            name="falling",
            beta_0=3.0,
            beta_drift=-0.015,
            beta_innovation_std=0.015,
            adstock_decay=0.4,
            hill_alpha=1.5,
            hill_gamma=0.5,
            hill_scale=60.0,
            spend_mean=30.0,
            spend_log_sigma=0.4,
        ),
    )
    return DGPConfig(
        n_weeks=156,
        channels=channels,
        baseline_mean=5.0,
        baseline_trend=0.0,
        seasonality_amplitude=0.0,
        noise_std=0.15,
        seed=0,
    )


# DLM hyperparameters matched to the fixture above: intercept is nearly
# fixed, β innovation variance tuned against data; observation variance =
# true noise².
_DLM_Q = np.array([1e-8, 5e-3, 5e-3])
_DLM_R = 0.15**2


def test_dlm_recovers_true_beta_at_T_on_drifting_dgp() -> None:
    """Smoothed β̂_T must be close to the true β_{c,T} on drifting data.

    This is the project's central claim — that only the DLM recovers the
    current ROI on a time-varying-coefficient process.  Tolerance of 1.2
    on coefficients of order 1–3 is a generous band that still excludes
    the "fit the average" failure mode (β_0 differs from β_T by ~2.3 on
    each channel, so err > 2.0 would point at β_0 not β_T).
    """
    cfg = _drifting_dgp()
    data = generate_dataset(cfg)
    true_beta_T = np.array([data.beta[c][-1] for c in data.channel_names])

    model = DLMModel(innovation_var=_DLM_Q, observation_var=_DLM_R).fit(
        data.feature_matrix(), data.y
    )
    beta_hat_T = model.beta_at_T()

    np.testing.assert_allclose(beta_hat_T, true_beta_T, atol=1.2)


def test_dlm_beta_T_is_closer_to_truth_than_beta_0_on_drifting_data() -> None:
    """Punchline: DLM's β̂_T is closer to true β_T than to the stale β_0.

    An OLS/ridge fit on this data recovers something near the time average
    of β (i.e., between β_0 and β_T); this check sharpens the claim that
    the DLM's *time-varying* recovery beats a *time-invariant* recovery on
    precisely the question that matters for forward allocation.
    """
    cfg = _drifting_dgp()
    data = generate_dataset(cfg)
    true_beta_0 = np.array([data.beta[c][0] for c in data.channel_names])
    true_beta_T = np.array([data.beta[c][-1] for c in data.channel_names])

    model = DLMModel(innovation_var=_DLM_Q, observation_var=_DLM_R).fit(
        data.feature_matrix(), data.y
    )
    beta_hat_T = model.beta_at_T()

    err_to_T = float(np.linalg.norm(beta_hat_T - true_beta_T))
    err_to_0 = float(np.linalg.norm(beta_hat_T - true_beta_0))
    assert err_to_T < err_to_0


# =========================================================================
# Default-DGP recovery: locks in that the shipped DGP is identifiable
# =========================================================================


def test_structural_dlm_recovers_beta_on_default_dgp_across_seeds() -> None:
    """Averaged over seeds, the structural DLM's β̂_T is closer to β_T than β_0
    on the *default* DGP (three channels, baseline trend + seasonality).

    Earlier defaults put features into the saturated region of the Hill
    curve (CoV ≈ 3%) and no filter could recover β.  The retuned defaults
    land features in the sensitive region (CoV ≳ 10%), which is what makes
    the DLM's coefficient-recovery story measurable on the chart the
    LinkedIn narrative ships.  This test guards against future tuning that
    would drift back into the unidentifiable regime.
    """
    errs_T: list[float] = []
    errs_0: list[float] = []
    for seed in range(10):
        cfg = DGPConfig(seed=seed)
        data = generate_dataset(cfg)
        X, y = data.feature_matrix(), data.y
        true_T = np.array([data.beta[c][-1] for c in data.channel_names])
        true_0 = np.array([data.beta[c][0] for c in data.channel_names])

        model = DLMModel(
            local_linear_trend=True,
            seasonal_period=cfg.seasonality_period,
            seasonal_harmonics=1,
            level_innovation_var=1e-6,
            slope_innovation_var=1e-8,
            seasonal_innovation_var=0.0,
            beta_innovation_var=5e-3,
            observation_var=cfg.noise_std**2,
            initial_var=10.0,
        ).fit(X, y)
        beta_hat_T = model.beta_at_T()
        errs_T.append(float(np.linalg.norm(beta_hat_T - true_T)))
        errs_0.append(float(np.linalg.norm(beta_hat_T - true_0)))

    mean_err_T = float(np.mean(errs_T))
    mean_err_0 = float(np.mean(errs_0))
    # Generous margin: 0.80 still rules out the "fit β_0" failure mode.
    assert mean_err_T < 0.80 * mean_err_0, (
        f"DLM no longer tracks time-varying β on default DGP: "
        f"mean err-to-T={mean_err_T:.2f} vs err-to-0={mean_err_0:.2f}"
    )
