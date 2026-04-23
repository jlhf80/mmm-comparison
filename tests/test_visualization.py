"""Smoke tests for the visualization module.

We do not assert on pixel output — only that each plotting function runs,
returns a Figure, populates the expected number of axes, and accepts both
(T, C) trajectories and (C,) time-invariant β estimates for the coefficient
chart.  A non-interactive backend keeps these fast and display-free.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402  must precede pyplot import

import numpy as np
import pytest
from matplotlib.figure import Figure

from dgp.config import ChannelConfig, DGPConfig
from models.dlm_model import DLMModel
from simulation.results import ModelResult, SimulationResult
from visualization.allocation_plot import (
    plot_allocation_error_distribution,
    plot_allocation_shares,
)
from visualization.coef_plot import plot_coefficient_trajectories
from visualization.dlm_components_plot import plot_dlm_components
from visualization.residual_plot import plot_fit_and_residuals


def _toy_config() -> DGPConfig:
    channels = (
        ChannelConfig(
            name="a", beta_0=2.0, beta_drift=0.0, beta_innovation_std=0.0,
            adstock_decay=0.3, hill_alpha=1.5, hill_gamma=0.5, hill_scale=60.0,
            spend_mean=30.0, spend_log_sigma=0.3,
        ),
        ChannelConfig(
            name="b", beta_0=1.0, beta_drift=0.0, beta_innovation_std=0.0,
            adstock_decay=0.2, hill_alpha=1.5, hill_gamma=0.5, hill_scale=60.0,
            spend_mean=30.0, spend_log_sigma=0.3,
        ),
    )
    return DGPConfig(
        n_weeks=20, channels=channels,
        baseline_mean=0.0, baseline_trend=0.0, seasonality_amplitude=0.0,
        noise_std=0.1, seed=0,
    )


def _toy_result(seed: int) -> SimulationResult:
    config = _toy_config()
    channel_names = ("a", "b")
    true_beta_T = np.array([2.0, 1.0])
    optimal_budget = {"a": 70.0, "b": 30.0}
    models = {
        "robyn": ModelResult(
            name="robyn",
            beta_T_hat=np.array([1.8, 1.1]),
            allocated_budget={"a": 60.0, "b": 40.0},
            mape=0.12, rmse=0.3, allocation_error=0.2,
        ),
        "pymc": ModelResult(
            name="pymc",
            beta_T_hat=np.array([1.9, 1.05]),
            allocated_budget={"a": 65.0, "b": 35.0},
            mape=0.11, rmse=0.28, allocation_error=0.1,
        ),
        "dlm": ModelResult(
            name="dlm",
            beta_T_hat=np.array([2.05, 0.95]),
            allocated_budget={"a": 72.0, "b": 28.0},
            mape=0.08, rmse=0.22, allocation_error=0.04,
        ),
    }
    return SimulationResult(
        seed=seed, config=config, total_budget=100.0,
        channel_names=channel_names, true_beta_T=true_beta_T,
        optimal_budget=optimal_budget, models=models,
    )


# --- coef_plot ------------------------------------------------------------


def test_plot_coefficient_trajectories_returns_figure():
    n_weeks, n_channels = 30, 3
    true_beta = np.random.default_rng(0).normal(size=(n_weeks, n_channels))
    model_betas = {
        "dlm": np.random.default_rng(1).normal(size=(n_weeks, n_channels)),
        "robyn": np.array([1.0, 2.0, 3.0]),
    }
    fig = plot_coefficient_trajectories(true_beta, model_betas, ["a", "b", "c"])
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == n_channels
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_plot_coefficient_trajectories_rejects_wrong_length_names():
    true_beta = np.zeros((10, 2))
    with pytest.raises(ValueError):
        plot_coefficient_trajectories(true_beta, {}, ["only-one"])


def test_plot_coefficient_trajectories_rejects_mismatched_trajectory():
    true_beta = np.zeros((10, 2))
    with pytest.raises(ValueError):
        plot_coefficient_trajectories(
            true_beta, {"dlm": np.zeros((12, 2))}, ["a", "b"]
        )


def test_plot_coefficient_trajectories_accepts_bands():
    """Bands of both (T, C) and (C,) shape should render without error."""
    n_weeks, n_channels = 30, 3
    rng = np.random.default_rng(0)
    true_beta = rng.normal(size=(n_weeks, n_channels))
    dlm_mean = rng.normal(size=(n_weeks, n_channels))
    dlm_std = np.abs(rng.normal(size=(n_weeks, n_channels)))
    pymc_mean = np.array([1.0, 2.0, 3.0])
    model_betas = {"dlm": dlm_mean, "pymc": pymc_mean}
    model_bands = {
        "dlm": (dlm_mean - 1.96 * dlm_std, dlm_mean + 1.96 * dlm_std),
        "pymc": (pymc_mean - 0.5, pymc_mean + 0.5),
    }
    fig = plot_coefficient_trajectories(
        true_beta, model_betas, ["a", "b", "c"], model_bands=model_bands
    )
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == n_channels
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_plot_coefficient_trajectories_partial_bands():
    """A model without a band entry should still render as a bandless line."""
    n_weeks, n_channels = 20, 2
    rng = np.random.default_rng(1)
    true_beta = rng.normal(size=(n_weeks, n_channels))
    dlm_mean = rng.normal(size=(n_weeks, n_channels))
    dlm_std = np.full_like(dlm_mean, 0.1)
    model_betas = {"dlm": dlm_mean, "robyn": np.array([1.0, 2.0])}
    model_bands = {
        "dlm": (dlm_mean - 1.96 * dlm_std, dlm_mean + 1.96 * dlm_std)
    }
    fig = plot_coefficient_trajectories(
        true_beta, model_betas, ["a", "b"], model_bands=model_bands
    )
    try:
        assert len(fig.axes) == n_channels
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_plot_coefficient_trajectories_rejects_mismatched_band_shape():
    """Band shape must match its model's estimate shape."""
    true_beta = np.zeros((10, 2))
    model_betas = {"dlm": np.zeros((10, 2))}
    # Lower bound is (12, 2) — doesn't match (10, 2) estimate.
    bad_band = (np.zeros((12, 2)), np.zeros((12, 2)))
    with pytest.raises(ValueError):
        plot_coefficient_trajectories(
            true_beta, model_betas, ["a", "b"],
            model_bands={"dlm": bad_band},
        )


def test_plot_coefficient_trajectories_rejects_band_for_unknown_model():
    true_beta = np.zeros((10, 2))
    model_betas = {"dlm": np.zeros((10, 2))}
    with pytest.raises(ValueError):
        plot_coefficient_trajectories(
            true_beta, model_betas, ["a", "b"],
            model_bands={"ghost": (np.zeros((10, 2)), np.zeros((10, 2)))},
        )


# --- dlm_components_plot ---------------------------------------------------


def _structural_dlm(n_channels: int = 3, n_obs: int = 40, seed: int = 0) -> DLMModel:
    """Fit a small structural DLM for visualization-test fixtures."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_obs, n_channels))
    y = rng.normal(size=n_obs)
    return DLMModel(
        local_linear_trend=True,
        seasonal_period=52.0,
        seasonal_harmonics=1,
        level_innovation_var=1e-6,
        slope_innovation_var=1e-8,
        seasonal_innovation_var=0.0,
        beta_innovation_var=5e-3,
        observation_var=0.25,
        initial_var=10.0,
    ).fit(X, y)


def test_plot_dlm_components_returns_figure_with_six_axes():
    """Structural DLM fit produces exactly 6 component panels."""
    dlm = _structural_dlm(n_channels=3)
    fig = plot_dlm_components(dlm, ["a", "b", "c"])
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 6
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_plot_dlm_components_rejects_nonstructural_dlm():
    """Plain DLM lacks level/slope/seasonal — must raise clearly."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 2))
    y = rng.normal(size=30)
    dlm = DLMModel().fit(X, y)
    with pytest.raises(ValueError, match="structural"):
        plot_dlm_components(dlm, ["a", "b"])


def test_plot_dlm_components_rejects_wrong_length_names():
    """Number of channel names must match the β block width."""
    dlm = _structural_dlm(n_channels=3)
    with pytest.raises(ValueError, match="channel names"):
        plot_dlm_components(dlm, ["a", "b"])


# --- residual_plot --------------------------------------------------------


def test_plot_fit_and_residuals_returns_figure():
    rng = np.random.default_rng(0)
    y = rng.normal(size=40)
    preds = {"robyn": y + 0.1, "pymc": y + 0.05, "dlm": y - 0.02}
    fig = plot_fit_and_residuals(y, preds)
    try:
        assert isinstance(fig, Figure)
        # 2 rows (fit, residual) × 3 models = 6 axes.
        assert len(fig.axes) == 6
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_plot_fit_and_residuals_rejects_shape_mismatch():
    y = np.zeros(40)
    with pytest.raises(ValueError):
        plot_fit_and_residuals(y, {"m": np.zeros(39)})


def test_plot_fit_and_residuals_rejects_empty_predictions():
    with pytest.raises(ValueError):
        plot_fit_and_residuals(np.zeros(10), {})


# --- allocation_plot ------------------------------------------------------


def test_plot_allocation_shares_returns_figure():
    results = [_toy_result(s) for s in range(4)]
    fig = plot_allocation_shares(results)
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_plot_allocation_error_distribution_returns_figure():
    results = [_toy_result(s) for s in range(4)]
    fig = plot_allocation_error_distribution(results)
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_plot_allocation_functions_reject_empty_results():
    with pytest.raises(ValueError):
        plot_allocation_shares([])
    with pytest.raises(ValueError):
        plot_allocation_error_distribution([])
