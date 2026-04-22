"""Tests for `simulation.runner.run_single`."""

from __future__ import annotations

import numpy as np
import pytest

from dgp.config import ChannelConfig, DGPConfig
from simulation.results import ModelResult, SimulationResult
from simulation.runner import run_single

# Fast-path model kwargs — shrink PyMC + Nevergrad so CI stays under a minute.
_FAST_PYMC = dict(draws=100, tune=100, chains=1, target_accept=0.9)
_FAST_ROBYN = dict(nevergrad_budget=5)


def _small_config() -> DGPConfig:
    """Two-channel stationary-β DGP with few weeks, tuned for fast fits."""
    channels = (
        ChannelConfig(
            name="a",
            beta_0=2.0,
            beta_drift=0.0,
            beta_innovation_std=0.0,
            adstock_decay=0.3,
            hill_alpha=1.5,
            hill_gamma=0.5,
            hill_scale=60.0,
            spend_mean=30.0,
            spend_log_sigma=0.3,
        ),
        ChannelConfig(
            name="b",
            beta_0=1.0,
            beta_drift=0.0,
            beta_innovation_std=0.0,
            adstock_decay=0.2,
            hill_alpha=1.5,
            hill_gamma=0.5,
            hill_scale=60.0,
            spend_mean=30.0,
            spend_log_sigma=0.3,
        ),
    )
    return DGPConfig(
        n_weeks=40,
        channels=channels,
        baseline_mean=5.0,
        baseline_trend=0.0,
        seasonality_amplitude=0.0,
        noise_std=0.2,
        seed=0,
    )


@pytest.fixture(scope="module")
def small_run() -> SimulationResult:
    return run_single(
        seed=0,
        config=_small_config(),
        total_budget=100.0,
        pymc_kwargs=_FAST_PYMC,
        robyn_kwargs=_FAST_ROBYN,
    )


def test_run_single_returns_simulation_result(small_run: SimulationResult) -> None:
    assert isinstance(small_run, SimulationResult)
    assert small_run.seed == 0
    assert small_run.total_budget == 100.0
    assert small_run.channel_names == ("a", "b")


def test_run_single_includes_all_three_fitters(small_run: SimulationResult) -> None:
    assert set(small_run.models.keys()) == {"robyn", "pymc", "dlm"}
    for result in small_run.models.values():
        assert isinstance(result, ModelResult)


def test_run_single_true_beta_T_has_channel_shape(small_run: SimulationResult) -> None:
    assert small_run.true_beta_T.shape == (2,)


def test_run_single_optimal_budget_sums_to_total(small_run: SimulationResult) -> None:
    assert sum(small_run.optimal_budget.values()) == pytest.approx(
        small_run.total_budget, abs=1e-4
    )
    assert set(small_run.optimal_budget.keys()) == {"a", "b"}


def test_run_single_each_model_allocates_full_budget(small_run: SimulationResult) -> None:
    for result in small_run.models.values():
        assert sum(result.allocated_budget.values()) == pytest.approx(
            small_run.total_budget, abs=1e-4
        )
        assert set(result.allocated_budget.keys()) == {"a", "b"}


def test_run_single_each_model_beta_T_hat_shape(small_run: SimulationResult) -> None:
    for result in small_run.models.values():
        assert result.beta_T_hat.shape == (2,)


def test_run_single_metrics_are_finite(small_run: SimulationResult) -> None:
    for result in small_run.models.values():
        assert np.isfinite(result.mape)
        assert np.isfinite(result.rmse)
        assert np.isfinite(result.allocation_error)
        assert result.allocation_error >= 0


def test_run_single_same_seed_produces_identical_dgp_truth() -> None:
    """Re-running with the same seed must produce identical ground truth.

    Model-fit metrics may vary if the fitters have internal non-determinism
    (NUTS random init, Nevergrad RNG), but the DGP and the true β_T are
    fully determined by the seed.
    """
    cfg = _small_config()
    r1 = run_single(
        seed=7, config=cfg, total_budget=50.0,
        pymc_kwargs=_FAST_PYMC, robyn_kwargs=_FAST_ROBYN,
    )
    r2 = run_single(
        seed=7, config=cfg, total_budget=50.0,
        pymc_kwargs=_FAST_PYMC, robyn_kwargs=_FAST_ROBYN,
    )
    np.testing.assert_array_equal(r1.true_beta_T, r2.true_beta_T)
    assert r1.optimal_budget == pytest.approx(r2.optimal_budget)


def test_run_single_overrides_config_seed() -> None:
    """Passing `seed=X` must override `config.seed` without mutating config."""
    cfg = _small_config()
    assert cfg.seed == 0
    result = run_single(
        seed=99, config=cfg, total_budget=50.0,
        pymc_kwargs=_FAST_PYMC, robyn_kwargs=_FAST_ROBYN,
    )
    assert result.config.seed == 99
    assert cfg.seed == 0  # original untouched
