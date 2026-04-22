"""Tests for `simulation.monte_carlo.run_monte_carlo` and summary stats."""

from __future__ import annotations

import numpy as np
import pytest

from dgp.config import ChannelConfig, DGPConfig
from simulation.monte_carlo import run_monte_carlo, summarize_allocation_error

_FAST_PYMC = dict(draws=100, tune=100, chains=1, target_accept=0.9)
_FAST_ROBYN = dict(nevergrad_budget=5)


def _tiny_config() -> DGPConfig:
    """Minimal DGP for CI speed: 30 weeks, 2 channels, stationary."""
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
        n_weeks=30,
        channels=channels,
        baseline_mean=5.0,
        baseline_trend=0.0,
        seasonality_amplitude=0.0,
        noise_std=0.2,
        seed=0,
    )


def test_run_monte_carlo_returns_one_result_per_seed() -> None:
    seeds = [1, 2, 3]
    results = run_monte_carlo(
        seeds, _tiny_config(), total_budget=80.0,
        n_workers=1,  # in-process to keep test traces readable
        pymc_kwargs=_FAST_PYMC, robyn_kwargs=_FAST_ROBYN,
    )
    assert len(results) == 3
    assert [r.seed for r in results] == seeds


def test_run_monte_carlo_preserves_seed_order() -> None:
    """Input order must be preserved even though runs are independent."""
    seeds = [42, 7, 19, 3]
    results = run_monte_carlo(
        seeds, _tiny_config(), total_budget=80.0,
        n_workers=1, pymc_kwargs=_FAST_PYMC, robyn_kwargs=_FAST_ROBYN,
    )
    assert [r.seed for r in results] == seeds


def test_summarize_allocation_error_returns_per_model_arrays() -> None:
    seeds = [1, 2]
    results = run_monte_carlo(
        seeds, _tiny_config(), total_budget=80.0,
        n_workers=1, pymc_kwargs=_FAST_PYMC, robyn_kwargs=_FAST_ROBYN,
    )
    summary = summarize_allocation_error(results)
    assert set(summary.keys()) == {"robyn", "pymc", "dlm"}
    for name, arr in summary.items():
        assert arr.shape == (2,)
        assert np.all(np.isfinite(arr))
        assert np.all(arr >= 0), f"{name} has negative allocation error"


def test_summarize_allocation_error_on_empty_input() -> None:
    assert summarize_allocation_error([]) == {}


@pytest.mark.parametrize("total_budget", [50.0, 200.0])
def test_run_monte_carlo_respects_total_budget(total_budget: float) -> None:
    results = run_monte_carlo(
        [0, 1], _tiny_config(), total_budget=total_budget,
        n_workers=1, pymc_kwargs=_FAST_PYMC, robyn_kwargs=_FAST_ROBYN,
    )
    for r in results:
        assert r.total_budget == total_budget
        assert sum(r.optimal_budget.values()) == pytest.approx(total_budget, abs=1e-4)
        for model in r.models.values():
            assert sum(model.allocated_budget.values()) == pytest.approx(
                total_budget, abs=1e-4
            )
