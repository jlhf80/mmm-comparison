"""Parallel Monte Carlo sweep over seeds.

Each seed is an independent DGP realization + fit of all three models, so
they parallelize trivially across a `ProcessPoolExecutor`.  `run_monte_carlo`
returns a list of `SimulationResult`s in the seed order passed in, not in
completion order — stable ordering matters when downstream code joins
results against a seed index for per-run diagnostics.

`summarize_allocation_error` aggregates the per-seed allocation errors
into the per-model distribution that anchors the project's headline chart.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from dgp.config import DGPConfig
from simulation.results import SimulationResult
from simulation.runner import run_single


def run_monte_carlo(
    seeds: Sequence[int],
    config: DGPConfig,
    total_budget: float,
    *,
    n_workers: int | None = None,
    pymc_kwargs: dict[str, Any] | None = None,
    robyn_kwargs: dict[str, Any] | None = None,
    dlm_kwargs: dict[str, Any] | None = None,
) -> list[SimulationResult]:
    """Run `run_single` over `seeds`, in parallel, preserving input order.

    `n_workers=None` defers to `ProcessPoolExecutor`'s default (CPU count).
    Pass `n_workers=1` for in-process execution (useful under `pytest` or
    a debugger, since forked workers hide stack traces).
    """
    if n_workers == 1:
        return [
            run_single(
                s, config, total_budget,
                pymc_kwargs=pymc_kwargs,
                robyn_kwargs=robyn_kwargs,
                dlm_kwargs=dlm_kwargs,
            )
            for s in seeds
        ]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(
                run_single, int(s), config, total_budget,
                pymc_kwargs=pymc_kwargs,
                robyn_kwargs=robyn_kwargs,
                dlm_kwargs=dlm_kwargs,
            )
            for s in seeds
        ]
        return [f.result() for f in futures]


def summarize_allocation_error(
    results: Sequence[SimulationResult],
) -> dict[str, np.ndarray]:
    """Collect per-model allocation errors across seeds.

    Returns a dict mapping model name → 1-D array of length `len(results)`
    containing each seed's allocation error.  Callers can `.mean()` /
    `.std()` / plot distributions directly.
    """
    if not results:
        return {}
    model_names = list(results[0].models.keys())
    return {
        name: np.array([r.models[name].allocation_error for r in results])
        for name in model_names
    }
