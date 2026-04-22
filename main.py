"""CLI entry point: run the Monte Carlo sweep and render the three charts.

Usage:
    python main.py --n-seeds 50 --output results/

Writes three PNGs and a small summary.json into the output directory:

    coef_trajectories.png       — estimated vs true β_{c,t}, seed 0
    residual_structure.png      — fit vs actual + residuals, seed 0
    allocation_shares.png       — mean budget share per channel, MC sweep
    allocation_error.png        — per-seed L1 share error, MC sweep
    summary.json                — per-model allocation error summary stats

The single-seed "intuition" charts (coef + residual) are generated from seed
0 so every run of the CLI produces a directly comparable figure; the
allocation charts sweep `--n-seeds` starting at 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dgp.config import DGPConfig
from dgp.generator import generate_dataset
from models.dlm_model import DLMModel
from models.pymc_model import PyMCModel
from models.robyn_model import RobynModel
from simulation.monte_carlo import run_monte_carlo, summarize_allocation_error
from simulation.runner import DEFAULT_PYMC_KWARGS, DEFAULT_ROBYN_KWARGS, _default_dlm_kwargs
from visualization.allocation_plot import (
    plot_allocation_error_distribution,
    plot_allocation_shares,
)
from visualization.coef_plot import plot_coefficient_trajectories
from visualization.residual_plot import plot_fit_and_residuals


def _intuition_charts(
    config: DGPConfig,
    seed: int,
    output_dir: Path,
    *,
    pymc_kwargs: dict,
    robyn_kwargs: dict,
) -> None:
    """Fit all three models once and save the coef + residual charts."""
    from dataclasses import replace

    run_config = replace(config, seed=seed)
    data = generate_dataset(run_config)
    X = data.feature_matrix()
    y = data.y
    channel_names = data.channel_names

    robyn = RobynModel(seed=seed, **robyn_kwargs).fit(X, y)
    pymc = PyMCModel(seed=seed, **pymc_kwargs).fit(X, y)
    dlm = DLMModel(**_default_dlm_kwargs(run_config)).fit(X, y)

    true_beta = data.beta_matrix()
    model_betas = {
        "robyn": robyn.beta_at_T(),
        "pymc": pymc.beta_at_T(),
        "dlm": dlm.smoothed_states[:, 1:],
    }
    coef_fig = plot_coefficient_trajectories(true_beta, model_betas, channel_names)
    coef_fig.savefig(output_dir / "coef_trajectories.png", dpi=150)

    predictions = {
        "robyn": robyn.predict(X),
        "pymc": pymc.predict(X),
        "dlm": dlm.fitted_values(),
    }
    resid_fig = plot_fit_and_residuals(y, predictions)
    resid_fig.savefig(output_dir / "residual_structure.png", dpi=150)


def _monte_carlo_charts(
    config: DGPConfig,
    n_seeds: int,
    total_budget: float,
    output_dir: Path,
    *,
    n_workers: int | None,
    pymc_kwargs: dict,
    robyn_kwargs: dict,
) -> None:
    """Run MC sweep and save allocation-share + error-distribution charts."""
    seeds = list(range(n_seeds))
    results = run_monte_carlo(
        seeds,
        config,
        total_budget,
        n_workers=n_workers,
        pymc_kwargs=pymc_kwargs,
        robyn_kwargs=robyn_kwargs,
    )

    shares_fig = plot_allocation_shares(results)
    shares_fig.savefig(output_dir / "allocation_shares.png", dpi=150)

    err_fig = plot_allocation_error_distribution(results)
    err_fig.savefig(output_dir / "allocation_error.png", dpi=150)

    errors = summarize_allocation_error(results)
    summary = {
        name: {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "n": int(arr.size),
        }
        for name, arr in errors.items()
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=50,
                        help="Number of Monte Carlo seeds (default: 50)")
    parser.add_argument("--output", type=Path, default=Path("results"),
                        help="Directory to write charts into (default: results/)")
    parser.add_argument("--total-budget", type=float, default=500.0,
                        help="Total weekly budget for allocation at t=T")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Process pool size; 1 runs in-process")
    parser.add_argument("--fast", action="store_true",
                        help="Use cheap PyMC/Robyn settings for a quick smoke run")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)

    if args.fast:
        pymc_kwargs = dict(draws=200, tune=200, chains=1, target_accept=0.9)
        robyn_kwargs = dict(nevergrad_budget=10)
    else:
        pymc_kwargs = dict(DEFAULT_PYMC_KWARGS)
        robyn_kwargs = dict(DEFAULT_ROBYN_KWARGS)

    config = DGPConfig()

    _intuition_charts(
        config,
        seed=0,
        output_dir=args.output,
        pymc_kwargs=pymc_kwargs,
        robyn_kwargs=robyn_kwargs,
    )
    _monte_carlo_charts(
        config,
        n_seeds=args.n_seeds,
        total_budget=args.total_budget,
        output_dir=args.output,
        n_workers=args.n_workers,
        pymc_kwargs=pymc_kwargs,
        robyn_kwargs=robyn_kwargs,
    )


if __name__ == "__main__":
    main()
