"""Single-seed pipeline: generate DGP → fit Robyn / PyMC / DLM → allocate.

This is the per-seed unit of work that `monte_carlo.run_monte_carlo`
distributes across a process pool.  Everything needed to evaluate the
three models against ground truth is computed here:

  1. `generate_dataset(config)` produces features, β_{c,t}, and y.
  2. Ground truth β_T and the optimal budget are computed from the known
     coefficient trajectory.
  3. Each fitter is trained on the *same* pre-transformed feature matrix.
  4. Each fitter's β̂_T drives `allocate_budget`, which is compared to the
     ground-truth optimum.

Keeping the DLM's innovation variance proportional to the true noise (and
structure) is a deliberate concession — in a real workflow the analyst
would estimate Q/R, here we grant the DLM well-specified hyperparameters
since the question we are answering is *not* "can you tune a Kalman
filter?" but "does correctly specified time variation beat misspecified
time-invariance on the allocation problem?".
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from dgp.config import DGPConfig
from dgp.generator import generate_dataset
from models.dlm_model import DLMModel
from models.metrics import allocation_error, mape, rmse
from models.optimizer import allocate_budget
from models.pymc_model import PyMCModel
from models.robyn_model import RobynModel
from simulation.results import ModelResult, SimulationResult

# Default PyMC / Robyn settings for production runs.  Tests override these
# to keep CI under a minute.
DEFAULT_PYMC_KWARGS: dict[str, Any] = dict(
    draws=1000, tune=1000, chains=2, target_accept=0.9
)
DEFAULT_ROBYN_KWARGS: dict[str, Any] = dict(nevergrad_budget=40)


def _default_dlm_kwargs(config: DGPConfig) -> dict[str, Any]:
    """Structural DLM matching the DGP's α_t: local linear trend + annual Fourier.

    The runner knows what the DGP looks like (linear baseline trend + annual
    seasonality), so it hands the DLM a matching structural spec.  This
    routes α_t's variance into dedicated level/slope/seasonal states
    instead of letting it bleed into the β block and corrupt the
    coefficient-recovery story.  `seasonal_innovation_var=0` makes the
    seasonal shape deterministic (fixed amplitude/phase, which is what the
    DGP does too).
    """
    return dict(
        local_linear_trend=True,
        seasonal_period=config.seasonality_period,
        seasonal_harmonics=1,
        level_innovation_var=1e-6,
        slope_innovation_var=1e-8,
        seasonal_innovation_var=0.0,
        beta_innovation_var=5e-3,
        observation_var=config.noise_std**2,
        initial_var=10.0,
    )


def run_single(
    seed: int,
    config: DGPConfig,
    total_budget: float,
    *,
    pymc_kwargs: dict[str, Any] | None = None,
    robyn_kwargs: dict[str, Any] | None = None,
    dlm_kwargs: dict[str, Any] | None = None,
) -> SimulationResult:
    """Run the full pipeline for one seed and return a `SimulationResult`.

    `seed` overrides `config.seed` so a caller can sweep seeds without
    mutating a shared config object.  PyMC/Robyn/DLM kwargs are exposed so
    tests can run with cheap settings.
    """
    run_config = replace(config, seed=seed)
    data = generate_dataset(run_config)
    X = data.feature_matrix()
    y = data.y
    channel_names = tuple(data.channel_names)
    channels = run_config.channels

    true_beta_T = np.array([data.beta[c][-1] for c in channel_names])
    optimal_budget = allocate_budget(true_beta_T, channels, total_budget)

    robyn = RobynModel(seed=seed, **(robyn_kwargs or DEFAULT_ROBYN_KWARGS))
    pymc = PyMCModel(seed=seed, **(pymc_kwargs or DEFAULT_PYMC_KWARGS))
    dlm = DLMModel(**(dlm_kwargs or _default_dlm_kwargs(run_config)))

    model_results: dict[str, ModelResult] = {}
    for name, m in (("robyn", robyn), ("pymc", pymc), ("dlm", dlm)):
        m.fit(X, y)
        preds = m.predict(X)
        beta_T_hat = m.beta_at_T()
        allocated = allocate_budget(beta_T_hat, channels, total_budget)
        model_results[name] = ModelResult(
            name=name,
            beta_T_hat=beta_T_hat,
            allocated_budget=allocated,
            mape=mape(y, preds),
            rmse=rmse(y, preds),
            allocation_error=allocation_error(allocated, optimal_budget),
        )

    return SimulationResult(
        seed=seed,
        config=run_config,
        total_budget=total_budget,
        channel_names=channel_names,
        true_beta_T=true_beta_T,
        optimal_budget=optimal_budget,
        models=model_results,
    )
