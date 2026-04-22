"""Result containers for single runs and aggregated Monte Carlo sweeps.

`ModelResult` captures everything a downstream chart or table needs from a
single fitted model: its β̂_T, the budget it implies, and its fit / decision
metrics vs the ground truth.  `SimulationResult` bundles one per fitter,
plus the ground-truth β_T and the optimal budget the DGP itself would
choose.  The design is flat and numpy-only so results pickle cleanly
across a `ProcessPoolExecutor` boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dgp.config import DGPConfig


@dataclass(frozen=True)
class ModelResult:
    """Artifacts from a single fitted model within one simulation run."""

    name: str
    beta_T_hat: np.ndarray                 # (C,)
    allocated_budget: dict[str, float]     # channel → spend
    mape: float
    rmse: float
    allocation_error: float


@dataclass(frozen=True)
class SimulationResult:
    """One seed's worth of output: ground truth + all fitters."""

    seed: int
    config: DGPConfig
    total_budget: float
    channel_names: tuple[str, ...]
    true_beta_T: np.ndarray                # (C,)
    optimal_budget: dict[str, float]
    models: dict[str, ModelResult] = field(default_factory=dict)
