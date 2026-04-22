"""Shared test fixtures — synthetic datasets with stationary β trajectories.

Stationary β (drift=0, innovation_std=0) is the regime where Robyn and PyMC
are *not* misspecified, so coefficient recovery can be tested directly.
The time-varying-β regime is what separates the DLM from the others; that
is tested in `test_kalman.py` (PR #4) and at the simulation level (PR #5).
"""

from __future__ import annotations

from dgp.config import ChannelConfig, DGPConfig
from dgp.generator import SimulationData, generate_dataset


def stationary_config(
    *,
    n_weeks: int = 80,
    noise_std: float = 0.2,
    seed: int = 0,
) -> DGPConfig:
    """Two-channel DGP with constant β (no drift, no innovation, no seasonality)."""
    channels = (
        ChannelConfig(
            name="a",
            beta_0=2.0,
            beta_drift=0.0,
            beta_innovation_std=0.0,
            adstock_decay=0.3,
            hill_alpha=2.0,
            hill_gamma=0.5,
            hill_scale=100.0,
            spend_mean=80.0,
            spend_log_sigma=0.25,
        ),
        ChannelConfig(
            name="b",
            beta_0=1.0,
            beta_drift=0.0,
            beta_innovation_std=0.0,
            adstock_decay=0.2,
            hill_alpha=2.0,
            hill_gamma=0.5,
            hill_scale=100.0,
            spend_mean=60.0,
            spend_log_sigma=0.25,
        ),
    )
    return DGPConfig(
        n_weeks=n_weeks,
        channels=channels,
        baseline_mean=5.0,
        baseline_trend=0.0,
        seasonality_amplitude=0.0,
        noise_std=noise_std,
        seed=seed,
    )


def stationary_dataset(**kwargs) -> SimulationData:
    return generate_dataset(stationary_config(**kwargs))
