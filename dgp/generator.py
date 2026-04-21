"""Synthetic dataset generator for the MMM comparison.

Generates:
  - raw weekly spend per channel
  - pre-transformed adstock+Hill features a_{c,t}
  - time-varying coefficients β_{c,t} (random walk with drift)
  - baseline α_t (trend + annual seasonality)
  - observations y_t = α_t + Σ_c β_{c,t} · a_{c,t} + ε_t

The ground-truth β_{c,t} trajectory and baseline are retained in the returned
`SimulationData` so downstream code can evaluate coefficient-recovery, allocation
error, and residual structure against the truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dgp.config import ChannelConfig, DGPConfig
from dgp.transforms import precompute_features


@dataclass(frozen=True)
class SimulationData:
    """Everything downstream code needs: observations + ground truth.

    All per-channel fields are dicts keyed by channel name; all 1-D arrays
    have length `config.n_weeks`.  `features` are the pre-transformed
    a_{c,t} that appear in the observation equation.
    """

    config: DGPConfig
    spend: dict[str, np.ndarray]
    features: dict[str, np.ndarray]
    beta: dict[str, np.ndarray]
    baseline: np.ndarray
    noise: np.ndarray
    y: np.ndarray

    @property
    def channel_names(self) -> list[str]:
        return [c.name for c in self.config.channels]

    def feature_matrix(self) -> np.ndarray:
        """(T, C) matrix of pre-transformed features in channel order."""
        return np.column_stack([self.features[c] for c in self.channel_names])

    def beta_matrix(self) -> np.ndarray:
        """(T, C) matrix of ground-truth β_{c,t} in channel order."""
        return np.column_stack([self.beta[c] for c in self.channel_names])


def simulate_beta_trajectory(
    channel: ChannelConfig,
    n_weeks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random walk with drift.

    β_{c,0} = channel.beta_0 (deterministic).
    β_{c,t} = β_{c,t-1} + drift + η_t,  η_t ~ N(0, innovation_std^2) for t ≥ 1.
    """
    innovations = rng.normal(
        loc=channel.beta_drift,
        scale=channel.beta_innovation_std,
        size=n_weeks,
    )
    innovations[0] = 0.0
    return channel.beta_0 + np.cumsum(innovations)


def simulate_spend(
    channel: ChannelConfig,
    n_weeks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Log-normal weekly spend with E[s] = channel.spend_mean."""
    sigma = channel.spend_log_sigma
    mu = np.log(channel.spend_mean) - 0.5 * sigma**2
    return rng.lognormal(mean=mu, sigma=sigma, size=n_weeks)


def simulate_baseline(config: DGPConfig) -> np.ndarray:
    """Deterministic baseline: intercept + linear trend + annual sinusoid."""
    t = np.arange(config.n_weeks, dtype=float)
    seasonal = config.seasonality_amplitude * np.sin(
        2.0 * np.pi * t / config.seasonality_period
    )
    return config.baseline_mean + config.baseline_trend * t + seasonal


def generate_dataset(config: DGPConfig) -> SimulationData:
    """Run the full DGP pipeline and return a `SimulationData` bundle."""
    rng = np.random.default_rng(config.seed)
    n_weeks = config.n_weeks

    spend: dict[str, np.ndarray] = {}
    features: dict[str, np.ndarray] = {}
    beta: dict[str, np.ndarray] = {}
    for channel in config.channels:
        spend[channel.name] = simulate_spend(channel, n_weeks, rng)
        features[channel.name] = precompute_features(spend[channel.name], channel)
        beta[channel.name] = simulate_beta_trajectory(channel, n_weeks, rng)

    baseline = simulate_baseline(config)
    noise = rng.normal(loc=0.0, scale=config.noise_std, size=n_weeks)

    contribution = np.zeros(n_weeks)
    for channel in config.channels:
        contribution += beta[channel.name] * features[channel.name]
    y = baseline + contribution + noise

    return SimulationData(
        config=config,
        spend=spend,
        features=features,
        beta=beta,
        baseline=baseline,
        noise=noise,
        y=y,
    )
