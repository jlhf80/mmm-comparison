"""Forward-looking budget allocation via SLSQP.

Given a fitted coefficient vector β̂ (from any model's `beta_at_T`) and the
channel transform parameters, solve

    max_{s_1,...,s_C}  Σ_c β̂_c · Hill(s_c / scale_c; α_c, γ_c)
    s.t.               Σ_c s_c = total_budget,  s_c ≥ 0.

Convention: we evaluate at a single forward time step with no adstock
history — the decision-relevant response is the instantaneous Hill map of
a one-week spend.  Adstock is what makes the *time-series* problem hard;
the budget-allocation problem is separable in c once β̂ is fixed.

By construction the transform parameters are the true DGP's (per the
"Option A — pre-transformed features" design), so this function imports
from `dgp.config` / `dgp.transforms`.  When we compare allocations across
fitters, only β̂ differs — everything else is held fixed.  That is exactly
what isolates the coefficient-dynamics question.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize

from dgp.config import ChannelConfig
from dgp.transforms import hill_saturation


def _channel_response(spend: float, channel: ChannelConfig) -> float:
    """Instantaneous Hill response at a single spend value (no adstock)."""
    x = np.array([spend / channel.hill_scale])
    return float(
        hill_saturation(x, alpha=channel.hill_alpha, gamma=channel.hill_gamma)[0]
    )


def expected_revenue(
    spend: np.ndarray,
    beta_hat: np.ndarray,
    channels: Sequence[ChannelConfig],
) -> float:
    """Σ_c β̂_c · Hill(s_c / scale_c).  Exposed so tests can check KKT."""
    spend = np.asarray(spend, dtype=float)
    beta_hat = np.asarray(beta_hat, dtype=float)
    return float(
        sum(
            beta_hat[i] * _channel_response(float(spend[i]), ch)
            for i, ch in enumerate(channels)
        )
    )


def allocate_budget(
    beta_hat: np.ndarray,
    channels: Sequence[ChannelConfig],
    total_budget: float,
) -> dict[str, float]:
    """Solve the SLSQP budget-allocation problem.

    `beta_hat` must be in the same order as `channels`.  Returns a dict
    mapping channel name → allocated spend.  Sums to `total_budget` up to
    solver tolerance.
    """
    beta_hat = np.asarray(beta_hat, dtype=float)
    n = len(channels)
    if beta_hat.shape != (n,):
        raise ValueError(f"beta_hat shape {beta_hat.shape} != ({n},)")
    if total_budget <= 0:
        raise ValueError(f"total_budget must be > 0; got {total_budget}")

    def neg_revenue(s: np.ndarray) -> float:
        return -expected_revenue(s, beta_hat, channels)

    x0 = np.full(n, total_budget / n)
    constraints = [{"type": "eq", "fun": lambda s: float(np.sum(s) - total_budget)}]
    bounds = [(0.0, total_budget) for _ in range(n)]

    result = minimize(
        neg_revenue,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"allocate_budget failed: {result.message}")

    return {ch.name: float(result.x[i]) for i, ch in enumerate(channels)}
