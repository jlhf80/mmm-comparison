"""Robyn-style MMM: Ridge regression + Nevergrad hyperparameter search.

This is the "wrong and confident" fitter of the three.  It commits to a
single point estimate β̂ (no posterior, no time variation), regularized via
ridge with non-negative coefficients — which is the standard Robyn recipe
once adstock and saturation have been pre-applied.  Nevergrad searches the
ridge penalty on a time-based holdout.

On data where the true β_{c,t} drifts, `beta_at_T` returning a single
time-invariant β̂ is the punchline: Robyn answers "what was average channel
ROI?", not "what is channel ROI right now?".
"""

from __future__ import annotations

import nevergrad as ng
import numpy as np
from sklearn.linear_model import Ridge

from models.base import MMMModel


class RobynModel(MMMModel):
    """Ridge + Nevergrad fitter.

    Parameters
    ----------
    nevergrad_budget:
        Number of objective evaluations Nevergrad is allowed.
    holdout_frac:
        Fraction of the tail of the series held out for hyperparameter
        selection (time-based split, no shuffling).
    log_alpha_bounds:
        (lower, upper) bounds on log ridge penalty searched by Nevergrad.
    seed:
        Seed for Nevergrad's internal RNG.  Ridge itself is deterministic.
    """

    def __init__(
        self,
        *,
        nevergrad_budget: int = 40,
        holdout_frac: float = 0.2,
        log_alpha_bounds: tuple[float, float] = (-4.0, 4.0),
        seed: int | None = 0,
    ) -> None:
        if not 0.0 < holdout_frac < 1.0:
            raise ValueError(f"holdout_frac must be in (0, 1); got {holdout_frac}")
        self.nevergrad_budget = nevergrad_budget
        self.holdout_frac = holdout_frac
        self.log_alpha_bounds = log_alpha_bounds
        self.seed = seed

        self._ridge: Ridge | None = None
        self._best_alpha: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> RobynModel:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError(
                f"expected X (T, C) and y (T,); got {X.shape} and {y.shape}"
            )

        n = X.shape[0]
        split = max(1, int((1.0 - self.holdout_frac) * n))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        def objective(log_alpha: float) -> float:
            alpha = float(np.exp(log_alpha))
            m = Ridge(alpha=alpha, positive=True, fit_intercept=True)
            m.fit(X_train, y_train)
            preds = m.predict(X_val)
            return float(np.mean((y_val - preds) ** 2))

        low, high = self.log_alpha_bounds
        param = ng.p.Scalar(init=0.0).set_bounds(low, high)
        if self.seed is not None:
            param.random_state.seed(self.seed)
        optimizer = ng.optimizers.NGOpt(
            parametrization=param, budget=self.nevergrad_budget
        )
        recommendation = optimizer.minimize(objective)
        self._best_alpha = float(np.exp(recommendation.value))

        self._ridge = Ridge(
            alpha=self._best_alpha, positive=True, fit_intercept=True
        )
        self._ridge.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._ridge is None:
            raise RuntimeError("RobynModel.predict called before fit")
        return self._ridge.predict(np.asarray(X, dtype=float))

    def beta_at_T(self) -> np.ndarray:
        if self._ridge is None:
            raise RuntimeError("RobynModel.beta_at_T called before fit")
        return self._ridge.coef_.copy()

    @property
    def intercept_(self) -> float:
        if self._ridge is None:
            raise RuntimeError("RobynModel.intercept_ read before fit")
        return float(self._ridge.intercept_)

    @property
    def best_alpha(self) -> float:
        if self._best_alpha is None:
            raise RuntimeError("RobynModel.best_alpha read before fit")
        return self._best_alpha
