"""PyMC Bayesian MMM: Normal likelihood, HalfNormal β, NUTS posterior.

This is the "wrong but honest" fitter.  The coefficient model is
time-invariant — β_c is a single scalar — but we get a full posterior over
it, which makes the model's disagreement with the time-varying truth
visible via (a) wide/skewed posteriors on β̂ and (b) structured residuals
over time.  Both are diagnostic signals that a sophisticated user can read,
which is the point: PyMC does not pretend to know more than it does.

WAIC requires per-observation log-likelihoods; we compute those via
`pm.compute_log_likelihood` and expose `loglik_matrix` as a (S, T) array
that matches the shape expected by `models.metrics.waic`.
"""

from __future__ import annotations

import numpy as np
import pymc as pm

from models.base import MMMModel


class PyMCModel(MMMModel):
    """Bayesian linear regression on pre-transformed features.

    Priors:
        α ~ Normal(0, intercept_prior_sigma)
        β_c ~ HalfNormal(beta_prior_sigma)   (non-negative — marketing helps)
        σ ~ HalfNormal(sigma_prior_sigma)

    Observation:
        y_t ~ Normal(α + Σ_c β_c · a_{c,t}, σ)
    """

    def __init__(
        self,
        *,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        target_accept: float = 0.9,
        intercept_prior_sigma: float = 10.0,
        beta_prior_sigma: float = 5.0,
        sigma_prior_sigma: float = 2.0,
        seed: int | None = 0,
    ) -> None:
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.intercept_prior_sigma = intercept_prior_sigma
        self.beta_prior_sigma = beta_prior_sigma
        self.sigma_prior_sigma = sigma_prior_sigma
        self.seed = seed

        self._idata = None
        self._beta_posterior: np.ndarray | None = None  # (C, S)
        self._alpha_posterior: np.ndarray | None = None  # (S,)
        self._n_channels: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> PyMCModel:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError(
                f"expected X (T, C) and y (T,); got {X.shape} and {y.shape}"
            )
        n_channels = X.shape[1]

        with pm.Model() as model:
            x_data = pm.Data("x", X)
            alpha = pm.Normal("alpha", mu=0.0, sigma=self.intercept_prior_sigma)
            beta = pm.HalfNormal("beta", sigma=self.beta_prior_sigma, shape=n_channels)
            sigma = pm.HalfNormal("sigma", sigma=self.sigma_prior_sigma)
            mu = alpha + pm.math.dot(x_data, beta)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
            idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                target_accept=self.target_accept,
                random_seed=self.seed,
                progressbar=False,
            )
            pm.compute_log_likelihood(idata, progressbar=False)

        self._model = model
        self._idata = idata
        self._n_channels = n_channels

        beta_stacked = idata.posterior["beta"].stack(sample=("chain", "draw"))
        alpha_stacked = idata.posterior["alpha"].stack(sample=("chain", "draw"))
        self._beta_posterior = beta_stacked.values  # (C, S)
        self._alpha_posterior = alpha_stacked.values  # (S,)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._beta_posterior is None or self._alpha_posterior is None:
            raise RuntimeError("PyMCModel.predict called before fit")
        X = np.asarray(X, dtype=float)
        beta_mean = self._beta_posterior.mean(axis=1)
        alpha_mean = float(self._alpha_posterior.mean())
        return alpha_mean + X @ beta_mean

    def beta_at_T(self) -> np.ndarray:
        if self._beta_posterior is None:
            raise RuntimeError("PyMCModel.beta_at_T called before fit")
        return self._beta_posterior.mean(axis=1)

    def loglik_matrix(self) -> np.ndarray:
        """(S, T) per-observation log-likelihood, matching `metrics.waic`."""
        if self._idata is None:
            raise RuntimeError("PyMCModel.loglik_matrix called before fit")
        ll = self._idata.log_likelihood["y_obs"].stack(sample=("chain", "draw"))
        # xarray stacks the new `sample` dim last; transpose to (sample, obs).
        return ll.transpose("sample", ...).values

    @property
    def idata(self):
        if self._idata is None:
            raise RuntimeError("PyMCModel.idata read before fit")
        return self._idata
