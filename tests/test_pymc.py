"""Tests for the PyMC Bayesian fitter.

Runs NUTS on a small problem (T=50, C=2, 1 chain, 200 draws) to keep CI
under ~20s.  Recovery tolerances are intentionally loose — this is a
contract test, not a convergence test.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.metrics import waic
from models.pymc_model import PyMCModel
from tests._fixtures import stationary_dataset

_FAST_KWARGS = dict(draws=200, tune=200, chains=1, seed=0)


@pytest.fixture(scope="module")
def fitted_pymc() -> tuple[PyMCModel, np.ndarray, np.ndarray, tuple[str, ...]]:
    data = stationary_dataset(n_weeks=50, noise_std=0.2, seed=0)
    X = data.feature_matrix()
    model = PyMCModel(**_FAST_KWARGS).fit(X, data.y)
    return model, X, data.y, tuple(data.channel_names)


def test_pymc_beta_at_T_shape(fitted_pymc) -> None:
    model, _, _, names = fitted_pymc
    assert model.beta_at_T().shape == (len(names),)


def test_pymc_beta_is_non_negative(fitted_pymc) -> None:
    """HalfNormal prior ⇒ posterior support is R_+; posterior mean is ≥ 0."""
    model, _, _, _ = fitted_pymc
    assert (model.beta_at_T() >= 0).all()


def test_pymc_predict_shape(fitted_pymc) -> None:
    model, X, y, _ = fitted_pymc
    preds = model.predict(X)
    assert preds.shape == y.shape


def test_pymc_loglik_matrix_shape(fitted_pymc) -> None:
    """loglik_matrix is (S, T) — ready for `metrics.waic`."""
    model, _, y, _ = fitted_pymc
    ll = model.loglik_matrix()
    assert ll.ndim == 2
    assert ll.shape[1] == y.shape[0]
    expected_samples = _FAST_KWARGS["draws"] * _FAST_KWARGS["chains"]
    assert ll.shape[0] == expected_samples


def test_pymc_waic_is_finite(fitted_pymc) -> None:
    model, _, _, _ = fitted_pymc
    score = waic(model.loglik_matrix())
    assert np.isfinite(score)


def test_pymc_recovers_beta_ranking_on_stationary_data(fitted_pymc) -> None:
    """Posterior mean must preserve true β_a = 2.0 > β_b = 1.0 ordering."""
    model, _, _, _ = fitted_pymc
    beta = model.beta_at_T()
    assert beta[0] > beta[1]


def test_pymc_predict_before_fit_raises() -> None:
    model = PyMCModel()
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(np.zeros((5, 2)))


def test_pymc_loglik_matrix_before_fit_raises() -> None:
    model = PyMCModel()
    with pytest.raises(RuntimeError, match="before fit"):
        model.loglik_matrix()


def test_pymc_beta_posterior_samples_shape(fitted_pymc) -> None:
    """beta_posterior_samples returns (C, S) with S = draws × chains."""
    model, _, _, names = fitted_pymc
    draws = model.beta_posterior_samples()
    expected_samples = _FAST_KWARGS["draws"] * _FAST_KWARGS["chains"]
    assert draws.shape == (len(names), expected_samples)
    # Mean across samples matches beta_at_T().
    np.testing.assert_allclose(draws.mean(axis=1), model.beta_at_T())


def test_pymc_beta_posterior_samples_before_fit_raises() -> None:
    model = PyMCModel()
    with pytest.raises(RuntimeError, match="before fit"):
        model.beta_posterior_samples()


def test_pymc_rejects_shape_mismatch() -> None:
    model = PyMCModel()
    with pytest.raises(ValueError, match="expected X"):
        model.fit(np.zeros((10, 2)), np.zeros(9))
