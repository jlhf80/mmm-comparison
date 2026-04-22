"""Tests for the Robyn-style Ridge + Nevergrad fitter."""

from __future__ import annotations

import numpy as np
import pytest

from models.robyn_model import RobynModel
from tests._fixtures import stationary_dataset


def test_robyn_fit_returns_self() -> None:
    data = stationary_dataset()
    model = RobynModel(nevergrad_budget=10, seed=0)
    out = model.fit(data.feature_matrix(), data.y)
    assert out is model


def test_robyn_beta_at_T_shape_matches_num_channels() -> None:
    data = stationary_dataset()
    model = RobynModel(nevergrad_budget=10, seed=0).fit(data.feature_matrix(), data.y)
    beta = model.beta_at_T()
    assert beta.shape == (len(data.channel_names),)


def test_robyn_coefficients_are_non_negative() -> None:
    """positive=True constraint — marketing coefficients cannot be negative."""
    data = stationary_dataset()
    model = RobynModel(nevergrad_budget=10, seed=0).fit(data.feature_matrix(), data.y)
    assert (model.beta_at_T() >= 0).all()


def test_robyn_predict_shape() -> None:
    data = stationary_dataset()
    model = RobynModel(nevergrad_budget=10, seed=0).fit(data.feature_matrix(), data.y)
    preds = model.predict(data.feature_matrix())
    assert preds.shape == data.y.shape


def test_robyn_recovers_beta_ranking_on_stationary_data() -> None:
    """True β_a = 2.0 > β_b = 1.0 — Robyn should recover the ordering."""
    data = stationary_dataset(n_weeks=120, noise_std=0.1, seed=1)
    model = RobynModel(nevergrad_budget=20, seed=0).fit(data.feature_matrix(), data.y)
    beta = model.beta_at_T()
    assert beta[0] > beta[1]


def test_robyn_predict_before_fit_raises() -> None:
    model = RobynModel()
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(np.zeros((5, 2)))


def test_robyn_beta_at_T_before_fit_raises() -> None:
    model = RobynModel()
    with pytest.raises(RuntimeError, match="before fit"):
        model.beta_at_T()


def test_robyn_rejects_shape_mismatch() -> None:
    model = RobynModel()
    with pytest.raises(ValueError, match="expected X"):
        model.fit(np.zeros((10, 2)), np.zeros(9))


def test_robyn_rejects_bad_holdout_frac() -> None:
    with pytest.raises(ValueError, match="holdout_frac"):
        RobynModel(holdout_frac=0.0)
