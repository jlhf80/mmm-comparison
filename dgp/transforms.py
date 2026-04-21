"""Media transforms: geometric adstock and Hill saturation.

These are the shared transforms used both to *generate* the DGP features and to
construct the pre-transformed inputs handed to each MMM model.  Keeping a
single implementation guarantees that the DGP and the fitters agree on the
functional form; the isolation we want is over coefficient dynamics, not over
the nonlinearities.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from dgp.config import ChannelConfig


def geometric_adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """Geometric adstock: a*_t = s_t + decay · a*_{t-1}, with a*_{-1} = 0.

    Implemented as an IIR filter via `scipy.signal.lfilter` with transfer
    function 1 / (1 - decay·z^{-1}), which is the exact recursion above.
    """
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must be in [0, 1); got {decay}")
    spend = np.asarray(spend, dtype=float)
    if spend.ndim != 1:
        raise ValueError(f"spend must be 1-D; got shape {spend.shape}")
    return lfilter(b=[1.0], a=[1.0, -decay], x=spend)


def hill_saturation(x: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """Hill function: x^alpha / (x^alpha + gamma^alpha).

    Returns values in [0, 1).  `x` must be non-negative; `alpha, gamma > 0`.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0; got {alpha}")
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0; got {gamma}")
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        raise ValueError("hill_saturation input must be non-negative")
    x_a = np.power(x, alpha)
    g_a = gamma**alpha
    return x_a / (x_a + g_a)


def precompute_features(spend: np.ndarray, channel: ChannelConfig) -> np.ndarray:
    """Apply adstock then (scale-normalized) Hill saturation to raw weekly spend.

    This returns the `a_{c,t}` values that appear in the observation equation
    y_t = α_t + Σ_c β_{c,t} · a_{c,t} + ε_t.  Both the DGP and the fitted
    models consume these features directly — by construction, the λ/Hill
    parameters are shared with the true DGP.
    """
    adstocked = geometric_adstock(spend, channel.adstock_decay)
    normalized = adstocked / channel.hill_scale
    return hill_saturation(normalized, channel.hill_alpha, channel.hill_gamma)
