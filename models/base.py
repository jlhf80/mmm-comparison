"""Abstract base class for the MMM fitters.

Every fitter (Robyn, PyMC, DLM) ingests a (T, C) matrix of pre-transformed
adstock+Hill features and a (T,) response vector, and exposes a common
interface downstream code can rely on: prediction on new features, and a
single (C,) coefficient estimate suitable for forward-looking budget
allocation.

The polymorphism lives in `beta_at_T`: Robyn and PyMC collapse the whole
history into a single fixed β̂_c (broadcast across time), while the DLM
returns the last Kalman-smoothed state β_{c,T}.  The budget optimizer
consumes whichever vector `beta_at_T` returns — that is where the punchline
of this project plays out.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class MMMModel(ABC):
    """Common interface for Robyn / PyMC / DLM fitters.

    Subclasses operate on pre-transformed features: `X` is the (T, C) matrix
    of a_{c,t} = Hill(adstock(spend_c) / scale_c), shared across all models.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> MMMModel:  # noqa: N803
        """Fit the model. Returns self for chaining."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Predict y on a (T', C) feature matrix. Returns shape (T',)."""

    @abstractmethod
    def beta_at_T(self) -> np.ndarray:  # noqa: N802
        """Coefficient estimate for forward-looking budget allocation.

        Shape (C,) in channel order.  For Robyn/PyMC this is a single
        time-invariant β̂_c; for the DLM it is β_{c,T} (last Kalman-smoothed
        state).
        """
