"""Dynamic Linear Model with a hand-rolled Kalman filter + RTS smoother.

This is the "correctly specified" fitter of the three.  The state vector
x_t = (α_t, β_{1,t}, ..., β_{C,t}) follows a random walk with drift, and
the observation h_t^T x_t is exactly the DGP's observation equation with
the intercept absorbed into the first state.

The filter and smoother are written explicitly rather than wrapped around a
library.  That is intentional — the whole project hinges on the claim that
the DLM recovers the true β_{c,T} when the other fitters can only recover
an average.  Showing the mechanism (innovations, Kalman gain, Joseph-form
covariance update, RTS backward recursion) makes that claim auditable.

State-space model:
    x_t = x_{t-1} + u + w_t,  w_t ~ N(0, Q)           (state: (C+1,))
    y_t = h_t^T x_t + ε_t,    ε_t ~ N(0, R),  h_t = (1, a_{1,t}, ..., a_{C,t})
"""

from __future__ import annotations

import numpy as np

from models.base import MMMModel


def kalman_filter(
    y: np.ndarray,
    H: np.ndarray,
    u: np.ndarray,
    Q: np.ndarray,
    R: float,
    x0: np.ndarray,
    P0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Run the forward Kalman filter on a linear-Gaussian state-space model.

    Returns:
        x_filt: (T, D) filtered means  x_{t|t}
        P_filt: (T, D, D) filtered covariances  P_{t|t}
        x_pred: (T, D) predicted means  x_{t|t-1}
        P_pred: (T, D, D) predicted covariances  P_{t|t-1}
        loglik: scalar marginal log-likelihood Σ_t log p(y_t | y_{1:t-1})

    Joseph-form covariance update is used for numerical stability.
    """
    y = np.asarray(y, dtype=float)
    H = np.asarray(H, dtype=float)
    n_obs, D = H.shape
    if y.shape != (n_obs,):
        raise ValueError(f"y shape {y.shape} inconsistent with H rows {n_obs}")

    x_filt = np.zeros((n_obs, D))
    P_filt = np.zeros((n_obs, D, D))
    x_pred = np.zeros((n_obs, D))
    P_pred = np.zeros((n_obs, D, D))

    x_prev = x0.astype(float).copy()
    P_prev = P0.astype(float).copy()
    loglik = 0.0
    eye_d = np.eye(D)

    for t in range(n_obs):
        # --- predict ---
        x_p = x_prev + u
        P_p = P_prev + Q
        x_pred[t] = x_p
        P_pred[t] = P_p

        # --- update ---
        h = H[t]
        s = float(h @ P_p @ h + R)  # innovation variance (scalar — y is 1-D)
        k = (P_p @ h) / s            # Kalman gain (D,)
        v = float(y[t] - h @ x_p)    # innovation
        x_f = x_p + k * v

        # Joseph form: P_f = (I - kh^T) P_p (I - kh^T)^T + k R k^T
        ikh = eye_d - np.outer(k, h)
        P_f = ikh @ P_p @ ikh.T + np.outer(k, k) * R

        x_filt[t] = x_f
        P_filt[t] = P_f
        loglik += -0.5 * (np.log(2.0 * np.pi * s) + v * v / s)

        x_prev = x_f
        P_prev = P_f

    return x_filt, P_filt, x_pred, P_pred, float(loglik)


def rts_smoother(
    x_filt: np.ndarray,
    P_filt: np.ndarray,
    x_pred: np.ndarray,
    P_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel backward recursion.

    Consumes the filter's predicted and filtered states and returns the
    smoothed means and covariances x_{t|T}, P_{t|T}.

    At t = T-1 the smoother identities collapse to the filter — this is a
    correctness invariant the tests assert.
    """
    n_obs, D = x_filt.shape
    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()

    for t in range(n_obs - 2, -1, -1):
        # J_t = P_{t|t} P_{t+1|t}^{-1}  (smoother gain)
        gain = np.linalg.solve(P_pred[t + 1].T, P_filt[t].T).T
        x_smooth[t] = x_filt[t] + gain @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = (
            P_filt[t] + gain @ (P_smooth[t + 1] - P_pred[t + 1]) @ gain.T
        )
        # Keep symmetric despite floating-point drift.
        P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)

    return x_smooth, P_smooth


class DLMModel(MMMModel):
    """Time-varying-coefficient DLM fit with a Kalman filter + RTS smoother.

    The state x_t has length C+1 and stacks (α_t, β_{1,t}, ..., β_{C,t});
    the observation row h_t is (1, a_{1,t}, ..., a_{C,t}).  The feature
    matrix passed to `fit` is the same (T, C) pre-transformed a_{c,t}
    matrix every other fitter consumes — the DLM does not privilege
    knowledge of the DGP beyond the transform parameters everyone shares.

    Parameters
    ----------
    drift:
        (C+1,) vector u in the state equation.  Defaults to zeros (pure
        random walk).  Reflects the analyst's prior that coefficients drift
        in a known direction.
    innovation_var:
        Diagonal of Q (per-state innovation variance).  Scalar → isotropic;
        (C+1,) → diagonal Q with those variances.
    observation_var:
        Scalar R in the observation equation.
    initial_var:
        Diagonal of P_0 (diffuse prior).  A large value encodes "don't know
        the initial state".
    initial_mean:
        (C+1,) prior mean.  Defaults to zeros.
    """

    def __init__(
        self,
        *,
        drift: np.ndarray | None = None,
        innovation_var: float | np.ndarray = 1e-3,
        observation_var: float = 1.0,
        initial_var: float = 10.0,
        initial_mean: np.ndarray | None = None,
    ) -> None:
        self.drift = drift
        self.innovation_var = innovation_var
        self.observation_var = observation_var
        self.initial_var = initial_var
        self.initial_mean = initial_mean

        self._H_train: np.ndarray | None = None
        self._x_filt: np.ndarray | None = None
        self._P_filt: np.ndarray | None = None
        self._x_smooth: np.ndarray | None = None
        self._P_smooth: np.ndarray | None = None
        self._loglik: float | None = None
        self._state_dim: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> DLMModel:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError(
                f"expected X (T, C) and y (T,); got {X.shape} and {y.shape}"
            )

        n_obs, n_channels = X.shape
        D = n_channels + 1  # state dim: intercept + per-channel coefficient
        H = np.hstack([np.ones((n_obs, 1)), X])

        u = self._resolve_drift(D)
        Q = self._resolve_Q(D)
        R = float(self.observation_var)
        x0 = self._resolve_initial_mean(D)
        P0 = np.eye(D) * float(self.initial_var)

        x_filt, P_filt, x_pred, P_pred, ll = kalman_filter(y, H, u, Q, R, x0, P0)
        x_smooth, P_smooth = rts_smoother(x_filt, P_filt, x_pred, P_pred)

        self._H_train = H
        self._x_filt = x_filt
        self._P_filt = P_filt
        self._x_smooth = x_smooth
        self._P_smooth = P_smooth
        self._loglik = ll
        self._state_dim = D
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forecast y on new features using the last smoothed state x_{T|T}.

        This is the forward-looking prediction — same β that `beta_at_T`
        returns, broadcast across rows of X.  For in-sample reconstruction
        that uses the full smoothed trajectory, call `fitted_values()`.
        """
        if self._x_smooth is None or self._state_dim is None:
            raise RuntimeError("DLMModel.predict called before fit")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] + 1 != self._state_dim:
            raise ValueError(
                f"X has shape {X.shape}; expected (T, {self._state_dim - 1})"
            )
        H = np.hstack([np.ones((X.shape[0], 1)), X])
        return H @ self._x_smooth[-1]

    def fitted_values(self) -> np.ndarray:
        """In-sample ŷ_t using the smoothed state x_{t|T} at each time step."""
        if self._H_train is None or self._x_smooth is None:
            raise RuntimeError("DLMModel.fitted_values called before fit")
        return np.einsum("td,td->t", self._H_train, self._x_smooth)

    def beta_at_T(self) -> np.ndarray:
        """Last smoothed β_{c,T} (excludes the intercept state)."""
        if self._x_smooth is None:
            raise RuntimeError("DLMModel.beta_at_T called before fit")
        return self._x_smooth[-1, 1:].copy()

    @property
    def smoothed_states(self) -> np.ndarray:
        """(T, C+1) smoothed state means."""
        if self._x_smooth is None:
            raise RuntimeError("DLMModel.smoothed_states read before fit")
        return self._x_smooth.copy()

    @property
    def smoothed_covariances(self) -> np.ndarray:
        """(T, C+1, C+1) smoothed state covariances."""
        if self._P_smooth is None:
            raise RuntimeError("DLMModel.smoothed_covariances read before fit")
        return self._P_smooth.copy()

    @property
    def filtered_states(self) -> np.ndarray:
        if self._x_filt is None:
            raise RuntimeError("DLMModel.filtered_states read before fit")
        return self._x_filt.copy()

    @property
    def loglik(self) -> float:
        if self._loglik is None:
            raise RuntimeError("DLMModel.loglik read before fit")
        return self._loglik

    # --- private helpers --------------------------------------------------

    def _resolve_drift(self, D: int) -> np.ndarray:
        if self.drift is None:
            return np.zeros(D)
        u = np.asarray(self.drift, dtype=float)
        if u.shape != (D,):
            raise ValueError(f"drift shape {u.shape} != ({D},)")
        return u

    def _resolve_Q(self, D: int) -> np.ndarray:
        q = self.innovation_var
        if np.ndim(q) == 0:
            return np.eye(D) * float(q)
        q = np.asarray(q, dtype=float)
        if q.shape == (D,):
            return np.diag(q)
        if q.shape == (D, D):
            return q
        raise ValueError(f"innovation_var shape {q.shape} invalid for D={D}")

    def _resolve_initial_mean(self, D: int) -> np.ndarray:
        if self.initial_mean is None:
            return np.zeros(D)
        x0 = np.asarray(self.initial_mean, dtype=float)
        if x0.shape != (D,):
            raise ValueError(f"initial_mean shape {x0.shape} != ({D},)")
        return x0
