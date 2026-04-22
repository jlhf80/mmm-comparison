"""Dynamic Linear Model with a hand-rolled Kalman filter + RTS smoother.

This is the "correctly specified" fitter of the three.  In its simplest form
the state vector is x_t = (α_t, β_{1,t}, ..., β_{C,t}), a random walk with
drift, and h_t^T x_t is the DGP's observation equation with the intercept
absorbed.

For a DGP whose baseline carries a linear trend and annual seasonality,
that plain random walk is misspecified — the true α_t cannot be represented
by a single near-frozen state, so its variance leaks into β_c and corrupts
the coefficient-recovery story.  The structural-time-series extension
(Harvey 1989 / West & Harrison 1997) fixes this by giving the model
dedicated state blocks for the level, trend, and seasonal components, so
the β-block absorbs only what it should — the regression signal.

State-space model (general form):
    x_t = F x_{t-1} + u + w_t,   w_t ~ N(0, Q)
    y_t = h_t^T x_t + ε_t,        ε_t ~ N(0, R)

The plain random-walk DLM is the special case F = I, u = drift; the
structural DLM uses a block-diagonal F that composes a local linear trend,
a Fourier seasonal oscillator, and a random-walk β block.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag

from models.base import MMMModel


def kalman_filter(
    y: np.ndarray,
    H: np.ndarray,
    u: np.ndarray,
    Q: np.ndarray,
    R: float,
    x0: np.ndarray,
    P0: np.ndarray,
    F: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Run the forward Kalman filter on a linear-Gaussian state-space model.

    Returns:
        x_filt: (T, D) filtered means  x_{t|t}
        P_filt: (T, D, D) filtered covariances  P_{t|t}
        x_pred: (T, D) predicted means  x_{t|t-1}
        P_pred: (T, D, D) predicted covariances  P_{t|t-1}
        loglik: scalar marginal log-likelihood Σ_t log p(y_t | y_{1:t-1})

    `F` is the state transition matrix; None selects the identity (pure
    random walk with drift).  Joseph-form covariance update is used for
    numerical stability.
    """
    y = np.asarray(y, dtype=float)
    H = np.asarray(H, dtype=float)
    n_obs, D = H.shape
    if y.shape != (n_obs,):
        raise ValueError(f"y shape {y.shape} inconsistent with H rows {n_obs}")

    F_arr = np.eye(D) if F is None else np.asarray(F, dtype=float)
    if F_arr.shape != (D, D):
        raise ValueError(f"F shape {F_arr.shape} != ({D}, {D})")

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
        x_p = F_arr @ x_prev + u
        P_p = F_arr @ P_prev @ F_arr.T + Q
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
    F: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel backward recursion.

    Consumes the filter's predicted and filtered states and returns the
    smoothed means and covariances x_{t|T}, P_{t|T}.  At t = T-1 the
    identities collapse to the filter — a correctness invariant the tests
    assert.

    The smoother gain is J_t = P_{t|t} F^T P_{t+1|t}^{-1}.  F = None is
    treated as the identity for backward compatibility with the original
    random-walk filter.
    """
    n_obs, D = x_filt.shape
    F_arr = np.eye(D) if F is None else np.asarray(F, dtype=float)

    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()

    for t in range(n_obs - 2, -1, -1):
        # J_t = P_{t|t} F^T P_{t+1|t}^{-1}.  Solve in transposed form for
        # numerical stability: J^T = P_{t+1|t}^{-1} F P_{t|t}^T.
        gain = np.linalg.solve(P_pred[t + 1], F_arr @ P_filt[t].T).T
        x_smooth[t] = x_filt[t] + gain @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = (
            P_filt[t] + gain @ (P_smooth[t + 1] - P_pred[t + 1]) @ gain.T
        )
        # Keep symmetric despite floating-point drift.
        P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)

    return x_smooth, P_smooth


# --- structural assembly --------------------------------------------------

def _seasonal_block(period: float, n_harmonics: int) -> np.ndarray:
    """Block-diagonal rotation matrix stacking `n_harmonics` Fourier pairs.

    Harmonic h has angular frequency ω_h = 2π h / period; its 2×2 rotation
    block is [[cos ω_h, sin ω_h], [-sin ω_h, cos ω_h]].  Only the first
    state of each pair is observed (loading 1); the second is a latent
    quadrature companion keeping the oscillator spinning.
    """
    blocks = []
    for h in range(1, n_harmonics + 1):
        omega = 2.0 * np.pi * h / period
        c, s = np.cos(omega), np.sin(omega)
        blocks.append(np.array([[c, s], [-s, c]]))
    return block_diag(*blocks)


def build_structural_state_space(
    n_channels: int,
    *,
    local_linear_trend: bool,
    seasonal_period: float | None,
    seasonal_harmonics: int,
    level_innovation_var: float,
    slope_innovation_var: float,
    seasonal_innovation_var: float,
    beta_innovation_var: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, slice], int]:
    """Assemble F, Q, and component index slices for a structural DLM.

    The returned `slices` dict maps component names ('level', 'slope',
    'seasonal', 'beta') to half-open state-index slices.  Callers use
    these to (a) build the observation matrix H, (b) read trajectories
    back out of the smoothed states, and (c) set component-specific
    initial means / variances.
    """
    if seasonal_period is not None and seasonal_period <= 0.0:
        raise ValueError("seasonal_period must be positive or None")
    if seasonal_harmonics < 1:
        raise ValueError("seasonal_harmonics must be >= 1")

    f_blocks: list[np.ndarray] = []
    q_blocks: list[np.ndarray] = []
    slices: dict[str, slice] = {}
    offset = 0

    # Level (+ optional slope) block.
    if local_linear_trend:
        f_blocks.append(np.array([[1.0, 1.0], [0.0, 1.0]]))
        q_blocks.append(np.diag([level_innovation_var, slope_innovation_var]))
        slices["level"] = slice(offset, offset + 1)
        slices["slope"] = slice(offset + 1, offset + 2)
        offset += 2
    else:
        f_blocks.append(np.array([[1.0]]))
        q_blocks.append(np.array([[level_innovation_var]]))
        slices["level"] = slice(offset, offset + 1)
        offset += 1

    # Seasonal block.
    if seasonal_period is not None:
        f_seas = _seasonal_block(seasonal_period, seasonal_harmonics)
        seas_dim = f_seas.shape[0]
        f_blocks.append(f_seas)
        q_blocks.append(np.eye(seas_dim) * seasonal_innovation_var)
        slices["seasonal"] = slice(offset, offset + seas_dim)
        offset += seas_dim

    # Beta block.
    if np.ndim(beta_innovation_var) == 0:
        q_beta = np.eye(n_channels) * float(beta_innovation_var)
    else:
        q_beta_arr = np.asarray(beta_innovation_var, dtype=float)
        if q_beta_arr.shape != (n_channels,):
            raise ValueError(
                f"beta_innovation_var shape {q_beta_arr.shape} "
                f"!= ({n_channels},)"
            )
        q_beta = np.diag(q_beta_arr)
    f_blocks.append(np.eye(n_channels))
    q_blocks.append(q_beta)
    slices["beta"] = slice(offset, offset + n_channels)
    offset += n_channels

    F = block_diag(*f_blocks)
    Q = block_diag(*q_blocks)
    return F, Q, slices, offset


def build_observation_matrix(
    X: np.ndarray, slices: dict[str, slice], state_dim: int
) -> np.ndarray:
    """Construct (T, D) observation matrix H from the feature matrix X.

    Rows are h_t = (level=1, slope=0, {seasonal pair: 1, 0 per harmonic},
    β block = a_{c,t}).  Only the *first* state of each seasonal pair is
    observed — the second is a hidden quadrature partner.
    """
    n_obs = X.shape[0]
    H = np.zeros((n_obs, state_dim))
    H[:, slices["level"].start] = 1.0
    if "seasonal" in slices:
        seas = slices["seasonal"]
        n_harm = (seas.stop - seas.start) // 2
        for h in range(n_harm):
            H[:, seas.start + 2 * h] = 1.0
    H[:, slices["beta"]] = X
    return H


class DLMModel(MMMModel):
    """Time-varying-coefficient DLM fit with a Kalman filter + RTS smoother.

    Two modes:

    **Plain random-walk (default)** — state is (α_t, β_{1,t}, ..., β_{C,t})
    evolving as x_t = x_{t-1} + u + w.  Controlled by `drift`,
    `innovation_var`, `observation_var`, `initial_var`, `initial_mean`.

    **Structural** — activated by setting any of `local_linear_trend`,
    `seasonal_period`, or the `*_innovation_var` component overrides.  The
    state is augmented with a local linear trend and/or a Fourier seasonal
    basis before the β block, with F a block-diagonal composition of
    rotations + random walks.  Use this when the baseline α_t carries
    structured dynamics (trend, annual seasonality) that a frozen intercept
    cannot absorb.

    In both modes the β block is a C-dimensional random walk and
    `beta_at_T` returns its last smoothed state.

    Parameters
    ----------
    drift:
        Random-walk drift u (plain mode only).  Zeros by default.
    innovation_var:
        Q (plain mode only).  Scalar → isotropic; (C+1,) → diagonal.
    observation_var:
        Scalar R in the observation equation (both modes).
    initial_var:
        Diagonal of P_0.  Scalar is broadcast to all states (diffuse prior).
    initial_mean:
        Prior mean of x_0.  Defaults to zeros of the right length.
    local_linear_trend:
        If True, replaces the random-walk intercept with a (level, slope)
        pair whose dynamics are level_t = level_{t-1} + slope_{t-1} and
        slope_t = slope_{t-1} + w.  Captures smooth non-stationary drift.
    seasonal_period:
        Period of the Fourier seasonal basis (e.g. 52 for annual seasonality
        on weekly data).  None disables the seasonal block.
    seasonal_harmonics:
        Number of harmonics to include when the seasonal block is active.
        1 gives a pure sinusoid; 2-3 captures sharper shapes at a modest
        cost in extra state dimensions.
    level_innovation_var, slope_innovation_var, seasonal_innovation_var:
        Component innovation variances for structural mode.  Defaults are
        small positive numbers so the filter is robust without being sloppy.
    beta_innovation_var:
        Innovation variance on the β block (structural mode).  Scalar →
        isotropic across channels; (C,) → per-channel.
    """

    def __init__(
        self,
        *,
        drift: np.ndarray | None = None,
        innovation_var: float | np.ndarray = 1e-3,
        observation_var: float = 1.0,
        initial_var: float = 10.0,
        initial_mean: np.ndarray | None = None,
        local_linear_trend: bool = False,
        seasonal_period: float | None = None,
        seasonal_harmonics: int = 1,
        level_innovation_var: float = 1e-6,
        slope_innovation_var: float = 1e-6,
        seasonal_innovation_var: float = 0.0,
        beta_innovation_var: float | np.ndarray = 5e-3,
    ) -> None:
        self.drift = drift
        self.innovation_var = innovation_var
        self.observation_var = observation_var
        self.initial_var = initial_var
        self.initial_mean = initial_mean

        self.local_linear_trend = local_linear_trend
        self.seasonal_period = seasonal_period
        self.seasonal_harmonics = seasonal_harmonics
        self.level_innovation_var = level_innovation_var
        self.slope_innovation_var = slope_innovation_var
        self.seasonal_innovation_var = seasonal_innovation_var
        self.beta_innovation_var = beta_innovation_var

        self._H_train: np.ndarray | None = None
        self._F: np.ndarray | None = None
        self._x_filt: np.ndarray | None = None
        self._P_filt: np.ndarray | None = None
        self._x_smooth: np.ndarray | None = None
        self._P_smooth: np.ndarray | None = None
        self._loglik: float | None = None
        self._state_dim: int | None = None
        self._slices: dict[str, slice] | None = None

    @property
    def is_structural(self) -> bool:
        return self.local_linear_trend or self.seasonal_period is not None

    def fit(self, X: np.ndarray, y: np.ndarray) -> DLMModel:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError(
                f"expected X (T, C) and y (T,); got {X.shape} and {y.shape}"
            )

        n_obs, n_channels = X.shape

        if self.is_structural:
            F, Q, slices, D = build_structural_state_space(
                n_channels,
                local_linear_trend=self.local_linear_trend,
                seasonal_period=self.seasonal_period,
                seasonal_harmonics=self.seasonal_harmonics,
                level_innovation_var=self.level_innovation_var,
                slope_innovation_var=self.slope_innovation_var,
                seasonal_innovation_var=self.seasonal_innovation_var,
                beta_innovation_var=self.beta_innovation_var,
            )
            H = build_observation_matrix(X, slices, D)
            u = np.zeros(D)
        else:
            D = n_channels + 1
            F = None  # identity; kalman_filter resolves
            Q = self._resolve_Q(D)
            slices = {
                "level": slice(0, 1),
                "beta": slice(1, D),
            }
            H = np.hstack([np.ones((n_obs, 1)), X])
            u = self._resolve_drift(D)

        R = float(self.observation_var)
        x0 = self._resolve_initial_mean(D)
        P0 = np.eye(D) * float(self.initial_var)

        x_filt, P_filt, x_pred, P_pred, ll = kalman_filter(
            y, H, u, Q, R, x0, P0, F=F
        )
        x_smooth, P_smooth = rts_smoother(x_filt, P_filt, x_pred, P_pred, F=F)

        self._H_train = H
        self._F = np.eye(D) if F is None else F
        self._x_filt = x_filt
        self._P_filt = P_filt
        self._x_smooth = x_smooth
        self._P_smooth = P_smooth
        self._loglik = ll
        self._state_dim = D
        self._slices = slices
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forecast y on new features using the last smoothed state x_{T|T}.

        Structural components propagate through their transition (so the
        seasonal state keeps spinning into the future, the slope keeps
        accumulating level, etc.).  β states are held fixed at β_T — this
        is the forward-looking estimate that drives the budget optimizer.
        """
        if self._x_smooth is None or self._state_dim is None:
            raise RuntimeError("DLMModel.predict called before fit")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self._n_channels():
            raise ValueError(
                f"X has shape {X.shape}; expected (T, {self._n_channels()})"
            )

        slices = self._slices
        assert slices is not None
        H_new = build_observation_matrix(X, slices, self._state_dim) \
            if self.is_structural \
            else np.hstack([np.ones((X.shape[0], 1)), X])

        x = self._x_smooth[-1].copy()
        F = self._F
        assert F is not None
        preds = np.empty(X.shape[0])
        for t in range(X.shape[0]):
            x = F @ x  # propagate structural dynamics forward
            preds[t] = H_new[t] @ x
        return preds

    def fitted_values(self) -> np.ndarray:
        """In-sample ŷ_t using the smoothed state x_{t|T} at each time step."""
        if self._H_train is None or self._x_smooth is None:
            raise RuntimeError("DLMModel.fitted_values called before fit")
        return np.einsum("td,td->t", self._H_train, self._x_smooth)

    def beta_at_T(self) -> np.ndarray:
        """Last smoothed β block (excludes intercept / structural states)."""
        if self._x_smooth is None or self._slices is None:
            raise RuntimeError("DLMModel.beta_at_T called before fit")
        return self._x_smooth[-1, self._slices["beta"]].copy()

    # --- introspection helpers ----------------------------------------

    @property
    def smoothed_states(self) -> np.ndarray:
        """(T, D) smoothed state means."""
        if self._x_smooth is None:
            raise RuntimeError("DLMModel.smoothed_states read before fit")
        return self._x_smooth.copy()

    @property
    def smoothed_covariances(self) -> np.ndarray:
        """(T, D, D) smoothed state covariances."""
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

    @property
    def state_slices(self) -> dict[str, slice]:
        """Maps component name ('level', 'slope', 'seasonal', 'beta') to slice."""
        if self._slices is None:
            raise RuntimeError("DLMModel.state_slices read before fit")
        return dict(self._slices)

    def component_trajectory(self, name: str) -> np.ndarray:
        """Smoothed trajectory of the named component — e.g. 'level', 'beta'."""
        if self._x_smooth is None or self._slices is None:
            raise RuntimeError("DLMModel.component_trajectory called before fit")
        if name not in self._slices:
            raise KeyError(f"component {name!r} not in state {list(self._slices)}")
        return self._x_smooth[:, self._slices[name]].copy()

    # --- private helpers ----------------------------------------------

    def _n_channels(self) -> int:
        assert self._slices is not None
        beta = self._slices["beta"]
        return beta.stop - beta.start

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
