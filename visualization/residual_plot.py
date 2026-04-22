"""Chart 2: fitted vs actual + residual structure over time.

Two stacked panes per model: top shows y_t and ŷ_t superimposed, bottom shows
the residual e_t = y_t - ŷ_t.  Misspecification from fitting a time-invariant
β to a time-varying DGP shows up as *structured* residuals (trend, seasonality
leak, autocorrelation).  The DLM pane should look like white noise; the other
two should not.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_fit_and_residuals(
    y: np.ndarray,
    model_predictions: dict[str, np.ndarray],
    *,
    figsize_per_model: tuple[float, float] = (9.0, 3.2),
) -> Figure:
    """Render a fit+residual pair of panes for each model.

    Parameters
    ----------
    y:
        (T,) observed response.
    model_predictions:
        Mapping from model name to (T,) predicted ŷ.  One row of panes is
        drawn per entry, in insertion order.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D; got shape {y.shape}")

    names = list(model_predictions.keys())
    if not names:
        raise ValueError("model_predictions is empty")

    n_models = len(names)
    fig, axes = plt.subplots(
        2,
        n_models,
        figsize=(figsize_per_model[0] * n_models / 3.0 + 3.0,
                 figsize_per_model[1] * 2.0),
        sharex=True,
        squeeze=False,
    )
    t = np.arange(y.shape[0])

    for i, name in enumerate(names):
        yhat = np.asarray(model_predictions[name], dtype=float)
        if yhat.shape != y.shape:
            raise ValueError(
                f"{name}: predictions shape {yhat.shape} != y shape {y.shape}"
            )
        residuals = y - yhat

        ax_top = axes[0, i]
        ax_top.plot(t, y, color="black", lw=1.2, label="actual")
        ax_top.plot(t, yhat, color="tab:blue", lw=1.2, alpha=0.85, label="fitted")
        ax_top.set_title(name)
        if i == 0:
            ax_top.set_ylabel("y")
        if i == n_models - 1:
            ax_top.legend(loc="best", frameon=False, fontsize=9)

        ax_bot = axes[1, i]
        ax_bot.axhline(0.0, color="black", lw=0.8, alpha=0.5)
        ax_bot.plot(t, residuals, color="tab:red", lw=1.0)
        ax_bot.set_xlabel("week")
        if i == 0:
            ax_bot.set_ylabel("residual")

    fig.tight_layout()
    return fig
