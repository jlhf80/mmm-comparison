"""Chart 1: estimated β̂_{c,t} vs true β_{c,t} per model, per channel.

This is where the coefficient-recovery story is visible.  The DLM's smoothed
β trajectory should hug the ground-truth random walk; Robyn's and PyMC's
time-invariant β̂ appear as horizontal lines pinned near the time-average of
the truth — the gap between a horizontal line and β_{c,T} is precisely the
error the allocation chart later quantifies.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _broadcast_to_trajectory(beta: np.ndarray, n_weeks: int) -> np.ndarray:
    """Accept either a (T, C) trajectory or a (C,) constant; return (T, C)."""
    beta = np.asarray(beta, dtype=float)
    if beta.ndim == 1:
        return np.broadcast_to(beta, (n_weeks, beta.shape[0])).copy()
    if beta.ndim == 2:
        if beta.shape[0] != n_weeks:
            raise ValueError(
                f"trajectory has {beta.shape[0]} rows; expected {n_weeks}"
            )
        return beta
    raise ValueError(f"β must be 1-D or 2-D; got shape {beta.shape}")


def plot_coefficient_trajectories(
    true_beta: np.ndarray,
    model_betas: dict[str, np.ndarray],
    channel_names: Sequence[str],
    *,
    axes: Sequence[Axes] | None = None,
    figsize: tuple[float, float] = (12.0, 3.5),
) -> Figure:
    """Plot true vs estimated β_{c,t} for each channel as a row of subplots.

    Parameters
    ----------
    true_beta:
        (T, C) ground-truth β_{c,t} trajectory in channel order.
    model_betas:
        Mapping from model name to either a (T, C) trajectory (e.g. DLM) or
        a (C,) time-invariant estimate (Robyn, PyMC).  1-D estimates are
        drawn as horizontal lines so the viewer can read off the bias.
    channel_names:
        Labels for the C subplots, in column order.
    axes:
        Optional pre-created axes of length C.  If None, a new figure is
        created; either way the figure is returned.
    """
    true_beta = np.asarray(true_beta, dtype=float)
    if true_beta.ndim != 2:
        raise ValueError(f"true_beta must be (T, C); got shape {true_beta.shape}")
    n_weeks, n_channels = true_beta.shape
    if len(channel_names) != n_channels:
        raise ValueError(
            f"{len(channel_names)} channel names for {n_channels} columns"
        )

    if axes is None:
        fig, axes_arr = plt.subplots(
            1, n_channels, figsize=figsize, sharex=True, squeeze=False
        )
        axes = list(axes_arr[0])
    else:
        if len(axes) != n_channels:
            raise ValueError(f"need {n_channels} axes; got {len(axes)}")
        fig = axes[0].figure

    t = np.arange(n_weeks)
    model_trajectories = {
        name: _broadcast_to_trajectory(beta, n_weeks)
        for name, beta in model_betas.items()
    }

    for c, (ax, name) in enumerate(zip(axes, channel_names, strict=True)):
        ax.plot(t, true_beta[:, c], color="black", lw=2.0, label="true β")
        for model_name, traj in model_trajectories.items():
            ax.plot(t, traj[:, c], lw=1.4, alpha=0.9, label=model_name)
        ax.set_title(name)
        ax.set_xlabel("week")
        if c == 0:
            ax.set_ylabel(r"$\beta_{c,t}$")

    axes[-1].legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig
