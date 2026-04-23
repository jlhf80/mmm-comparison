"""Appendix chart: DLM component decomposition.

For a structural DLM (local linear trend + Fourier seasonal), plots the
smoothed posterior of each state block in a 2×3 grid:

    level     | slope    | seasonal
    β_channel1| β_ch2    | β_ch3

Each panel draws the smoothed mean and a ±1.96σ (95%) credible band
using the diagonal of the RTS-smoothed covariance.  Band color matches
the line via `ax.plot(...)[0].get_color()` — same convention as
`visualization/coef_plot.py`.

Intended as a repo-only appendix artifact — not one of the three
narrative charts.  Reveals how the DLM decomposes y into additive
state components.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from models.dlm_model import DLMModel

_REQUIRED_STATE_BLOCKS = ("level", "slope", "seasonal", "beta")


def plot_dlm_components(
    dlm: DLMModel,
    channel_names: Sequence[str],
    *,
    figsize: tuple[float, float] = (12.0, 6.0),
) -> Figure:
    """Render the structural DLM's component decomposition as a 2×3 grid.

    Parameters
    ----------
    dlm:
        A fitted DLMModel with `local_linear_trend=True` AND
        `seasonal_period` set.  Raises `ValueError` if the DLM is plain
        (no LLT or no seasonal block).
    channel_names:
        Labels for the β panels in the bottom row; must match the number
        of β states in the model.  The chart layout assumes exactly 3
        channels (which is the project's fixed shape); other values
        raise.
    figsize:
        Figure size in inches.  Default 12×6 is sized for the 2×3 grid.

    Returns
    -------
    matplotlib.figure.Figure with exactly 6 axes.
    """
    slices = dlm.state_slices
    missing = [name for name in _REQUIRED_STATE_BLOCKS if name not in slices]
    if missing:
        raise ValueError(
            "plot_dlm_components requires a structural DLM "
            f"(LLT + seasonal); missing state blocks: {missing}"
        )

    beta_slice = slices["beta"]
    n_channels = beta_slice.stop - beta_slice.start
    if len(channel_names) != n_channels:
        raise ValueError(
            f"{len(channel_names)} channel names for {n_channels} β states"
        )
    if n_channels != 3:
        raise ValueError(
            f"plot_dlm_components layout assumes 3 channels; got {n_channels}"
        )

    fig, axes_arr = plt.subplots(2, 3, figsize=figsize, sharex=True)
    axes = axes_arr.flatten()

    level_mean = dlm.component_trajectory("level")[:, 0]
    level_std = dlm.component_trajectory_std("level")[:, 0]
    slope_mean = dlm.component_trajectory("slope")[:, 0]
    slope_std = dlm.component_trajectory_std("slope")[:, 0]
    seasonal_mean = dlm.component_trajectory("seasonal")[:, 0]
    seasonal_std = dlm.component_trajectory_std("seasonal")[:, 0]
    beta_mean = dlm.component_trajectory("beta")
    beta_std = dlm.component_trajectory_std("beta")

    t = np.arange(level_mean.shape[0])

    panels = [
        (axes[0], "level (baseline)", level_mean, level_std),
        (axes[1], "slope (Δ level / week)", slope_mean, slope_std),
        (axes[2], "seasonal (harmonic 1)", seasonal_mean, seasonal_std),
    ]
    for c in range(n_channels):
        panels.append(
            (axes[3 + c], str(channel_names[c]), beta_mean[:, c], beta_std[:, c])
        )

    for ax, title, mean, std in panels:
        line = ax.plot(t, mean, lw=1.6)[0]
        ax.fill_between(
            t,
            mean - 1.96 * std,
            mean + 1.96 * std,
            color=line.get_color(),
            alpha=0.15,
            linewidth=0,
        )
        ax.set_title(title)
        ax.set_xlabel("week")

    fig.tight_layout()
    return fig
