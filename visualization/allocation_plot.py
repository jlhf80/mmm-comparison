"""Chart 3: implied budget allocation vs ground-truth optimum at t=T.

The headline chart.  Two views over a Monte Carlo sweep of seeds:

- `plot_allocation_shares`: grouped bars of mean budget *share* per channel
  (optimal + each fitter).  Lets the reader see which channels each model
  over- or under-weights relative to the forward-looking optimum.
- `plot_allocation_error_distribution`: boxplot of per-seed L1-share error
  across fitters.  Compresses the story into one pane — DLM's error
  distribution should sit well below the other two.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from simulation.results import SimulationResult


def _budget_to_share_row(
    budget: dict[str, float], channel_names: Sequence[str]
) -> np.ndarray:
    """Convert a channel→spend dict to a share vector in `channel_names` order."""
    values = np.array([float(budget[c]) for c in channel_names], dtype=float)
    total = values.sum()
    if total <= 0.0:
        raise ValueError("budget total must be positive to compute shares")
    return values / total


def _stack_shares(
    results: Sequence[SimulationResult],
) -> tuple[tuple[str, ...], dict[str, np.ndarray], np.ndarray]:
    """Return (channel_names, model_name → (N, C) share matrix, optimal (N, C))."""
    if not results:
        raise ValueError("results is empty")
    channel_names = results[0].channel_names
    model_names = list(results[0].models.keys())

    optimal = np.vstack(
        [_budget_to_share_row(r.optimal_budget, channel_names) for r in results]
    )
    model_shares = {
        name: np.vstack(
            [
                _budget_to_share_row(r.models[name].allocated_budget, channel_names)
                for r in results
            ]
        )
        for name in model_names
    }
    return channel_names, model_shares, optimal


def plot_allocation_shares(
    results: Sequence[SimulationResult],
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (9.0, 4.0),
) -> Figure:
    """Grouped bar chart: mean budget share per channel, optimal vs each fitter.

    Error bars show ±1 SD across seeds.  Groups are channels; within each
    group the first bar is the ground-truth optimum, then one bar per
    fitter in `results[0].models` order.
    """
    channel_names, model_shares, optimal = _stack_shares(results)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    labels = ["optimal", *model_shares.keys()]
    values = [optimal, *model_shares.values()]

    n_channels = len(channel_names)
    n_bars = len(labels)
    x = np.arange(n_channels, dtype=float)
    bar_width = 0.8 / n_bars

    for i, (label, arr) in enumerate(zip(labels, values, strict=True)):
        offset = (i - (n_bars - 1) / 2.0) * bar_width
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)
        ax.bar(x + offset, means, width=bar_width, yerr=stds, capsize=3,
               label=label, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(list(channel_names))
    ax.set_ylabel("budget share at t=T")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def plot_allocation_error_distribution(
    results: Sequence[SimulationResult],
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6.0, 4.0),
) -> Figure:
    """Boxplot of per-seed L1-share allocation error across fitters.

    One box per model.  Lower is better.  This is the single-pane summary
    of the project's punchline — fit quality does not determine allocation
    quality once coefficients drift.
    """
    if not results:
        raise ValueError("results is empty")
    model_names = list(results[0].models.keys())
    errors = [
        np.array([r.models[name].allocation_error for r in results])
        for name in model_names
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.boxplot(errors, tick_labels=model_names, showfliers=True)
    ax.set_ylabel("allocation error (L1 on shares)")
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    return fig
