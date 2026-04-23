# Piece 1 finishing touches: LinkedIn post + DLM component-decomposition chart

**Date:** 2026-04-22
**Repo:** `mmm-comparison` (piece 1 of 3 LinkedIn portfolio series)
**Status:** approved — ready for implementation planning

## Context

The three-model MMM comparison (Robyn / PyMC / DLM on a time-varying-coefficient DGP) is code-complete after PR #10. The repo now ships a retuned default DGP, a structural DLM, three narrative charts (coef+bands, residual structure, allocation error/shares), a full Monte Carlo sweep, and 138 tests. What remains before shipping piece 1 publicly are two artifacts:

1. A gitignored LinkedIn post draft that turns the existing charts into a ~500-word post.
2. A DLM-only diagnostic appendix chart that shows how the structural DLM decomposes `y` into its component states (level, slope, seasonal, and per-channel β). This chart is a repo-only artifact — it's for readers who click through to the repo after reading the post, not featured in the post itself.

The two are combined in a single design because they ship together and the post will reference the repo where the chart lives.

## LinkedIn post

**File and gitignore.** Single markdown file `LINKEDIN_POST.md` at the repo root. Add `LINKEDIN_POST.md` under a new `# LinkedIn draft` block in `.gitignore` (sibling to the existing `# Personal study notes` block that hides `DLM_TUTORIAL.*`). The file is never committed; it lives locally and gets copy-pasted into LinkedIn's composer when ready.

**Angle.** *"Time-varying β is the blind spot."* Most MMMs quietly assume channel effectiveness doesn't change over the fit window. On three years of data it does, and pretending otherwise produces a biased budget. The post demonstrates the blind spot with three charts that already live in the repo.

**Voice.** First-person analyst — "I wanted to know… I simulated… I found…". Portfolio-honest, not thinkpiece. Target length ~500 words.

**Structure (section-by-section word budget).**

| # | Section | Words | Purpose |
|---|---|---|---|
| 1 | Hook | ~40 | Single-sentence claim about the blind spot + one-line on what was built. |
| 2 | Setup | ~80 | DGP: 156 weeks, 3 channels (TV declining, Digital rising, Search stable), identical pre-transformed features across all three models. Only β-dynamics differ across fitters. |
| 3 | Chart 1 — coef+bands commentary | ~120 | Robyn = point (no uncertainty). PyMC = horizontal posterior band (honest but immobile). DLM = smoothed trajectory with 95% credible band. Explicit callout of the Search panel where PyMC's band excludes the truth — the cleanest misspecification visual in the project. |
| 4 | Chart 2 — residual structure commentary | ~100 | Sinusoidal residuals = misspecified model, not noise. Diagnostic framing: RMSE can look OK even when a model is structurally wrong. DLM residuals are white noise. |
| 5 | Chart 3 — allocation error commentary | ~100 | MC sweep medians (DLM ~0.11, Robyn ~0.18, PyMC ~0.20 — L1 on shares). Core point: the question this data answers (next-quarter allocation) is not the question RMSE is scoring. |
| 6 | Punchline + repo link | ~50 | Time-invariance is a modeling *choice*, not a physical assumption. GitHub link. |

Total: ~490 words, leaving headroom under 500.

**Copy-paste convention.** The markdown file is plain prose suitable for pasting into LinkedIn's composer. No inline `![image](...)` syntax (LinkedIn strips it). A terminal `## Charts to attach (in order)` section lists the three PNG paths — `results/coef_trajectories.png`, `results/residual_structure.png`, `results/allocation_error.png` — so you know which images to drag into the composer and in what order.

**Drafting discipline.** Numbers in the post must match what `python main.py` currently produces (with the retuned DGP). Before publishing, regenerate charts with a full-quality sweep (`python main.py --n-seeds 50`) and refresh the quoted allocation medians against the new `summary.json`.

## DLM component-decomposition chart

**File and function.**

```python
# visualization/dlm_components_plot.py

def plot_dlm_components(
    dlm: DLMModel,
    channel_names: Sequence[str],
    *,
    figsize: tuple[float, float] = (12.0, 6.0),
) -> Figure
```

Validates upfront that the DLM was fit with LLT + seasonal enabled by checking `dlm.state_slices` contains keys `"level"`, `"slope"`, `"seasonal"`, and `"beta"`. If a required block is missing, raises `ValueError("plot_dlm_components requires a structural DLM (LLT + seasonal)")`.

**Layout.** 2 rows × 3 cols.

```
┌─────────┬─────────┬──────────┐
│  level  │  slope  │ seasonal │
├─────────┼─────────┼──────────┤
│  β_tv   │β_digital│ β_search │
└─────────┴─────────┴──────────┘
```

**Per-panel drawing.** Each subplot draws:

- A solid colored line for the smoothed mean — pulled from `dlm.component_trajectory(name)` (and from the specific column for the β row).
- A shaded ±1.96σ band — pulled from `dlm.component_trajectory_std(name)` with the same column indexing. Band color matches the line via `ax.plot(...)[0].get_color()`, alpha=0.15, linewidth=0. Matches the convention already used in `visualization/coef_plot.py`.
- A panel title naming the component.
- x-axis shared across all panels ("week"); y-axis free per panel (components have different scales).

**Panel specifics.**

- **level**: `component_trajectory("level")[:, 0]` with matching std. Title "level (baseline)".
- **slope**: `component_trajectory("slope")[:, 0]`. Title "slope (Δ level / week)". Usually visually flat; that's the diagnostic — the slope is the data-driven estimate of the baseline's linear-trend rate.
- **seasonal**: `component_trajectory("seasonal")[:, 0]` — the observed first state of the Fourier pair. Title "seasonal (harmonic 1)". Docstring notes that the sibling column (phase partner) isn't directly observed and is kept out of the plot. For the multi-harmonic case (not currently used in the project), only the first harmonic's first state is plotted; we don't attempt to reconstruct a composite seasonal signal here.
- **β_channel**: one panel per channel from `component_trajectory("beta")[:, c]` and `component_trajectory_std("beta")[:, c]`. Title = channel name.

**Integration.**

- `visualization/__init__.py` — re-export `plot_dlm_components` alongside existing plot functions.
- `main.py::_intuition_charts` — add one call after the coef/residual saves, writing `output_dir / "dlm_components.png"` at 150 dpi. Add `dlm_components.png — smoothed trajectory per structural block (appendix)` to the module-level docstring's output manifest.
- `notebook.ipynb` — after the existing Punchline cell (cell 15), append two new cells:
  - A markdown cell titled "Appendix: what the DLM is doing internally" with ~60 words framing — the chart is a look inside the structural decomposition: level + slope + seasonal + β states summing to the observed y.
  - A code cell `plot_dlm_components(dlm, channel_names); plt.show()`.

**Tests — `tests/test_visualization.py`.**

- `test_plot_dlm_components_returns_figure_with_six_axes`: fit a short structural DLM on synthetic (T=40, C=3) data, assert `isinstance(fig, Figure)` and `len(fig.axes) == 6`. Use matplotlib Agg backend like the other visualization tests.
- `test_plot_dlm_components_rejects_nonstructural_dlm`: fit plain `DLMModel()` (no LLT, no seasonal) on random data, assert `ValueError` with message matching `"structural"`.
- `test_plot_dlm_components_rejects_wrong_length_names`: structural DLM with 3 channels, pass 2 names, assert `ValueError`. Matches the existing contract on `plot_coefficient_trajectories`.

**Non-goals.**

- Filter vs smoother comparison panels (pedagogically valuable but a separate chart in its own right; out of scope here).
- Innovation diagnostics (ACF, rolling variance).
- Cross-correlation-aware uncertainty for multi-harmonic seasonal blocks — in practice we always use 1 harmonic, so the per-column std is exact for what's plotted.
- Featuring this chart in the LinkedIn post. The post stays at three charts.

## Delivery

**Branch and PR.** `feat/post-and-appendix-chart` off `main`, one PR containing:

- `visualization/dlm_components_plot.py` (new)
- `visualization/__init__.py` (re-export)
- `main.py` (one call + docstring update)
- `notebook.ipynb` (two new appendix cells)
- `tests/test_visualization.py` (three new tests)
- `.gitignore` (one new block for `LINKEDIN_POST.md`)

The `LINKEDIN_POST.md` file itself is gitignored and doesn't appear in the diff.

**PR title.** *"Add DLM component-decomposition diagnostic chart"* (does not mention the LinkedIn post since that file isn't in the PR).

**Acceptance.** `pytest -q` passes with 3 new tests (141 total). `ruff check .` clean. `python main.py --fast --n-seeds 10 --output results/appendix` produces `dlm_components.png` showing 6 populated panels with shaded bands, and the existing 4 charts still render.

## References

- Uses `component_trajectory_std` added in PR #10 (`models/dlm_model.py`).
- Uses `beta_posterior_samples` from PR #10 (`models/pymc_model.py`) — not strictly needed here but part of the same "expose what's already computed" pattern.
- Seasonal state layout documented in `models/dlm_model.py::build_structural_state_space` (Fourier rotation pairs, observation loading (1, 0) per harmonic).
- Existing chart conventions in `visualization/coef_plot.py` (color capture, band alpha, Agg backend testing).
