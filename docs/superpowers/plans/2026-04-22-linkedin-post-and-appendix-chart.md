# LinkedIn post + DLM component-decomposition chart — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship piece 1's finishing touches — a gitignored `LINKEDIN_POST.md` draft and a DLM-only diagnostic chart (`visualization/dlm_components_plot.py`) showing the structural decomposition (level / slope / seasonal / β per channel) in a 2×3 grid. One PR; post file never committed.

**Architecture:** New visualization module reads existing DLM accessors (`component_trajectory`, `component_trajectory_std` from PR #10) and draws six subplots using the same color-capture band convention as `visualization/coef_plot.py`. Wired into `main.py` as an additional saved PNG and into `notebook.ipynb` as two new appendix cells. The LinkedIn post is a local-only markdown draft; only its `.gitignore` line appears in the PR.

**Tech Stack:** Python 3.11+, matplotlib (Agg backend for tests), pytest, ruff. All already in `pyproject.toml`.

**Spec:** `docs/superpowers/specs/2026-04-22-linkedin-post-and-appendix-chart-design.md`

**Branch:** `feat/post-and-appendix-chart` (already created off `main`, currently contains the committed spec).

---

## Pre-flight check

- [ ] **Verify branch and working tree are clean**

```bash
git branch --show-current
git status
```

Expected: current branch `feat/post-and-appendix-chart`; only untracked is `DLM_TUTORIAL.html` (Jupyter export noise — ignore it). If `notebook.ipynb` is dirty with Jupyter cell-ID churn, `git stash push notebook.ipynb -m "pre-task stash"` before starting.

- [ ] **Verify test baseline**

```bash
python -m pytest -q
```

Expected: `138 passed`. If it's different, stop and resolve before starting Task 1.

---

## Task 1: Gitignore LINKEDIN_POST.md

**Files:**
- Modify: `/Users/jameshenson/Documents/mmm-comparison/.gitignore`

- [ ] **Step 1: Read the current gitignore tail**

```bash
tail -6 .gitignore
```

Expected output (note: `DLM_TUTORIAL.md` is an exact filename, not a wildcard):

```
# Claude Code local state (private to this machine)
.claude/
CLAUDE.md

# Personal study notes (not part of the public portfolio)
DLM_TUTORIAL.md
```

- [ ] **Step 2: Append the LinkedIn draft block**

Edit `.gitignore`. Add these three lines at the very end of the file (after the `DLM_TUTORIAL.md` line and its trailing blank line):

```
# LinkedIn draft (written locally, never committed)
LINKEDIN_POST.md
```

- [ ] **Step 3: Verify the entry is recognised**

```bash
touch LINKEDIN_POST.md
git status
rm LINKEDIN_POST.md
```

Expected: `git status` shows `.gitignore` as modified and does NOT list `LINKEDIN_POST.md` as untracked. (The `touch` + `rm` is just the check — we'll create the real file in Task 6.)

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "$(cat <<'EOF'
Gitignore LINKEDIN_POST.md for local-only post draft

The piece 1 LinkedIn post is written locally and pasted into
LinkedIn's composer by hand — never committed.  New sibling block to
the existing DLM_TUTORIAL personal-notes pattern.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Create the DLM component-decomposition chart module (TDD)

**Files:**
- Test: `/Users/jameshenson/Documents/mmm-comparison/tests/test_visualization.py`
- Create: `/Users/jameshenson/Documents/mmm-comparison/visualization/dlm_components_plot.py`

- [ ] **Step 1: Add imports to the test file**

Open `tests/test_visualization.py`. Find the existing import block (around line 19–26) and add these two imports. The first imports `DLMModel` (not currently imported in this file); the second imports the function we're about to create (so pytest will fail at import time until the module exists).

Current import block to extend:

```python
from dgp.config import ChannelConfig, DGPConfig
from simulation.results import ModelResult, SimulationResult
from visualization.allocation_plot import (
    plot_allocation_error_distribution,
    plot_allocation_shares,
)
from visualization.coef_plot import plot_coefficient_trajectories
from visualization.residual_plot import plot_fit_and_residuals
```

Add after the `residual_plot` import:

```python
from models.dlm_model import DLMModel
from visualization.dlm_components_plot import plot_dlm_components
```

- [ ] **Step 2: Add the three failing tests**

In `tests/test_visualization.py`, find the existing `# --- coef_plot ---` section header (roughly line 81). The coef-plot section runs through `test_plot_coefficient_trajectories_rejects_band_for_unknown_model` (added in PR #10). After that test and before the next section header (`# --- residual_plot ---`), insert a new section and three tests:

```python
# --- dlm_components_plot ---------------------------------------------------


def _structural_dlm(n_channels: int = 3, n_obs: int = 40, seed: int = 0) -> DLMModel:
    """Fit a small structural DLM for visualization-test fixtures."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_obs, n_channels))
    y = rng.normal(size=n_obs)
    return DLMModel(
        local_linear_trend=True,
        seasonal_period=52.0,
        seasonal_harmonics=1,
        level_innovation_var=1e-6,
        slope_innovation_var=1e-8,
        seasonal_innovation_var=0.0,
        beta_innovation_var=5e-3,
        observation_var=0.25,
        initial_var=10.0,
    ).fit(X, y)


def test_plot_dlm_components_returns_figure_with_six_axes():
    """Structural DLM fit produces exactly 6 component panels."""
    dlm = _structural_dlm(n_channels=3)
    fig = plot_dlm_components(dlm, ["a", "b", "c"])
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 6
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_plot_dlm_components_rejects_nonstructural_dlm():
    """Plain DLM lacks level/slope/seasonal — must raise clearly."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 2))
    y = rng.normal(size=30)
    dlm = DLMModel().fit(X, y)
    with pytest.raises(ValueError, match="structural"):
        plot_dlm_components(dlm, ["a", "b"])


def test_plot_dlm_components_rejects_wrong_length_names():
    """Number of channel names must match the β block width."""
    dlm = _structural_dlm(n_channels=3)
    with pytest.raises(ValueError, match="channel names"):
        plot_dlm_components(dlm, ["a", "b"])
```

- [ ] **Step 3: Run the tests to verify they fail for the right reason**

```bash
python -m pytest tests/test_visualization.py -q 2>&1 | tail -15
```

Expected: `ImportError` (or a collection error) because `visualization.dlm_components_plot` doesn't exist yet. That's the correct failure mode before we add the module.

- [ ] **Step 4: Create `visualization/dlm_components_plot.py`**

Create a new file at `/Users/jameshenson/Documents/mmm-comparison/visualization/dlm_components_plot.py` with this exact content:

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python -m pytest tests/test_visualization.py -q 2>&1 | tail -10
```

Expected: `15 passed` (12 existing visualization tests from PR #10 + 3 new). No failures, no warnings beyond the existing ruff-unrelated ones.

- [ ] **Step 6: Ruff-check the new module**

```bash
ruff check visualization/dlm_components_plot.py tests/test_visualization.py
```

Expected: `All checks passed!`. If there are N803/N806 warnings in `tests/test_visualization.py` from the new code, they should be suppressed by the existing per-file ignore in `pyproject.toml`. If new ignores are needed, prefer fixing the variable names over widening ignores.

- [ ] **Step 7: Commit**

```bash
git add visualization/dlm_components_plot.py tests/test_visualization.py
git commit -m "$(cat <<'EOF'
Add DLM component-decomposition diagnostic chart

New visualization module renders the structural DLM's smoothed state
components in a 2×3 grid: level, slope, seasonal on top; per-channel β
on the bottom.  Each panel carries a ±1.96σ credible band from the
diagonal of the RTS-smoothed covariance, color-matched to its line via
the same convention as the coef chart.

Intended as a repo-only appendix artifact: reveals what the DLM
actually decomposes y into, without cluttering the three narrative
charts.  Raises cleanly if invoked on a non-structural DLM or on a
DLM with channel-count ≠ 3.

Tests: three new unit tests in test_visualization.py cover the happy
path, non-structural rejection, and name/channel-count mismatch.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Re-export from the visualization package

**Files:**
- Modify: `/Users/jameshenson/Documents/mmm-comparison/visualization/__init__.py`

- [ ] **Step 1: Read the current `__init__.py`**

```bash
cat visualization/__init__.py
```

Expected content:

```python
from visualization.allocation_plot import (
    plot_allocation_error_distribution,
    plot_allocation_shares,
)
from visualization.coef_plot import plot_coefficient_trajectories
from visualization.residual_plot import plot_fit_and_residuals

__all__ = [
    "plot_allocation_error_distribution",
    "plot_allocation_shares",
    "plot_coefficient_trajectories",
    "plot_fit_and_residuals",
]
```

- [ ] **Step 2: Rewrite `__init__.py` to include the new export**

Replace the contents of `visualization/__init__.py` with exactly:

```python
from visualization.allocation_plot import (
    plot_allocation_error_distribution,
    plot_allocation_shares,
)
from visualization.coef_plot import plot_coefficient_trajectories
from visualization.dlm_components_plot import plot_dlm_components
from visualization.residual_plot import plot_fit_and_residuals

__all__ = [
    "plot_allocation_error_distribution",
    "plot_allocation_shares",
    "plot_coefficient_trajectories",
    "plot_dlm_components",
    "plot_fit_and_residuals",
]
```

- [ ] **Step 3: Verify the import surface**

```bash
python -c "from visualization import plot_dlm_components; print(plot_dlm_components.__module__)"
```

Expected: `visualization.dlm_components_plot`.

- [ ] **Step 4: Commit**

```bash
git add visualization/__init__.py
git commit -m "$(cat <<'EOF'
Re-export plot_dlm_components from visualization package

Makes the new appendix chart reachable via
`from visualization import plot_dlm_components` in main.py and the
notebook, matching the existing convention for the other four plot
functions.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Wire the chart into `main.py`

**Files:**
- Modify: `/Users/jameshenson/Documents/mmm-comparison/main.py` (imports, docstring, `_intuition_charts`)

- [ ] **Step 1: Update the module docstring's output manifest**

Find the docstring block at the top of `main.py` (currently at `main.py:1-17`). The output manifest currently reads:

```
    coef_trajectories.png       — estimated vs true β_{c,t}, seed 0
    residual_structure.png      — fit vs actual + residuals, seed 0
    allocation_shares.png       — mean budget share per channel, MC sweep
    allocation_error.png        — per-seed L1 share error, MC sweep
    summary.json                — per-model allocation error summary stats
```

Replace it with:

```
    coef_trajectories.png       — estimated vs true β_{c,t}, seed 0
    residual_structure.png      — fit vs actual + residuals, seed 0
    dlm_components.png          — smoothed trajectory per structural block (appendix), seed 0
    allocation_shares.png       — mean budget share per channel, MC sweep
    allocation_error.png        — per-seed L1 share error, MC sweep
    summary.json                — per-model allocation error summary stats
```

(One inserted line between `residual_structure.png` and `allocation_shares.png`.)

- [ ] **Step 2: Add the import**

Find the existing import block (roughly `main.py:34-39`):

```python
from visualization.allocation_plot import (
    plot_allocation_error_distribution,
    plot_allocation_shares,
)
from visualization.coef_plot import plot_coefficient_trajectories
from visualization.residual_plot import plot_fit_and_residuals
```

Replace it with (adds one new import, sorted alphabetically to satisfy ruff `I001`):

```python
from visualization.allocation_plot import (
    plot_allocation_error_distribution,
    plot_allocation_shares,
)
from visualization.coef_plot import plot_coefficient_trajectories
from visualization.dlm_components_plot import plot_dlm_components
from visualization.residual_plot import plot_fit_and_residuals
```

- [ ] **Step 3: Add the chart save call inside `_intuition_charts`**

Find the `_intuition_charts` function in `main.py`. The current tail of its body looks like this (roughly `main.py:84-90`):

```python
    predictions = {
        "robyn": robyn.predict(X),
        "pymc": pymc.predict(X),
        "dlm": dlm.fitted_values(),
    }
    resid_fig = plot_fit_and_residuals(y, predictions)
    resid_fig.savefig(output_dir / "residual_structure.png", dpi=150)
```

Add two lines immediately after the `resid_fig.savefig(...)` call (i.e., before the function's closing):

```python
    components_fig = plot_dlm_components(dlm, channel_names)
    components_fig.savefig(output_dir / "dlm_components.png", dpi=150)
```

The function body now ends with the new `components_fig.savefig(...)` line.

- [ ] **Step 4: Verify the CLI still parses and imports without error**

```bash
python -c "import main; print('ok')"
```

Expected: prints `ok`. No ImportError.

- [ ] **Step 5: Smoke-run the CLI in fast mode against a scratch output directory**

```bash
rm -rf results/smoke && python main.py --fast --n-seeds 3 --output results/smoke
ls -la results/smoke/
```

Expected: `results/smoke/` contains five PNGs (coef, residual, `dlm_components`, allocation_shares, allocation_error) and `summary.json`. All five PNG files should be non-empty (≥ 10 KB).

- [ ] **Step 6: Ruff-check the modified files**

```bash
ruff check main.py
```

Expected: `All checks passed!`.

- [ ] **Step 7: Commit**

```bash
git add main.py
git commit -m "$(cat <<'EOF'
Wire plot_dlm_components into main.py intuition charts

Adds a fifth PNG (dlm_components.png) generated on the seed-0 single-
fit pass, alongside the existing coef and residual charts.  Updates
the module-level docstring output manifest so `python main.py --help`
and any reader of the file knows to expect it.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Append appendix cells to the notebook

**Files:**
- Modify: `/Users/jameshenson/Documents/mmm-comparison/notebook.ipynb`

The notebook currently has 16 cells (index 0–15; cell 15 is the Punchline markdown). We append one markdown cell and one code cell at the end, forming the appendix.

- [ ] **Step 1: Check notebook state**

```bash
python -c "
import json
nb = json.load(open('notebook.ipynb'))
print(f'{len(nb[\"cells\"])} cells total')
print(f'last cell type: {nb[\"cells\"][-1][\"cell_type\"]}')
print(f'last cell source preview: {\" \".join(nb[\"cells\"][-1][\"source\"])[:120]}')
"
```

Expected: `16 cells total`, `last cell type: markdown`, preview starts with `## Punchline` or similar.

If `git status` shows `notebook.ipynb` as already modified (Jupyter cell-ID churn from running it locally), run `git stash push notebook.ipynb -m "pre-task5 stash"` and re-check.

- [ ] **Step 2: Append the appendix cells via a JSON edit**

Run this exact script:

```bash
python - <<'PY'
import json
import uuid

with open('notebook.ipynb') as f:
    nb = json.load(f)

md_cell = {
    "cell_type": "markdown",
    "id": uuid.uuid4().hex[:8],
    "metadata": {},
    "source": [
        "## Appendix: what the DLM is doing internally\n",
        "\n",
        "Chart 1 shows the DLM's β̂ trajectory with uncertainty — that's the *output*. Here's the *decomposition*: the structural DLM splits y into an additive mix of baseline level, baseline slope, a Fourier seasonal signal, and a per-channel β state. Each panel below plots the smoothed posterior mean and its 95% credible band.\n",
        "\n",
        "This is the internal structure that makes the coefficient-recovery chart possible — baseline and seasonal variance live in their own dedicated states instead of leaking into β̂."
    ],
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": uuid.uuid4().hex[:8],
    "metadata": {},
    "outputs": [],
    "source": [
        "from visualization.dlm_components_plot import plot_dlm_components\n",
        "\n",
        "fig = plot_dlm_components(dlm, channel_names)\n",
        "plt.show()"
    ],
}

nb['cells'].append(md_cell)
nb['cells'].append(code_cell)

with open('notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
    f.write('\n')

print(f"notebook now has {len(nb['cells'])} cells")
PY
```

Expected output: `notebook now has 18 cells`.

- [ ] **Step 3: Verify the notebook is still valid JSON + correct cell types**

```bash
python -c "
import json
nb = json.load(open('notebook.ipynb'))
assert len(nb['cells']) == 18, f\"expected 18 cells, got {len(nb['cells'])}\"
assert nb['cells'][16]['cell_type'] == 'markdown', 'cell 16 should be markdown'
assert nb['cells'][17]['cell_type'] == 'code', 'cell 17 should be code'
assert 'plot_dlm_components' in ''.join(nb['cells'][17]['source']), 'cell 17 must call plot_dlm_components'
print('notebook structure OK')
"
```

Expected: `notebook structure OK`.

- [ ] **Step 4: Verify nbformat doesn't reject the file**

```bash
python -c "
import nbformat
nb = nbformat.read('notebook.ipynb', as_version=4)
nbformat.validate(nb)
print('nbformat validation OK')
"
```

Expected: `nbformat validation OK`. If this raises, fix the cell structure before committing.

- [ ] **Step 5: Commit**

```bash
git add notebook.ipynb
git commit -m "$(cat <<'EOF'
Add DLM components appendix to notebook

Two new cells after the Punchline: a markdown intro framing the chart
as 'what the DLM is doing internally', followed by a code cell calling
plot_dlm_components(dlm, channel_names).  Placement keeps the three
narrative charts + punchline as the main flow and makes the
decomposition a quiet afterword for readers who want the mechanism.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Draft `LINKEDIN_POST.md` locally (no commit)

The file is gitignored and never committed. This task creates the local draft so the repo is fully usable for drafting. It does NOT produce a commit — verify at the end that `git status` is clean and `LINKEDIN_POST.md` is listed nowhere.

**Files:**
- Create (gitignored): `/Users/jameshenson/Documents/mmm-comparison/LINKEDIN_POST.md`

- [ ] **Step 1: Create the file with the full draft**

Write a file at `/Users/jameshenson/Documents/mmm-comparison/LINKEDIN_POST.md` containing exactly this content (copy verbatim — the Write tool is fine, since the file is gitignored, it will stay local):

```markdown
## Time-varying β is the blind spot most MMMs pretend away

Most MMMs encode an assumption they never name: channel effectiveness doesn't change over the fit window. On three years of weekly data, it does — and pretending otherwise biases the budget.

I spent a few nights rebuilding a small comparison of three MMM styles on a synthetic DGP with drifting channel ROI. 156 weeks, 3 channels (TV declining, Digital rising, Search stable), adstock + Hill saturation, noisy observation. Same pre-transformed features go into all three fitters. The only thing that varies is whether each model lets β move.

Robyn (Ridge + Nevergrad) produces a point estimate — no uncertainty at all. PyMC-Marketing runs a full Bayesian posterior, so it shows honest uncertainty as a horizontal shaded band — but because the model assumes β is constant, that band can't actually *move*. The DLM (Kalman filter + RTS smoother on a random-walk-with-drift state) does move: a smoothed trajectory for β̂_{c,t} with a 95% credible interval. On the Search panel, the true β sits visibly outside PyMC's band — the cleanest visual of "wrong and honest" in the project.

If your model is structurally too small, the unmodeled signal ends up in the residuals. Robyn's and PyMC's residuals come out as clean annual sinusoids — the seasonality their time-invariant coefficients can't absorb. The DLM's residuals look like white noise. A sinusoidal residual is a model telling you it's wrong, even when RMSE looks plausible.

The real test is the decision. I ran a Monte Carlo sweep of 50 seeds, each time asking the budget optimizer for the next-quarter allocation under each model's β̂_T. L1 error on shares (max 2.0) against the ground-truth optimum: DLM median 0.11, Robyn 0.18, PyMC 0.20. Two models with similar in-sample fit produce systematically different budget recommendations. That's the whole point — "how should I allocate next quarter?" is not the question that fit metrics are scoring.

Time-invariance isn't a physical truth about marketing channels; it's a modeling *choice* that gets absorbed silently into every downstream decision. If your channel effectiveness might drift — and it probably does — the right model is one that can move with it.

Code, notebook, and the hand-rolled DLM (Kalman filter + RTS smoother, no library wrapper): https://github.com/jlhf80/mmm-comparison

---

## Charts to attach (in order)

1. `results/coef_trajectories.png` — β̂_{c,t} vs truth with DLM/PyMC credible bands. Search panel shows PyMC's band excluding the truth.
2. `results/residual_structure.png` — fit + residual structure per model. DLM residuals are white noise; others are annual sinusoids.
3. `results/allocation_error.png` — L1 allocation error distribution over 50 seeds.

---

## Pre-publish checklist

- [ ] Rerun `python main.py --n-seeds 50 --output results/` (full-quality sweep, not `--fast`) before posting.
- [ ] Cross-check the quoted allocation medians (0.11 / 0.18 / 0.20) against `results/summary.json` from that run. Update the numbers in the post if they drifted.
- [ ] Rename the three attachments to something LinkedIn-friendly (e.g., `chart-1-coef-bands.png`) before uploading — LinkedIn shows the filename in hover text.
- [ ] Paste the body only; drag the three PNGs in manually. LinkedIn's composer strips markdown image syntax.
```

- [ ] **Step 2: Verify the file is ignored**

```bash
ls -la LINKEDIN_POST.md
git status
```

Expected: the file exists (ls succeeds, size ≳ 3 KB); `git status` does NOT mention it under "Untracked files". If it DOES appear in git status, stop — it means the `.gitignore` entry from Task 1 didn't land. Re-run `cat .gitignore | tail -5` to confirm the `LINKEDIN_POST.md` line is there.

- [ ] **Step 3: Do not commit**

There is deliberately no `git commit` step here. The file is local-only. Proceed to Task 7.

---

## Task 7: End-to-end verification, push, and open PR

**Files:** no file edits; final integration check and PR creation.

- [ ] **Step 1: Run the full test suite**

```bash
python -m pytest -q
```

Expected: `141 passed` (138 baseline + 3 new). No failures.

- [ ] **Step 2: Run ruff**

```bash
ruff check .
```

Expected: `All checks passed!`.

- [ ] **Step 3: Run the fast-mode CLI end-to-end and inspect**

```bash
rm -rf results/final && python main.py --fast --n-seeds 10 --output results/final
ls -la results/final/
```

Expected: six files in `results/final/`: `coef_trajectories.png`, `residual_structure.png`, `dlm_components.png`, `allocation_shares.png`, `allocation_error.png`, `summary.json`. The `dlm_components.png` file should be ≥ 40 KB (6-panel plot).

Open `results/final/dlm_components.png` visually. It should show six subplots in a 2×3 grid with titles `level (baseline)`, `slope (Δ level / week)`, `seasonal (harmonic 1)`, `tv`, `digital`, `search`, each with a line and a shaded band. If any panel is empty or an axis label is missing, stop and diagnose.

- [ ] **Step 4: Confirm the LINKEDIN_POST.md is not going to be pushed**

```bash
git status
git ls-files --others --exclude-standard
```

Expected: `git status` shows the branch at 5 commits ahead of main (one per task 1–5) with nothing staged or unstaged; `git ls-files --others --exclude-standard` returns nothing (LINKEDIN_POST.md is gitignored; DLM_TUTORIAL.html is gitignored via `DLM_TUTORIAL.md` — wait, it's NOT gitignored; see below).

**Note on `DLM_TUTORIAL.html`:** the current `.gitignore` has only `DLM_TUTORIAL.md`, not `DLM_TUTORIAL.*`. If that `.html` export exists in the working tree it will show up as untracked. That's pre-existing, unrelated to this PR, and should be left alone — do NOT widen the pattern here.

- [ ] **Step 5: Push the branch**

```bash
git push -u origin feat/post-and-appendix-chart
```

Expected: five commits pushed; branch tracking set.

- [ ] **Step 6: Open the PR**

```bash
gh pr create --title "Add DLM component-decomposition diagnostic chart" --body "$(cat <<'EOF'
## Summary
- New repo-only appendix chart `dlm_components.png` rendered by `visualization/dlm_components_plot.plot_dlm_components`. 2×3 grid shows the structural DLM's smoothed state components (level, slope, seasonal) on the top row and per-channel β on the bottom, each with a ±1.96σ credible band from the RTS-smoothed covariance diagonal.
- Wires into `main.py._intuition_charts` as a fifth saved PNG and appends two appendix cells to `notebook.ipynb` (one markdown intro + one code cell). Updates the module docstring's output manifest.
- Gitignores `LINKEDIN_POST.md` as a sibling to the existing `DLM_TUTORIAL.md` personal-notes block. The post itself is drafted locally and never committed.

## Design notes
- Raises `ValueError` if invoked on a non-structural DLM (missing level/slope/seasonal blocks) or on a channel-count other than 3 — the grid layout assumes the project's fixed three-channel shape.
- Band color is captured via `ax.plot(...)[0].get_color()`, matching the convention in `visualization/coef_plot.py`. Same alpha (0.15), same linewidth (0).
- Seasonal panel plots only the first state of the Fourier pair (the observed component). Multi-harmonic models would need per-harmonic panels; not in scope.
- Design spec committed in the first commit of this branch at `docs/superpowers/specs/2026-04-22-linkedin-post-and-appendix-chart-design.md`.

## Test plan
- [x] `pytest -q` → 141 passed (138 existing + 3 new: happy-path six-axes assertion, non-structural rejection, wrong-length names).
- [x] `ruff check .` → clean.
- [x] `python main.py --fast --n-seeds 10 --output results/final` produces all five PNGs; `dlm_components.png` renders six populated panels with shaded bands.
- [x] `nbformat.validate(...)` passes on the updated `notebook.ipynb`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: prints a PR URL (e.g., `https://github.com/jlhf80/mmm-comparison/pull/11`).

- [ ] **Step 7: Check CI**

```bash
gh pr checks --watch
```

Expected: CI job runs ruff + pytest against py 3.12 matrix (per `.github/workflows/ci.yml`), both pass. `gh pr view` should show state `OPEN` and mergeable status `MERGEABLE` once checks complete.

---

## Done-when

- PR is open with green CI.
- Local `LINKEDIN_POST.md` exists, is readable, and is NOT in `git ls-files`.
- `results/` has a fresh `dlm_components.png` that looks correct in a viewer.
- Future engineer running `python main.py --fast --n-seeds 10 --output results/` gets the same five-PNG output deterministically.

## After merge (not part of this PR)

Before actually publishing the LinkedIn post:

1. Re-run `python main.py --n-seeds 50 --output results/` for a full-quality MC sweep (not `--fast`, which uses cheap PyMC settings).
2. Open `results/summary.json` and update the three allocation medians in `LINKEDIN_POST.md` if they differ from the quoted 0.11 / 0.18 / 0.20.
3. Paste the body into LinkedIn and upload the three renamed PNGs as attachments.
