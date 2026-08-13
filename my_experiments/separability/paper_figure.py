"""Double-column paper figure for the separability experiment: the two plots of ``separability.py``, transposed to tasks-as-columns and stacked, re-plotted from the cached AUROCs."""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from separability import (ALL_TASKS, ROOT_DIR, _AXIS, _BLUE, _GRID, _GREEN, _INK, _MUTED,  # noqa: E402
                          _RED, _SECOND, _SURFACE, summarise)

#: Camera-ready names for the cache's task keys.
LABELS = {"push_t": "Push-T", "pretzel": "Pretzel", "push_chair": "Push Chair",
          "sorting": "Sorting", "stacking": "Stacking"}
#: Shared AUROC row limits; the mean curves span 0.37-0.96, so only the widest bands clip.
AUROC_YLIM = (0.30, 1.0)
SERIES = (("obs__success", "test success", _GREEN, "-", "o"),
          ("obs__failure", "test failure", _RED, "--", "s"))

# Type 42 rather than matplotlib's default Type 3, which several venues reject.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def load_curves(k: int) -> pd.DataFrame:
    """Fold-mean curves rebuilt from every cached per-fold AUROC -- no refitting."""
    paths = sorted(glob.glob(str(HERE / "cache" / "*__scores__*.csv")))
    if not paths:
        raise SystemExit("no cached scores; run separability.py first")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    print(f"{len(df)} cached fits from {len(paths)} tasks")
    return summarise(df, k)


def style(ax, xlabel: bool) -> None:
    """Recessive grid and hairline axes at print sizes."""
    ax.set_facecolor(_SURFACE)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=_MUTED, labelsize=6.5, length=2.5, width=0.6, pad=1.5)
    # Curves start at t = 0.1, so the axis is trimmed to the sampled range plus marker clearance.
    ax.set_xlim(0.06, 1.04)
    ax.set_xticks([0.2, 0.6, 1.0])
    if not xlabel:
        ax.tick_params(labelbottom=False)


def plot(curves: pd.DataFrame, model: str, stem: pathlib.Path) -> None:
    """Two rows (observation-only AUROC, then the DiD) by five task columns."""
    fig, axes = plt.subplots(2, 5, figsize=(7.0, 2.85), sharex=True)
    fig.patch.set_facecolor(_SURFACE)

    for col, task in enumerate(ALL_TASKS):
        g = curves[(curves.model == model) & (curves.task == task)].sort_values("t")
        top, bot = axes[0, col], axes[1, col]

        style(top, xlabel=False)
        top.set_title(LABELS[task], color=_INK, fontsize=8, pad=3)
        top.axhline(0.5, color=_AXIS, linewidth=0.8)
        for name, _, color, dash, marker in SERIES:
            top.fill_between(g.t, g[name + "__lo"], g[name + "__hi"], color=color, alpha=0.18,
                             linewidth=0)
            top.plot(g.t, g[name], color=color, linewidth=1.3, linestyle=dash, marker=marker,
                     markersize=2.4, markeredgewidth=0)
        top.set_ylim(*AUROC_YLIM)
        top.set_yticks([0.4, 0.6, 0.8, 1.0])

        style(bot, xlabel=True)
        bot.axhline(0.0, color=_AXIS, linewidth=0.8)
        bot.fill_between(g.t, g["did__lo"], g["did__hi"], color=_BLUE, alpha=0.18, linewidth=0)
        bot.plot(g.t, g["did"], color=_BLUE, linewidth=1.3)
        # Per-column DiD limits: push_chair's interval is ~15x sorting's, so a shared axis
        # would flatten every other task to a straight line.
        span = float(np.nanmax(np.abs(g[["did__lo", "did__hi"]].to_numpy())))
        # 1.3x, not a tight fit: it keeps the outer ticks off the spines, where their labels
        # would run into the neighbouring panels.
        bot.set_ylim(-1.3 * span, 1.3 * span)
        bot.yaxis.set_major_locator(MaxNLocator(3, symmetric=True))

        if col:  # one y scale across the AUROC row, so label it once
            top.tick_params(labelleft=False)

    axes[0, 0].set_ylabel("AUROC", color=_SECOND, fontsize=7.5, labelpad=2)
    axes[1, 0].set_ylabel("$\\Delta$AUROC", color=_SECOND, fontsize=7.5, labelpad=2)
    fig.legend(handles=[Line2D([], [], color=c, linestyle=d, marker=m, markersize=2.4,
                               markeredgewidth=0, linewidth=1.3, label=lab)
                        for _, lab, c, d, m in SERIES],
               loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False,
               fontsize=7, labelcolor=_SECOND, handlelength=2.4, columnspacing=1.8,
               handletextpad=0.5)
    fig.text(0.5, 0.028, "normalized episode time  $t$", color=_SECOND, fontsize=7.5, ha="center")
    # left clears the DiD row's tick labels, which are wider than the AUROC row's.
    fig.subplots_adjust(left=0.082, right=0.985, top=0.855, bottom=0.165, hspace=0.24, wspace=0.38)

    for ext in ("pdf", "png"):
        path = stem.with_suffix("." + ext)
        fig.savefig(path, dpi=400, facecolor=_SURFACE)
        print(f"  -> {os.path.relpath(path, ROOT_DIR)}")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="paper_figure", description=__doc__)
    parser.add_argument("--model", default="logreg", choices=("logreg", "rf"))
    parser.add_argument("--folds", type=int, default=5, help="k the cached scores were run at")
    args = parser.parse_args(argv)
    curves = load_curves(args.folds)
    plot(curves, args.model, HERE / f"separability_paper__{args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
