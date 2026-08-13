"""Double-column paper figure for the data-efficiency plot of ``efficiency.py``, transposed to tasks-as-columns and re-plotted from the cached summary table."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from efficiency import (ALL_TASKS, GRID, INK, LABEL, Q2_METHODS, ROOT_DIR, SECOND,  # noqa: E402
                        TASK_LABEL, method_style)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator, NullFormatter  # noqa: E402

SUMMARY = HERE / "efficiency__summary.csv"

# Type 42 rather than matplotlib's default Type 3, which several venues reject.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5})


def load_summary() -> pd.DataFrame:
    """The tidy per-point table written by the last ``efficiency.py --plot`` run."""
    if not SUMMARY.exists():
        raise SystemExit(f"no {SUMMARY.name}; run efficiency.py --plot first")
    df = pd.read_csv(SUMMARY, usecols=["task", "method", "train_episodes", "TWA"])
    print(f"{len(df)} rows <- {os.path.relpath(SUMMARY, ROOT_DIR)}")
    return df


def plot(df: pd.DataFrame, stem: pathlib.Path) -> None:
    """One panel per task, TWA against the size of the calibration split."""
    styles = method_style()
    methods = [m for m in Q2_METHODS if m in set(df.method)]
    tasks = [t for t in ALL_TASKS if t in set(df.task)]

    fig, axes = plt.subplots(1, len(tasks), figsize=(7.0, 1.75))
    for ax, task in zip(axes, tasks):
        g = df[df.task == task]
        for method in methods:
            h = g[g.method == method].sort_values("train_episodes")
            ax.plot(h.train_episodes, h.TWA, linewidth=1.1, markersize=2.6,
                    markeredgecolor="white", markeredgewidth=0.3, **styles[method])
        ax.set_title(TASK_LABEL[task], color=INK, fontsize=7.5, pad=3)
        ax.grid(color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.margins(x=0.07, y=0.14)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=SECOND, labelsize=6.5, pad=1.5)

        # Per-column x: the calibration splits differ by task (2-10 real, 5-50 simulated).
        # Every point gets a tick, but only three get a label -- five will not fit in 1.3 in.
        ticks = sorted(g.train_episodes.unique())
        shown = {ticks[0], ticks[len(ticks) // 2], ticks[-1]}
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([str(t) if t in shown else "" for t in ticks]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", length=0)
        # Per-column y as well: the TWA ranges span 0.06 (Push-T) to 0.11 (Push-Chair) at
        # levels 0.30 apart, so one shared scale would flatten every trend the panel is for.
        ax.yaxis.set_major_locator(MaxNLocator(4))

    axes[0].set_ylabel("TWA", color=INK, fontsize=7.5, labelpad=2)
    fig.legend(handles=[Line2D([], [], linewidth=1.1, markersize=2.6, label=LABEL[m], **styles[m])
                        for m in methods],
               loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=len(methods), frameon=False,
               fontsize=7, labelcolor=SECOND, columnspacing=1.8, handletextpad=0.4)
    fig.text(0.5, 0.045, "training episodes", color=INK, fontsize=7.5, ha="center")
    fig.subplots_adjust(left=0.052, right=0.99, top=0.775, bottom=0.235, wspace=0.32)

    for ext in ("pdf", "png"):
        path = stem.with_suffix("." + ext)
        fig.savefig(path, dpi=400)
        print(f"  -> {os.path.relpath(path, ROOT_DIR)}")
    plt.close(fig)


def main(argv=None) -> int:
    argparse.ArgumentParser(prog="paper_figure", description=__doc__).parse_args(argv)
    plot(load_summary(), HERE / "efficiency_data_paper")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
