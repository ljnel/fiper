"""Within-episode vs pooled pairwise distances between FIPER observation embeddings."""

from __future__ import annotations

import argparse
import os
import pathlib
import pickle
import sys
import textwrap
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT_DIR = str(HERE.parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

CACHE_DIR = HERE / "cache"
DATA_DIR = pathlib.Path(ROOT_DIR) / "data"

ALL_TASKS = ["push_t", "pretzel", "push_chair", "sorting", "stacking"]
#: The pair populations that are compared, narrowest first; "pooled" ignores episode identity.
SETS = ("adjacent", "within", "pooled")
#: Pairs drawn per set per task; below this the population is enumerated exactly instead.
MAX_PAIRS = 500_000
#: Rows per distance block, bounding the [block, dim] difference intermediate.
_BLOCK = 100_000
QUANTILES = (0.01, 0.25, 0.5, 0.75, 0.99)


# ---------------------------------------------------------------------------- the data
def load_task(task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observation embeddings and the per-episode start index and length of one task."""
    root = DATA_DIR / task / "processed_rollouts"
    with open(root / "metadata.pkl", "rb") as f:
        md = pickle.load(f)
    X = torch.load(root / "obs_embeddings.pt", weights_only=True).numpy().astype(np.float64)
    start = np.asarray(md["episode_start_indices"], dtype=np.int64)
    length = np.asarray(md["episode_end_indices"], dtype=np.int64) - start
    assert start[-1] + length[-1] <= len(X), f"{task}: episode indices overrun the embeddings"
    return X, start, length


# ------------------------------------------------------------------------- the sampling
def adjacent_pairs(start: np.ndarray,
                   length: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Index pairs of consecutive steps of the same episode; always the whole population."""
    a = np.concatenate([np.arange(s, s + L - 1) for s, L in zip(start, length) if L >= 2])
    return a, a + 1, len(a), True


def within_pairs(start: np.ndarray, length: np.ndarray, n_max: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Index pairs of steps from the same episode: every pair, or ``n_max`` drawn uniformly."""
    per_ep = length * (length - 1) // 2
    total = int(per_ep.sum())
    if total <= n_max:
        a, b = [], []
        for s, L in zip(start[length >= 2], length[length >= 2]):
            i, j = np.triu_indices(int(L), k=1)
            a.append(s + i)
            b.append(s + j)
        return np.concatenate(a), np.concatenate(b), total, True
    # Episode drawn with probability proportional to its pair count, so the sample is
    # uniform over the pair population rather than over episodes.
    ep = rng.choice(len(start), size=n_max, p=per_ep / total)
    L = length[ep]
    i = (rng.random(n_max) * L).astype(np.int64)
    j = (rng.random(n_max) * L).astype(np.int64)
    while (bad := i == j).any():  # rejection; hits ~1/L of the draws
        j[bad] = (rng.random(int(bad.sum())) * L[bad]).astype(np.int64)
    return start[ep] + i, start[ep] + j, total, False


def pooled_pairs(n_steps: int, n_max: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Index pairs of distinct steps anywhere in the task: every pair, or ``n_max`` drawn."""
    total = n_steps * (n_steps - 1) // 2
    if total <= n_max:
        i, j = np.triu_indices(n_steps, k=1)
        return i, j, total, True
    a = rng.integers(0, n_steps, size=n_max)
    b = rng.integers(0, n_steps, size=n_max)
    while (bad := a == b).any():
        b[bad] = rng.integers(0, n_steps, size=int(bad.sum()))
    return a, b, total, False


def distances(X: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance of each index pair, in blocks."""
    out = np.empty(len(a), dtype=np.float64)
    for s in range(0, len(a), _BLOCK):
        d = X[a[s : s + _BLOCK]] - X[b[s : s + _BLOCK]]
        out[s : s + _BLOCK] = np.sqrt(np.einsum("ij,ij->i", d, d))
    return out


def run_task(task: str, n_max: int, seed: int, force: bool) -> dict:
    """Sampled within-episode and pooled distances of one task, cached to npz."""
    path = CACHE_DIR / f"{task}__dists__n{n_max}__seed{seed}.npz"
    if path.exists() and not force:
        with np.load(path) as z:
            out = ({k: (z[k] if z[k].ndim else z[k].item()) for k in z.files}
                   if set(SETS) <= set(z.files) else None)  # a cache from a narrower SETS
        if out is not None:
            print(f"  {task:<11} cached <- {os.path.relpath(path, ROOT_DIR)}")
            return out

    started = time.time()
    X, start, length = load_task(task)
    rng = np.random.default_rng(seed)
    out = {"dim": X.shape[1], "n_steps": len(X), "n_episodes": len(start),
           "n_unique": len(np.unique(X, axis=0)),
           "ep_len_min": int(length.min()), "ep_len_med": int(np.median(length)),
           "ep_len_max": int(length.max())}
    for name, (a, b, total, exact) in (
        ("adjacent", adjacent_pairs(start, length)),
        ("within", within_pairs(start, length, n_max, rng)),
        ("pooled", pooled_pairs(len(X), n_max, rng)),
    ):
        out[name] = distances(X, a, b)
        out[f"{name}_total"] = total
        out[f"{name}_exact"] = exact
        # Cross-episode pairs are the pooled sample minus its same-episode part; the episode
        # of a step is the number of starts at or below it.
        if name == "pooled":
            ep = np.searchsorted(start, a, side="right")
            out["cross"] = out["pooled"][ep != np.searchsorted(start, b, side="right")]
            out["cross_total"] = total - int((length * (length - 1) // 2).sum())
            out["cross_exact"] = exact
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"  {task:<11} {time.time() - started:.1f}s -> {os.path.relpath(path, ROOT_DIR)}")
    return out


# ------------------------------------------------------------------------ the reporting
def summarise(task: str, res: dict, sets: tuple[str, ...]) -> pd.DataFrame:
    """One row per (task, pair set) of pair counts and distance summary statistics."""
    rows = []
    for name in sets:
        d = res[name]
        q = np.quantile(d, QUANTILES)
        rows.append({"task": task, "dim": int(res["dim"]), "steps": int(res["n_steps"]),
                     "episodes": int(res["n_episodes"]), "pairs": name,
                     "n_pairs_total": int(res[f"{name}_total"]), "n_pairs_used": len(d),
                     "exact": bool(res[f"{name}_exact"]), "mean": d.mean(), "std": d.std(),
                     "min": d.min(), **{f"q{int(p * 100):02d}": v for p, v in zip(QUANTILES, q)},
                     "max": d.max()})
    return pd.DataFrame(rows)


def report(table: pd.DataFrame, results: dict, sets: tuple[str, ...]) -> pd.DataFrame:
    """Print the per-task shapes, the summary table and each set's ratios to pooled."""
    print("\n=== embeddings and episodes")
    tasks = list(table.task.unique())
    print(table.drop_duplicates("task").assign(
        ep_len=[f"{results[t]['ep_len_min']}/{results[t]['ep_len_med']}/{results[t]['ep_len_max']}"
                for t in tasks],
        # Repeated embedding rows put an atom at distance 0 and pull the within-episode
        # summaries down, so they are reported next to the shapes.
        dups=[int(results[t]["n_steps"]) - int(results[t]["n_unique"]) for t in tasks],
    )[["task", "dim", "steps", "episodes", "ep_len", "dups"]].rename(
        columns={"dim": "embed_dim", "ep_len": "ep_len min/med/max",
                 "dups": "duplicate steps"}).to_string(index=False))

    print("\n=== pairwise Euclidean distances between observation embeddings")
    show = table.drop(columns=["steps", "episodes"]).copy()
    show["n_pairs_total"] = show.n_pairs_total.map(lambda v: f"{v:,}")
    show["n_pairs_used"] = show.n_pairs_used.map(lambda v: f"{v:,}")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    rows = []
    for task in tasks:
        res = results[task]
        pooled, q01 = res["pooled"], np.quantile(results[task]["pooled"], 0.01)
        for name in sets:
            d = res[name]
            rows.append({"task": task, "embed_dim": int(res["dim"]), "pairs": name,
                         "median": np.median(d), "mean": d.mean(),
                         "med/pooled": np.median(d) / np.median(pooled),
                         "mean/pooled": d.mean() / pooled.mean(),
                         # Share of the set below the pooled 1st percentile: how much of its
                         # mass sits where pooled pairs essentially never land.
                         "P(< pooled q01)": float((d < q01).mean()),
                         "P(= 0)": float((d == 0).mean())})
    ratio = pd.DataFrame(rows)
    print("\n=== each set relative to the pooled distribution")
    print(ratio.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return ratio


# ---------------------------------------------------------------------------- the plots
#: Chart chrome and the three series hues (dataviz reference palette, light mode: slots 1-3,
#: worst all-pairs CVD dE 9.2 / normal-vision 24.0; aqua is under 3:1 on this surface, which
#: the direct median labels and the printed table relieve).
_SURFACE, _INK, _SECOND, _MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
_GRID, _AXIS = "#e1e0d9", "#c3c2b7"
_BLUE, _ORANGE, _AQUA = "#2a78d6", "#eb6834", "#1baf7a"
_SERIES = (("adjacent", "consecutive steps", _AQUA, "-."),
           ("within", "within episode", _BLUE, "-"),
           ("pooled", "pooled over all episodes", _ORANGE, "--"))
#: Surface patch behind in-panel text, so a grid line never runs through a label.
_LABEL_BOX = dict(facecolor=_SURFACE, edgecolor="none", pad=1.5)


def plot_distributions(results: dict, tasks: list[str], bins: int, stem: pathlib.Path) -> None:
    """Density of both pair populations per task, one panel each, with median rules."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(len(tasks), 1, figsize=(8.0, 13.0))
    for ax, task in zip(np.atleast_1d(axes), tasks):
        res = results[task]
        hi = max(np.quantile(res[c], 0.999) for c, *_ in _SERIES)
        edges = np.linspace(0.0, hi, bins + 1)
        ax.set_facecolor(_SURFACE)
        ax.set_title(f"{task}   (embedding dim {int(res['dim'])}, {int(res['n_episodes'])} "
                     f"episodes, {int(res['n_steps'])} steps)", color=_INK, fontsize=11,
                     loc="left", pad=20)
        med = {c: float(np.median(res[c])) for c, *_ in _SERIES}
        ax.annotate("medians as a share of pooled:   "
                    + ",   ".join(f"{lab} {med[c] / med['pooled']:.0%}"
                                  for c, lab, *_ in _SERIES if c != "pooled"),
                    (0.0, 1.015), xycoords="axes fraction", va="bottom", ha="left",
                    color=_MUTED, fontsize=8.5)
        ax.grid(True, axis="y", color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_AXIS)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=_MUTED, labelsize=9)
        ax.set_ylabel("density", color=_SECOND, fontsize=9.5)
        ax.set_xlim(0.0, hi)

        heights = {}
        for col, label, color, dash in _SERIES:
            h, _ = np.histogram(res[col], bins=edges, density=True)
            ax.stairs(h, edges, color=color, linewidth=2.0, linestyle=dash, label=label)
            ax.stairs(h, edges, color=color, alpha=0.16, fill=True, linewidth=0)
            heights[col] = h
        # A task with duplicate embeddings has an atom at distance 0, whose density is set by
        # the bin width alone; the first bin is left out of the scaling and clipped instead.
        peak = max(h[1:].max() for h in heights.values())
        ax.set_ylim(0.0, peak * 1.42)
        clipped = [(lab, float((res[col] == 0).mean())) for col, lab, *_ in _SERIES
                   if heights[col][0] > peak * 1.42]
        if clipped:
            ax.annotate("first bin runs off the panel:   "
                        + ",   ".join(f"{lab} {p:.0%} at distance 0" for lab, p in clipped),
                        (0.985, 0.58), xycoords="axes fraction", ha="right", color=_MUTED,
                        fontsize=8.5, bbox=_LABEL_BOX)
        # Median rules carry the comparison the densities only imply; staggered heights so the
        # three labels cannot collide when the distributions overlap.
        for (col, label, color, _), y in zip(_SERIES, (1.30, 1.16, 1.02)):
            ax.plot([med[col], med[col]], [0.0, peak * (y - 0.05)], color=color, linewidth=1.2,
                    linestyle=":")
            ax.annotate(f"median {med[col]:.3g}", (med[col], peak * y), xytext=(4, 0),
                        va="center", textcoords="offset points", color=color, fontsize=8.5,
                        bbox=_LABEL_BOX)
        if task == tasks[-1]:  # one x-label for the stack; the quantity is the same in every panel
            ax.set_xlabel("pairwise Euclidean distance between observation embeddings",
                          color=_SECOND, fontsize=10)

    handles = [Line2D([], [], color=c, linestyle=d, linewidth=2.0, label=lab)
               for _, lab, c, d in _SERIES]
    _finish(fig, "How far apart are observation embeddings within an episode, and overall?",
            "Density of pairwise Euclidean distances over three nested pair populations: aqua = "
            "consecutive steps of an episode, blue = any two steps of the same episode, orange = "
            "any two steps of the task with episode identity ignored. Each panel has its own "
            "x-axis -- the embedding dimension and scale differ per task -- so compare the "
            "curves within a panel, not across panels. The x range is clipped at the 99.9th "
            "percentile of the widest population.", stem, handles)


def _finish(fig, title: str, subtitle: str, stem: pathlib.Path, handles) -> None:
    """Header block (title, wrapped subtitle, legend), spacing, and the png/pdf pair."""
    import matplotlib.pyplot as plt

    # Header laid out in points from the top, so no number here depends on the figure height.
    height = fig.get_figheight() * 72.0
    lines = textwrap.wrap(subtitle, width=int(fig.get_figwidth() * 72.0 / 5.1))
    y = 1.0 - 26.0 / height
    fig.text(0.055, y, title, color=_INK, fontsize=13.5, ha="left", va="baseline")
    y -= 20.0 / height
    fig.text(0.055, y, "\n".join(lines), color=_SECOND, fontsize=9.0, ha="left", va="top",
             linespacing=1.45)
    y -= (13.0 * len(lines) + 6.0) / height
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, y), ncol=len(handles),
               frameon=False, fontsize=9, labelcolor=_SECOND, handlelength=2.6)
    y -= 20.0 / height
    fig.tight_layout(rect=(0.0, 0.0, 1.0, y - 6.0 / height))
    fig.subplots_adjust(hspace=0.48)
    for ext in ("png", "pdf"):
        path = stem.with_suffix("." + ext)
        fig.savefig(path, dpi=150, facecolor=_SURFACE)
        print(f"  -> {os.path.relpath(path, ROOT_DIR)}")
    plt.close(fig)


# ----------------------------------------------------------------------------- the main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="embedding_distances", description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    parser.add_argument("--pairs", type=int, default=MAX_PAIRS, help="pairs per set per task")
    parser.add_argument("--bins", type=int, default=80, help="histogram bins per panel")
    parser.add_argument("--cross", action="store_true",
                        help="add the cross-episode-only pairs as a third row per task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="ignore the cache")
    args = parser.parse_args(argv)

    started = time.time()
    sets = SETS + ("cross",) if args.cross else SETS
    print(f"up to {args.pairs:,} pairs per set per task, seed {args.seed}, sets {sets}\n"
          f"=== distances")
    tasks = [t for t in ALL_TASKS if t in args.tasks]
    results = {t: run_task(t, args.pairs, args.seed, args.force) for t in tasks}

    table = pd.concat([summarise(t, results[t], sets) for t in tasks], ignore_index=True)
    ratio = report(table, results, sets)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for df, name in ((table, "summary"), (ratio, "ratios")):
        path = HERE / f"embedding_distances__{name}.csv"
        df.to_csv(path, index=False)
        print(f"  -> {os.path.relpath(path, ROOT_DIR)}")

    print("\n=== plot")
    plot_distributions(results, tasks, args.bins, HERE / "embedding_distances")
    print(f"\ntotal wall clock: {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
