"""How separable are FIPER's test episodes from its nominal data? -- ``specs/experiments/separability.md``.

The question
------------
Our kern_cd family (``specs/methods.md``) barely moves the FIPER numbers. That is either a
weakness of the methods or a property of the benchmark. This script measures the second:
it hands a classifier the success/failure labels that the benchmark withholds, and asks how
well an *oracle* two-sample test can tell FIPER's nominal (calibration) data from its test
episodes. FIPER itself is the one-class version of exactly this problem, so a supervised
AUROC near chance is a ceiling on what any one-class method could achieve.

Two comparisons per task, both against the same nominal pool:

* nominal vs **test success** episodes (green) -- the distribution shift the benchmark does
  *not* want flagged (policy rollouts that worked, drawn from a later/held-out pass).
* nominal vs **test failure** episodes (red) -- the shift it *does* want flagged.

The gap between the two lines is the signal available to a failure detector. Where the lines
sit on top of each other, no score computed from observation embeddings can separate failure
from success, however it is calibrated -- and where both sit near 1.0, the classifier is
picking up a nominal-vs-test shift that has nothing to do with failure.

Design
------
* **Features**: observation embeddings only, raw (``configs/eval/base.yaml`` sets
  ``normalize_tensors.obs_embeddings: False``, and ``specs/methods.md`` keeps that default).
* **Time axis**: normalized time ``tau = (i+1)/T`` within an episode. At each grid point
  ``t`` the model trains on the *pool of all steps with ``tau <= t``* -- steps, not episodes,
  so that the strong episode-length confound (failures run ~2x longer) cannot leak in
  through the label. ``--binned`` switches the pool to just the steps inside ``(t-dt, t]``,
  which trades sample size for a sharper read on where the two curves diverge.
* **Score**: AUROC of a random forest, cross-validated **over episodes**
  (``StratifiedGroupKFold`` on the episode id), so no fold ever scores a step whose
  near-duplicate neighbours it was trained on. The plotted value is the AUROC of the pooled
  out-of-fold probabilities; the band is +/-1 std of the per-fold AUROCs.
* **Balance**: at each ``(task, t)`` all pools -- nominal, success, failure -- are subsampled
  to one common size, so the green and red numbers are computed at the same n and the same
  50/50 class balance and are directly comparable to each other and across time.

Push T's calibration split contains 29 failed rollouts out of 50 and they are kept here (the
spec says to ignore that contamination); the printed report quotes the count so its effect on
push_t can be judged.

Cost
----
Observation embeddings only, so no action tensors are built and the whole dataset for the
biggest task loads in ~3 s. The pools are capped at ``--max-per-class`` steps, which bounds
every forest fit regardless of task size.

Running it
----------
    python my_experiments/separability/separability.py                 # all 5 tasks -> separability.{png,pdf}
    python my_experiments/separability/separability.py --tasks push_t  # one task
    python my_experiments/separability/separability.py --binned        # per-bin pools instead of cumulative
    python my_experiments/separability/separability.py --null-control  # + nominal-vs-nominal chance check
    python my_experiments/separability/separability.py --plot-only     # redraw from the cache

Per-task grids are cached as CSVs under ``my_experiments/separability/cache/`` and reused;
``--force`` re-runs. Nothing outside ``my_experiments/`` is written.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT_DIR = str(HERE.parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from my_methods.run import ALL_TASKS  # noqa: E402
from shared_utils.hydra_utils import load_config  # noqa: E402
from tasks import TaskManager  # noqa: E402

CACHE_DIR = HERE / "cache"
OUT_STEM = HERE / "separability"

#: The two comparisons, in plot order. Key -> (label, colour, linestyle, marker).
#: Green/red is the spec's choice and is a status encoding (success/failure), but the pair
#: sits in the 6-8 CVD floor band, so identity is carried a second time by the line style
#: and the marker as well as by the legend.
SERIES = {
    "success": ("test success vs nominal", "#116329", "-", "o"),
    "failure": ("test failure vs nominal", "#e5484d", "--", "^"),
}

#: Below this many steps per class the AUROC is noise, so it is reported as NaN and left
#: out of the line rather than drawn as a wild excursion. push_chair (3-7 step episodes,
#: 49 nominal steps in total) is the task this exists for.
MIN_PER_CLASS = 25


# --------------------------------------------------------------------------------- data
def load_task(task: str):
    """Raw observation embeddings + the episode index for one task.

    Returns ``(X, starts, ends, calib_mask, test_mask, success_mask)`` where ``X`` is the
    ``[N_steps, d_obs]`` float32 matrix of every step in the task, in dataset order, and the
    masks are per *episode*. Reading ``ds.data`` directly rather than going through
    ``get_subset`` keeps the row indices global, which is what the pooling below needs.
    """
    cfg = load_config("task", task, return_only_subdict=False)
    manager = TaskManager(
        cfg,
        task,
        os.path.join(ROOT_DIR, "configs"),
        os.path.join(ROOT_DIR, "data", task),
        required_tensors=["obs_embeddings"],
        optional_tensors=[],
        device="cpu",
    )
    # load_dataset_if_exists=False for the reason given in my_methods/run.py: the cached
    # processed_rollouts/*.pt load path is broken upstream. Re-converting is ~0.1-4 s here,
    # since obs_embeddings is the only tensor requested.
    ds = manager.get_rollout_dataset(load_dataset_if_exists=False)
    md = ds.data["metadata"]
    X = np.asarray(ds.data["obs_embeddings"], dtype=np.float32)
    return (
        X,
        np.asarray(md["episode_start_indices"], dtype=int),
        np.asarray(md["episode_end_indices"], dtype=int),
        np.asarray(md["calibration_rollout_labels"], dtype=bool),
        np.asarray(md["test_rollout_labels"], dtype=bool),
        np.asarray(md["successful_rollout_labels"], dtype=bool),
    )


def step_table(starts: np.ndarray, ends: np.ndarray, keep: np.ndarray) -> dict:
    """Flatten the episodes selected by ``keep`` into a per-step table.

    Returns ``{"row", "ep", "tau", "lengths", "ep_ids"}``: the global row index of each step,
    the episode it belongs to (the CV group), its normalized time ``(i+1)/T``, and the
    episode lengths. One entry per step, in dataset order.
    """
    ep_ids = np.where(keep)[0]
    rows, eps, taus, lengths = [], [], [], []
    for e in ep_ids:
        s, f = int(starts[e]), int(ends[e])
        n = f - s
        if n <= 0:
            continue
        rows.append(np.arange(s, f))
        eps.append(np.full(n, e))
        taus.append((np.arange(n) + 1) / n)
        lengths.append(n)
    if not rows:
        return {"row": np.zeros(0, int), "ep": np.zeros(0, int), "tau": np.zeros(0),
                "lengths": np.zeros(0, int), "ep_ids": ep_ids}
    return {
        "row": np.concatenate(rows),
        "ep": np.concatenate(eps),
        "tau": np.concatenate(taus),
        "lengths": np.asarray(lengths, dtype=int),
        "ep_ids": ep_ids,
    }


def pool_mask(tab: dict, t: float, dt: float, binned: bool) -> np.ndarray:
    """Steps of ``tab`` that belong to the pool at normalized time ``t``."""
    if binned:
        return (tab["tau"] > t - dt) & (tab["tau"] <= t + 1e-12)
    return tab["tau"] <= t + 1e-12


# ---------------------------------------------------------------------------- the score
def cv_auroc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int, trees: int,
             seed: int) -> tuple[float, float, float, int]:
    """Grouped-CV AUROC of a random forest.

    Returns ``(auroc_oof, fold_mean, fold_std, n_splits)``. ``auroc_oof`` -- the AUROC of the
    concatenated out-of-fold probabilities -- is the plotted number; it uses every point once
    and does not degenerate when a fold happens to hold few episodes. The per-fold spread is
    kept for the uncertainty band.

    The split is ``StratifiedGroupKFold`` over episode ids: consecutive steps of one episode
    are near-duplicates, so a step-level split would let the model memorise a neighbour of
    every test point and report a separability that is not there.
    """
    n_splits = int(min(folds, np.unique(groups[y == 0]).size, np.unique(groups[y == 1]).size))
    if n_splits < 2:
        return np.nan, np.nan, np.nan, n_splits

    oof = np.full(len(y), np.nan)
    per_fold = []
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(splitter.split(X, y, groups)):
        clf = RandomForestClassifier(
            n_estimators=trees,
            n_jobs=-1,
            random_state=seed + k,
            class_weight="balanced",
        )
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        if np.unique(y[te]).size == 2:
            per_fold.append(roc_auc_score(y[te], oof[te]))

    auroc = roc_auc_score(y, oof) if np.unique(y).size == 2 else np.nan
    fold_mean = float(np.mean(per_fold)) if per_fold else np.nan
    fold_std = float(np.std(per_fold)) if len(per_fold) > 1 else np.nan
    return float(auroc), fold_mean, fold_std, n_splits


def balanced_draw(tabs: list[dict], masks: list[np.ndarray], cap: int,
                  rng: np.random.Generator) -> tuple[list[np.ndarray], int]:
    """Subsample every pool to one common size, drawing steps (not whole episodes).

    A common ``n`` across the nominal, success and failure pools is what makes the two
    AUROCs comparable: same sample size, same 50/50 balance, so a difference between the
    green and the red curve cannot be an artefact of one pool being larger. Steps are drawn
    individually so the surviving sample still spans as many episodes as possible, which the
    grouped CV needs.
    """
    n = min(cap, *[int(m.sum()) for m in masks])
    out = []
    for tab, mask in zip(tabs, masks):
        idx = np.where(mask)[0]
        if len(idx) > n:
            idx = rng.choice(idx, n, replace=False)
        out.append(idx)
    return out, n


def two_sample(X: np.ndarray, a_tab: dict, a_idx: np.ndarray, b_tab: dict, b_idx: np.ndarray,
               folds: int, trees: int, seed: int) -> dict:
    """One grouped-CV AUROC for pool ``a`` (class 0) against pool ``b`` (class 1)."""
    rows = np.concatenate([a_tab["row"][a_idx], b_tab["row"][b_idx]])
    groups = np.concatenate([a_tab["ep"][a_idx], b_tab["ep"][b_idx]])
    y = np.concatenate([np.zeros(len(a_idx), int), np.ones(len(b_idx), int)])
    auroc, fold_mean, fold_std, n_splits = cv_auroc(X[rows], y, groups, folds, trees, seed)
    return {
        "auroc": auroc,
        "fold_mean": fold_mean,
        "fold_std": fold_std,
        "n_splits": n_splits,
        "n_per_class": len(a_idx),
        "n_ep_0": int(np.unique(groups[y == 0]).size),
        "n_ep_1": int(np.unique(groups[y == 1]).size),
    }


# ----------------------------------------------------------------------------- the sweep
def run_task(task: str, args) -> pd.DataFrame:
    """The full (time x comparison) AUROC grid for one task, with the shape report."""
    t0 = time.time()
    X, starts, ends, calib, test, success = load_task(task)
    tabs = {
        "nominal": step_table(starts, ends, calib),
        "success": step_table(starts, ends, test & success),
        "failure": step_table(starts, ends, test & ~success),
    }
    report_task(task, X, tabs, calib, success, time.time() - t0)

    grid = np.round(np.linspace(1.0 / args.grid, 1.0, args.grid), 6)
    dt = 1.0 / args.grid
    rng = np.random.default_rng(args.seed)

    rows = []
    for t in grid:
        masks = {k: pool_mask(tab, float(t), dt, args.binned) for k, tab in tabs.items()}
        (i_nom, i_suc, i_fail), n = balanced_draw(
            [tabs["nominal"], tabs["success"], tabs["failure"]],
            [masks["nominal"], masks["success"], masks["failure"]],
            args.max_per_class,
            rng,
        )
        for key, idx in (("success", i_suc), ("failure", i_fail)):
            row = {"Task": task, "t": float(t), "comparison": key,
                   "n_pool_nominal": int(masks["nominal"].sum()),
                   "n_pool_other": int(masks[key].sum())}
            if n < MIN_PER_CLASS:
                row.update(auroc=np.nan, fold_mean=np.nan, fold_std=np.nan, n_splits=0,
                           n_per_class=n, n_ep_0=0, n_ep_1=0)
            else:
                row.update(two_sample(X, tabs["nominal"], i_nom, tabs[key], idx,
                                      args.folds, args.trees, args.seed))
            rows.append(row)

        if args.null_control:
            rows.append(null_control(task, float(t), X, tabs["nominal"], i_nom, args))

    df = pd.DataFrame(rows)
    report_grid(task, df)
    return df


def null_control(task: str, t: float, X: np.ndarray, tab: dict, idx: np.ndarray, args) -> dict:
    """Nominal against nominal: the same pipeline where the answer must be chance.

    The nominal episodes are split in half *by episode* and relabelled. Anything the grouped
    CV reports above 0.5 here is pipeline bias (fold imbalance, forest overfitting a small
    pool), and is the level against which the two real curves should be read.
    """
    eps = np.unique(tab["ep"][idx])
    rng = np.random.default_rng(args.seed + 1)
    left = rng.choice(eps, len(eps) // 2, replace=False) if len(eps) > 1 else eps
    is_left = np.isin(tab["ep"][idx], left)
    a, b = idx[is_left], idx[~is_left]
    n = min(len(a), len(b))
    row = {"Task": task, "t": t, "comparison": "null_ctrl", "n_pool_nominal": len(a),
           "n_pool_other": len(b)}
    if n < MIN_PER_CLASS:
        row.update(auroc=np.nan, fold_mean=np.nan, fold_std=np.nan, n_splits=0,
                   n_per_class=n, n_ep_0=0, n_ep_1=0)
    else:
        row.update(two_sample(X, tab, a[:n], tab, b[:n], args.folds, args.trees, args.seed))
    return row


# --------------------------------------------------------------------------- the reports
def report_task(task: str, X: np.ndarray, tabs: dict, calib: np.ndarray, success: np.ndarray,
                load_s: float) -> None:
    """Shapes and summary stats of everything the sweep reads."""
    print(f"\n{'=' * 78}\n=== {task}  (dataset built in {load_s:.1f} s)\n{'=' * 78}")
    print(f"  obs_embeddings X                 {X.shape}  {X.dtype}   "
          f"(raw: eval/base.yaml sets normalize_tensors.obs_embeddings=False)")
    print(f"    per-element   mean {X.mean():+.4f}  std {X.std():.4f}  "
          f"min {X.min():+.4f}  max {X.max():+.4f}")
    norms = np.linalg.norm(X, axis=1)
    print(f"    row L2 norm   mean {norms.mean():.4f}  std {norms.std():.4f}  "
          f"[{norms.min():.4f}, {norms.max():.4f}]")
    n_fail_calib = int((calib & ~success).sum())
    if n_fail_calib:
        print(f"    NB {n_fail_calib}/{int(calib.sum())} calibration episodes are failures and are kept "
              "in the nominal pool (the spec says to ignore this contamination)")

    print(f"\n  {'pool':<10} {'episodes':>9} {'steps':>8} {'ep len (min/med/max)':>24}  "
          f"{'tau spacing (med)':>18}")
    for key, tab in tabs.items():
        L = tab["lengths"]
        if not len(L):
            print(f"  {key:<10} {0:>9} {0:>8}")
            continue
        print(f"  {key:<10} {len(L):>9} {len(tab['row']):>8} "
              f"{L.min():>8} /{int(np.median(L)):>5} /{L.max():>5}   {1 / np.median(L):>17.4f}")
    print("  (episode length is the success/failure confound the spec warns about -- it is "
          "why the pools are\n   built from steps and why every pool is subsampled to a "
          "common size at each t)")


def report_grid(task: str, df: pd.DataFrame) -> None:
    """The AUROC grid as a table, plus the success/failure gap that answers the question."""
    print(f"\n  --- AUROC grid, {task} ---")
    show = df.copy()
    show["auroc"] = show["auroc"].round(4)
    show["fold_mean"] = show["fold_mean"].round(4)
    show["fold_std"] = show["fold_std"].round(4)
    print(show.to_string(index=False, na_rep="  --  "))

    wide = df.pivot_table(index="t", columns="comparison", values="auroc")
    if {"success", "failure"} <= set(wide.columns):
        wide["gap (fail - succ)"] = wide["failure"] - wide["success"]
        print(f"\n  --- {task}: how much of the separability is actually about failure? ---")
        print(wide.round(4).to_string(na_rep="  --  "))
        gap = wide["gap (fail - succ)"].dropna()
        if len(gap):
            print(f"  gap: mean {gap.mean():+.4f}  max {gap.max():+.4f} at t={gap.idxmax():.2f}  "
                  f"min {gap.min():+.4f} at t={gap.idxmin():.2f}")


def report_summary(df: pd.DataFrame) -> None:
    """One block that puts every task side by side, which is what the plot shows."""
    print(f"\n\n{'=' * 78}\n=== summary: AUROC vs normalized time\n{'=' * 78}")
    for key in [k for k in ("success", "failure", "null_ctrl") if k in set(df["comparison"])]:
        wide = df[df["comparison"] == key].pivot_table(index="Task", columns="t", values="auroc")
        wide = wide.reindex([t for t in ALL_TASKS if t in wide.index])
        print(f"\n  -- {key} vs nominal --")
        print(wide.round(3).to_string(na_rep=" -- "))

    if {"success", "failure"} <= set(df["comparison"]):
        piv = df.pivot_table(index=["Task", "t"], columns="comparison", values="auroc")
        piv["gap"] = piv["failure"] - piv["success"]
        gaps = piv.groupby("Task")["gap"].agg(["mean", "max"])
        late = piv.reset_index()
        late = late[late["t"] >= 0.75].groupby("Task")["gap"].mean().rename("mean (t>=0.75)")
        out = gaps.join(late).reindex([t for t in ALL_TASKS if t in gaps.index])
        print("\n  -- failure-minus-success AUROC gap (the separability actually available "
              "to a failure detector) --")
        print(out.round(4).to_string(na_rep=" -- "))


# ------------------------------------------------------------------------------ the plot
def plot(df: pd.DataFrame, tasks: list[str], binned: bool, stem: pathlib.Path) -> None:
    """Five stacked panels, one per task: AUROC against normalized time."""
    tasks = [t for t in tasks if t in set(df["Task"])]
    fig, axes = plt.subplots(len(tasks), 1, figsize=(7.2, 2.15 * len(tasks) + 1.2),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    handles: list = []

    lo = min(0.45, float(np.nanmin(df["auroc"])) - 0.03) if df["auroc"].notna().any() else 0.4
    for ax, task in zip(axes, tasks):
        sub = df[df["Task"] == task]
        ax.axhline(0.5, color="#8c8c88", lw=1.0, ls=":", zorder=1)
        for key, (label, colour, ls, marker) in SERIES.items():
            s = sub[sub["comparison"] == key].sort_values("t")
            if s.empty:
                continue
            band = s["fold_std"].to_numpy()
            ax.fill_between(s["t"], s["auroc"] - band, s["auroc"] + band,
                            color=colour, alpha=0.13, lw=0, zorder=2)
            line, = ax.plot(s["t"], s["auroc"], color=colour, ls=ls, marker=marker, ms=5.0,
                            lw=2.0, mew=1.4, mec="white", label=label, zorder=3)
            if ax is axes[0]:
                handles.append(line)
        nul = sub[sub["comparison"] == "null_ctrl"].sort_values("t")
        if not nul.empty:
            line, = ax.plot(nul["t"], nul["auroc"], color="#8c8c88", ls="-", lw=1.4, marker="s",
                            ms=3.5, label="nominal vs nominal (null)", zorder=2)
            if ax is axes[0]:
                handles.append(line)

        ax.set_ylim(lo, 1.02)
        ax.set_ylabel("AUROC", fontsize=9)
        ax.set_title(task, fontsize=10, loc="left", pad=4)
        ax.grid(axis="y", color="#e6e6e2", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9c9c4")
        ax.tick_params(labelsize=8, color="#c9c9c4")

        # Sample size goes on the title line rather than inside the axes: with five panels
        # there is no corner that is free of a curve in all of them.
        counts = sub[sub["t"] == sub["t"].max()]
        if len(counts):
            r = counts.iloc[0]
            note = f"n = {int(r['n_per_class'])}/class at $t$=1, {int(r['n_splits'])} folds"
            if sub["auroc"].isna().any():
                first = sub.loc[sub["auroc"].notna(), "t"]
                note = (f"n < {MIN_PER_CLASS}/class for $t$ < {first.min():.1f}; " + note
                        if len(first) else f"n < {MIN_PER_CLASS}/class everywhere")
            ax.set_title(note, fontsize=7.5, loc="right", color="#6e6e69", pad=4)

    axes[-1].set_xlabel("normalized episode time  $t$   "
                        + ("(pool: steps in the bin ending at $t$)" if binned
                           else "(pool: all steps with $\\tau \\leq t$)"), fontsize=9)
    axes[-1].set_xlim(0.0, 1.03)
    # Bottom-outside: constrained_layout reserves space for an "outside" legend, and the
    # bottom strip is the only one that is not also claimed by the suptitle.
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), fontsize=8.5,
               frameon=False, handlelength=2.6, columnspacing=2.4)
    fig.suptitle("Oracle separability from nominal data: random forest on observation "
                 "embeddings,\ngrouped CV over episodes", fontsize=11)

    for ext in ("png", "pdf"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"  wrote {os.path.relpath(path, ROOT_DIR)}")
    plt.close(fig)


# -------------------------------------------------------------------------------- driver
def cache_path(task: str, args) -> pathlib.Path:
    tag = (f"grid{args.grid}_max{args.max_per_class}_k{args.folds}_trees{args.trees}"
           f"_seed{args.seed}{'_binned' if args.binned else ''}"
           f"{'_null' if args.null_control else ''}")
    return CACHE_DIR / f"{task}__{tag}.csv"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="separability", description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    parser.add_argument("--grid", type=int, default=10, help="number of normalized-time points")
    parser.add_argument("--max-per-class", type=int, default=2000,
                        help="cap on steps per class per fit; every pool is drawn to a common size")
    parser.add_argument("--folds", type=int, default=5, help="max grouped-CV folds")
    parser.add_argument("--trees", type=int, default=150, help="random forest size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--binned", action="store_true",
                        help="pool only the steps inside the bin ending at t, not all steps up to t")
    parser.add_argument("--null-control", action="store_true",
                        help="also run nominal-vs-nominal, which must come out at chance")
    parser.add_argument("--force", action="store_true", help="ignore the cache")
    parser.add_argument("--plot-only", action="store_true", help="redraw from the cache and exit")
    parser.add_argument("--out", default=str(OUT_STEM), help="output stem for the .png / .pdf")
    args = parser.parse_args(argv)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for task in args.tasks:
        path = cache_path(task, args)
        if path.exists() and not args.force:
            frames.append(pd.read_csv(path))
            print(f"[cache] {task} <- {os.path.relpath(path, ROOT_DIR)}")
            continue
        if args.plot_only:
            print(f"[skip] {task}: no cache at {os.path.relpath(path, ROOT_DIR)}", file=sys.stderr)
            continue
        df = run_task(task, args)
        df.to_csv(path, index=False)
        frames.append(df)

    if not frames:
        print("nothing to plot", file=sys.stderr)
        return 1

    df = pd.concat(frames, ignore_index=True)
    report_summary(df)
    print()
    plot(df, args.tasks, args.binned, pathlib.Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
