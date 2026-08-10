"""How separable is FIPER's data at all? -- ``specs/experiments/separability.md``.

An oracle two-sample test given the labels FIPER withholds. Calibration is the "nominal"
class; the test split is cut by its own success label into two positive classes, each
compared against nominal on its own:

    (Q1)  nominal vs test *success*      (Q2)  nominal vs test *failure*

A random forest scored by AUROC under 5-fold CV **over episodes** -- never over steps, since
consecutive steps of one rollout are near-duplicates and would leak. Figure 1 plots the
pooled out-of-fold AUROC per comparison. The success curve is the control: it absorbs any
calibration-vs-test shift unrelated to failing, so the meaningful quantity is the gap,
failure AUROC minus success AUROC.

At grid time ``t`` the model trains on the prefix pool of every step with normalized
position ``u = i/(T-1) <= t``. Episode length is a strong success/failure confound but
cannot leak, because a pooled step carries no information about its episode's length.

Q3 asks whether the action channels help, which is a change in the *gap*, not in the failure
AUROC alone. Figure 2 therefore plots one number per grid point, the difference-in-
differences ``(fail - succ | obs+act) - (fail - succ | obs)``, formed inside each fold and
then averaged, with a 95% Student-t CI on the DiD itself -- see :func:`did_curve`. The two
feature sets are:

* ``obs``     -- the observation embedding, ``[d_obs]``. All five tasks.
* ``obs+act`` -- that, concatenated with the flattened task-space (position, D=3) chunk
  batch, ``[d_obs + B*H*3]``. Restricted per the spec to the three tasks with a small chunk
  batch (sorting/stacking B=32, pretzel B=30); B=256 on push_t and push_chair would put
  12288 action features against 64 observation ones.

Both are scored in the same run on the same subsampled steps and the same folds, so an
AUROC difference between them is the features and nothing else. Neither block is scaled: a
forest splits one feature at a time, so it is invariant to per-feature rescaling.

Two things to know when reading figure 1. The success curve falls *below* 0.5 on stacking and
push_chair -- reproducibly, at a per-fold std of 0.03 over 4000-row pools. Sub-chance AUROC
under grouped CV means the nominal class is more heterogeneous episode-to-episode than it is
different from test successes, so a held-out nominal episode lands where no *training*
nominal episode did and gets called "test" more confidently than the held-out test episodes
do. Read it as "no separation that survives an unseen episode". And push_t's calibration
split is 29/50 failed rollouts, so its nominal class is contaminated; the spec says to accept
that, but it is why push_t is the least interpretable of the five.

Running it
----------
    pixi run python my_experiments/separability/separability.py             # everything
    pixi run python my_experiments/separability/separability.py --tasks pretzel push_chair
    pixi run python my_experiments/separability/separability.py --force     # ignore the cache
    pixi run python my_experiments/separability/separability.py --plot-only # just redraw

Per-task scores are cached under ``cache/`` and reused; the figures are written next to this
file as png and pdf. Nothing under ``data/`` is touched.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

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
from scipy.stats import t as student_t  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from datasets.rollout_datasets import ProcessedRolloutDataset  # noqa: E402

#: Plot / report order, matching ``my_methods.run.ALL_TASKS``.
TASKS = ["push_t", "pretzel", "push_chair", "sorting", "stacking"]

#: Tasks whose action chunk batch is small enough to concatenate (B <= 32, per the spec).
ACTION_TASKS = ["sorting", "stacking", "pretzel"]

TASK_LABELS = {
    "push_t": "Push T",
    "pretzel": "Pretzel",
    "push_chair": "Push Chair",
    "sorting": "Sorting",
    "stacking": "Stacking",
}

#: The two positive classes, each compared against "nominal" (= calibration) on its own.
COMPARISONS = {"success": "test success vs nominal", "failure": "test failure vs nominal"}

FEATURE_SETS = ["obs", "obs+act"]

CACHE_DIR = HERE / "cache"

#: Normalized times at which the prefix pool is cut.
TIME_GRID = np.round(np.arange(0.1, 1.0001, 0.1), 3)

#: Cap on pooled steps per class per grid point. Pools reach 48500 steps (stacking failures
#: at t=1), where another thousand near-duplicate steps no longer moves an AUROC.
MAX_STEPS_PER_CLASS = 2000

N_FOLDS = 5
N_TREES = 200

#: Below this many pooled steps (both classes together) an AUROC is not worth reporting.
MIN_ROWS = 20

#: What a grid point that could not be scored at all contributes.
EMPTY_SCORE = {"auroc": np.nan, "fold_mean": np.nan, "fold_std": np.nan,
               "n_folds_scored": 0, "fold_aurocs": ""}


# --------------------------------------------------------------------------------- data
def load_task(task: str, with_actions: bool) -> ProcessedRolloutDataset:
    """The processed rollout cache for ``task``, read-only.

    ``action_preds`` is requested only where used -- 1.4 GB on stacking, 512 MB on push_t.
    """
    tensors = ["obs_embeddings"] + (["action_preds"] if with_actions else [])
    ds = ProcessedRolloutDataset(
        task_data_path=os.path.join("data", task),
        base_config_path="configs",
        required_tensors=tensors,
    )
    ds.load_dataset()
    assert ds.dataset_loaded, f"no processed rollout cache for task '{task}'"
    # load_dataset() keys tensors as "<name>.pt" (same fix as sig_study/loader.py).
    for k in list(ds.data.keys()):
        if k.endswith(".pt"):
            ds.data[k[:-3]] = ds.data.pop(k)
    return ds


def build_classes(ds: ProcessedRolloutDataset) -> dict[str, dict]:
    """The three step pools: ``nominal``, ``success``, ``failure``.

    Each holds parallel per-step arrays: the flat step index, the episode id (offset per
    pool so it is unique and usable as a CV group), and normalized position ``u``.
    """
    out = {}
    for offset, (name, subset, want) in enumerate(
        (("nominal", "calibration", None), ("success", "test", True), ("failure", "test", False))
    ):
        starts, ends, labels = ds._filter_start_end_episode_indices(subset=subset)
        labels = np.asarray(labels).astype(bool)
        chosen = np.arange(len(starts)) if want is None else np.flatnonzero(labels == want)

        step, ep, u = [], [], []
        for e in chosen:
            s, t = int(starts[e]), int(ends[e])
            T = t - s
            step.append(np.arange(s, t))
            ep.append(np.full(T, offset * 100_000 + int(e)))
            u.append(np.zeros(T) if T == 1 else np.arange(T) / (T - 1))
        out[name] = {
            "step": np.concatenate(step),
            "ep": np.concatenate(ep),
            "u": np.concatenate(u),
            "n_ep": len(chosen),
            "lengths": (ends - starts)[chosen].astype(int),
        }
    return out


def gather(ds: ProcessedRolloutDataset, steps: np.ndarray, action_slice) -> tuple[np.ndarray, np.ndarray | None]:
    """``(obs [n, d_obs], act [n, B*H*3] or None)`` for the given flat step indices."""
    obs = np.asarray(ds.data["obs_embeddings"][steps], dtype=np.float32)
    if obs.ndim > 2:  # history_length > 1 would add an axis; keep the current step
        obs = obs[:, 0]
    if action_slice is None:
        return obs, None
    ap = ds.data["action_preds"][steps]
    if ap.ndim == 5:  # [n, robots, B, H, A]
        assert ap.shape[1] == 1, "multi-robot rollouts are not supported"
        ap = ap[:, 0]
    act = np.asarray(ap[..., action_slice], dtype=np.float32).reshape(len(steps), -1)
    return obs, act


# ------------------------------------------------------------------------------ sampling
def select_prefix(u: np.ndarray, prio: np.ndarray, t: float, cap: int) -> np.ndarray:
    """Row positions of the prefix pool ``u <= t``, subsampled to ``cap`` by ``prio``.

    Keeping the ``cap`` smallest priorities is a uniform draw (``prio`` is iid uniform) and
    stable across ``t``: the pools are nested, so neighbouring grid points share most rows
    instead of each re-drawing its own sample.
    """
    pool = np.flatnonzero(u <= t + 1e-12)
    if len(pool) > cap:
        pool = pool[np.argsort(prio[pool], kind="stable")[:cap]]
    return np.sort(pool)


def assign_folds(ep: np.ndarray, n_folds: int, rng: np.random.Generator) -> dict[int, int]:
    """``episode id -> fold``, round-robin over a shuffled episode list.

    Assigned per class once per task, before any grid point runs, so every ``t`` and both
    feature sets see the identical partition.
    """
    eps = np.unique(ep)
    rng.shuffle(eps)
    return {int(e): i % n_folds for i, e in enumerate(eps)}


# -------------------------------------------------------------------------------- scoring
def cv_auroc(X: np.ndarray, y: np.ndarray, fold: np.ndarray, n_folds: int, seed: int) -> dict:
    """Pooled out-of-fold AUROC of a random forest, plus the per-fold spread.

    A fold whose held-out part is single-class still contributes its predictions to the
    pooled ``auroc``; only its own per-fold AUROC is undefined.

    ``fold_aurocs`` keeps the per-fold numbers **indexed by fold id**, NaN where unscored.
    :func:`did_curve` needs that: all four AUROCs entering one DiD must come from the same
    fold, and a positional list would misalign the moment one of them dropped a fold.
    """
    oof_y, oof_p = [], []
    per_fold = np.full(n_folds, np.nan)
    for f in range(n_folds):
        test = fold == f
        train = ~test
        if not test.any() or len(np.unique(y[train])) < 2:
            continue
        clf = RandomForestClassifier(
            n_estimators=N_TREES, n_jobs=-1, random_state=seed, class_weight="balanced"
        )
        clf.fit(X[train], y[train])
        p = clf.predict_proba(X[test])[:, 1]
        oof_y.append(y[test])
        oof_p.append(p)
        if len(np.unique(y[test])) == 2:
            per_fold[f] = roc_auc_score(y[test], p)

    if not oof_y:
        return EMPTY_SCORE
    yy, pp = np.concatenate(oof_y), np.concatenate(oof_p)
    scored = per_fold[np.isfinite(per_fold)]
    return {
        "auroc": roc_auc_score(yy, pp) if len(np.unique(yy)) == 2 else np.nan,
        "fold_mean": float(scored.mean()) if len(scored) else np.nan,
        "fold_std": float(scored.std()) if len(scored) else np.nan,
        "n_folds_scored": len(scored),
        "fold_aurocs": ";".join("nan" if not np.isfinite(v) else f"{v:.6f}" for v in per_fold),
    }


def run_task(task: str, seed: int) -> pd.DataFrame:
    """The full (feature set x comparison x t) grid for one task, with a shape report."""
    with_actions = task in ACTION_TASKS
    ds = load_task(task, with_actions)
    classes = build_classes(ds)

    obs_shape = tuple(ds.data["obs_embeddings"].shape)
    d_obs = int(obs_shape[-1])
    B, H = chunk_shape(ds)
    if with_actions:
        action_slice = ds._get_action_slices(required_actions=["position"], optional_actions=[])
        ap_shape = tuple(ds.data["action_preds"].shape)
        d_act = B * H * len(action_slice)
    else:
        action_slice, ap_shape, d_act = None, None, 0

    print(f"\n{'=' * 88}\n=== {TASK_LABELS[task]} ({task})\n{'=' * 88}")
    print(f"  obs_embeddings tensor        {obs_shape}    d_obs = {d_obs}")
    if with_actions:
        print(f"  action_preds tensor          {ap_shape}    B = {B}, H = {H}, A = {ap_shape[-1]}")
        print(f"  position channels kept       {list(map(int, action_slice))}  ->  "
              f"d_act = B*H*D = {B}*{H}*{len(action_slice)} = {d_act}")
    else:
        print(f"  action_preds                 not loaded (B = {B} > 32: excluded from Q3 by the spec)")

    print("\n  --- step pools (whole episodes, before any prefix cut) ---")
    for name, c in classes.items():
        L = c["lengths"]
        print(f"  {name:<8} {c['n_ep']:>4} episodes  {len(c['step']):>6} steps   "
              f"episode length min/median/max = {L.min()}/{int(np.median(L))}/{L.max()}")
    if task == "push_t":
        starts, ends, labels = ds._filter_start_end_episode_indices(subset="calibration")
        n_fail = int((~np.asarray(labels).astype(bool)).sum())
        print(f"  NB: {n_fail}/{len(labels)} calibration episodes of push_t are failures, so "
              "'nominal' is contaminated (accepted, per the spec).")

    rng = np.random.default_rng(seed)
    prio = {name: rng.random(len(c["step"])) for name, c in classes.items()}
    fold_of = {name: assign_folds(c["ep"], N_FOLDS, rng) for name, c in classes.items()}

    # Resolve every row set the grid needs before gathering, so the union is read out of
    # the big tensors in one pass.
    plan = {}
    for comp in COMPARISONS:
        n_folds = min(N_FOLDS, classes["nominal"]["n_ep"], classes[comp]["n_ep"])
        for t in TIME_GRID:
            a = select_prefix(classes["nominal"]["u"], prio["nominal"], t, MAX_STEPS_PER_CLASS)
            b = select_prefix(classes[comp]["u"], prio[comp], t, MAX_STEPS_PER_CLASS)
            plan[(comp, float(t))] = {
                "rows_nom": a,
                "rows_pos": b,
                "n_folds": n_folds,
                "pool_nom": int((classes["nominal"]["u"] <= t + 1e-12).sum()),
                "pool_pos": int((classes[comp]["u"] <= t + 1e-12).sum()),
            }

    needed = np.unique(
        np.concatenate(
            [classes["nominal"]["step"][p["rows_nom"]] for p in plan.values()]
            + [classes[comp]["step"][p["rows_pos"]] for (comp, _), p in plan.items()]
        )
    )
    obs_u, act_u = gather(ds, needed, action_slice)
    print(f"\n  --- features gathered once for the {len(needed)} distinct steps the grid touches ---")
    print(f"  obs block   {obs_u.shape}  mean {obs_u.mean():+.4g}  std {obs_u.std():.4g}  "
          f"min {obs_u.min():+.4g}  max {obs_u.max():+.4g}")
    if act_u is not None:
        print(f"  act block   {act_u.shape}  mean {act_u.mean():+.4g}  std {act_u.std():.4g}  "
              f"min {act_u.min():+.4g}  max {act_u.max():+.4g}")
    def to_rows(steps: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(needed, steps)
        assert np.array_equal(needed[pos], steps), "a required step is missing from the gathered block"
        return pos

    feature_sets = FEATURE_SETS if with_actions else ["obs"]
    print(f"\n  --- {len(feature_sets)} feature set(s) x {len(COMPARISONS)} comparisons x "
          f"{len(TIME_GRID)} grid points, {N_FOLDS}-fold CV over episodes ---")

    rows = []
    for comp in COMPARISONS:
        print(f"\n  [{COMPARISONS[comp]}]")
        print(f"  {'t':>5} {'pool_nom':>9} {'pool_pos':>9} {'n_nom':>6} {'n_pos':>6} {'X shape':>16} "
              f"{'fold sizes':>22} {'AUROC':>7} {'fold mean+-std':>16}")
        for t in TIME_GRID:
            p = plan[(comp, float(t))]
            r_nom, r_pos = p["rows_nom"], p["rows_pos"]

            steps = np.concatenate([classes["nominal"]["step"][r_nom], classes[comp]["step"][r_pos]])
            eps = np.concatenate([classes["nominal"]["ep"][r_nom], classes[comp]["ep"][r_pos]])
            y = np.concatenate([np.zeros(len(r_nom)), np.ones(len(r_pos))])
            fold = np.array([fold_of["nominal" if e < 100_000 else comp][int(e)] for e in eps])
            pos = to_rows(steps)
            assert len(pos) == len(y) == len(fold) == len(eps)
            # No episode may straddle a fold boundary; that is the point of grouping by it.
            assert all(len(set(fold[eps == e])) == 1 for e in np.unique(eps))

            fold_sizes = [int((fold == f).sum()) for f in range(p["n_folds"])]
            for fs in feature_sets:
                X = obs_u[pos] if fs == "obs" else np.concatenate([obs_u[pos], act_u[pos]], axis=1)
                if len(y) < MIN_ROWS or len(np.unique(y)) < 2:
                    res = dict(EMPTY_SCORE)
                else:
                    res = cv_auroc(X, y, fold, p["n_folds"], seed)
                rows.append({
                    "task": task, "features": fs, "comparison": comp, "t": float(t),
                    "n_ep_nominal": classes["nominal"]["n_ep"], "n_ep_pos": classes[comp]["n_ep"],
                    "pool_nominal": p["pool_nom"], "pool_pos": p["pool_pos"],
                    "n_nominal": len(r_nom), "n_pos": len(r_pos),
                    "n_rows": len(y), "d": int(X.shape[1]),
                    "B": B, "H": H, "d_obs": d_obs, "d_act": d_act,
                    "n_folds": p["n_folds"], **res,
                })
                tag = "" if fs == "obs" else "  (+act)"
                print(f"  {t:>5.1f} {p['pool_nom']:>9} {p['pool_pos']:>9} {len(r_nom):>6} {len(r_pos):>6} "
                      f"{str(X.shape):>16} {str(fold_sizes):>22} {res['auroc']:>7.3f} "
                      f"{res['fold_mean']:>7.3f}+-{res['fold_std']:<6.3f}{tag}")

    return pd.DataFrame(rows)


def chunk_shape(ds: ProcessedRolloutDataset) -> tuple[int, int]:
    """``(B, H)`` of a step's action chunk batch.

    The loaded tensor is authoritative; where it was not loaded, fall back to the metadata,
    which ``load_dataset`` always reads. ``action_pred_shape`` is an *unresolved* OmegaConf
    interpolation on sorting and stacking, hence the digit check -- though both are action
    tasks and never reach that branch.
    """
    if "action_preds" in ds.data:
        shape = tuple(ds.data["action_preds"].shape)
        return int(shape[-3]), int(shape[-2])
    actions = ds.data["metadata"]["actions"]
    head = str(actions["action_pred_shape"]).strip("() ").split(",")[0].strip()
    assert head.isdigit(), f"cannot read B from action_pred_shape {actions['action_pred_shape']!r}"
    return int(head), int(actions["action_prediction_horizon"])


# ------------------------------------------------------------------------------- plotting
#: Green for success, red for failure, per the spec.
CURVES = (("success", "tab:green"), ("failure", "tab:red"))


def _fold_list(cell) -> np.ndarray:
    """The per-fold AUROCs of one row, by fold id. Empty when the point was never scored."""
    if not isinstance(cell, str) or not cell:
        return np.zeros(0)
    return np.array([float(v) for v in cell.split(";") if v])


def did_curve(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Per-``t`` difference-in-differences, with a confidence interval on the DiD itself.

    The estimand Q3 asks about is one number per grid point,

        DiD = (failure − success | obs+act) − (failure − success | obs),

    a contrast of four AUROCs. Forming it **inside each fold** before averaging is what makes
    an interval possible: the four AUROCs of one fold share its held-out episodes -- both
    comparisons hold out the same nominal episodes, both feature sets hold out the same
    episodes on both sides -- so each fold is one measurement of the estimand.

    The band is therefore a **95% Student-t CI on the mean DiD** (``k - 1`` df), not the
    fold-to-fold spread: the question is whether the DiD differs from zero, not how much the
    contrast wobbles between folds. At ``k = 5`` the critical value is 2.78, so the interval
    is about ±1.24 fold standard deviations.

    Caveats: neighbouring folds train on overlapping 4/5 subsets, so the interval is if
    anything optimistic, and it covers fold resampling only -- not drawing new rollouts.

    Returns per ``t`` the fold-mean DiD (the plotted line), its CI, and the same DiD from the
    pooled AUROCs as a cross-check.
    """
    parts = {}
    for fs in FEATURE_SETS:
        for comp in COMPARISONS:
            s = df[(df["task"] == task) & (df["features"] == fs) & (df["comparison"] == comp)]
            if s.empty:
                return pd.DataFrame()
            parts[(fs, comp)] = s.set_index("t").sort_index()

    grid = None
    for p in parts.values():
        grid = p.index if grid is None else grid.intersection(p.index)

    out = []
    for t in grid:
        folds = {k: _fold_list(p.loc[t, "fold_aurocs"]) for k, p in parts.items()}
        # Fold ids are 0-based everywhere, so truncating to the shortest keeps them aligned.
        width = min(len(a) for a in folds.values())
        f = {k: a[:width] for k, a in folds.items()}
        did = ((f[("obs+act", "failure")] - f[("obs+act", "success")])
               - (f[("obs", "failure")] - f[("obs", "success")]))
        did = did[np.isfinite(did)]

        k = len(did)
        mean = float(did.mean()) if k else np.nan
        half = (float(student_t.ppf(0.975, k - 1) * did.std(ddof=1) / np.sqrt(k))
                if k > 1 else np.nan)
        pooled = ((parts[("obs+act", "failure")].loc[t, "auroc"] - parts[("obs+act", "success")].loc[t, "auroc"])
                  - (parts[("obs", "failure")].loc[t, "auroc"] - parts[("obs", "success")].loc[t, "auroc"]))
        out.append({"t": t, "did": mean, "half": half, "lo": mean - half, "hi": mean + half,
                    "fold_std": float(did.std(ddof=1)) if k > 1 else np.nan,
                    "k": k, "pooled": float(pooled)})
    return pd.DataFrame(out).sort_values("t")


def _panel_title(df: pd.DataFrame, task: str, extra: str = "") -> str:
    """``Task — nominal N ep, test N success / N failure, d_obs = ...`` plus ``extra``."""
    head = df[(df["task"] == task) & (df["features"] == "obs")]
    name = TASK_LABELS[task]
    if len(head):
        h = head.iloc[0]
        n_succ = int(head[head["comparison"] == "success"]["n_ep_pos"].iloc[0])
        n_fail = int(head[head["comparison"] == "failure"]["n_ep_pos"].iloc[0])
        name += (f"   —   nominal {int(h['n_ep_nominal'])} ep,  test {n_succ} success / "
                 f"{n_fail} failure,  d_obs = {int(h['d_obs'])}")
    return name + extra


def _finish(fig, axes, drawn, stem: pathlib.Path, subtitle: str, title: str) -> None:
    """Shared scaffolding: legend on the first populated panel, labels, save as png + pdf."""
    if drawn is not None:
        drawn.legend(fontsize=8, loc="lower right", framealpha=0.9)
    axes[-1].set_xlabel("normalized episode time $t$   (model pooled over all steps with $u \\leq t$)",
                        fontsize=9)
    # One size throughout: a suptitle cannot mix sizes, and a larger title line would push
    # the subtitle past the 7.6 in canvas.
    fig.suptitle(f"{title}\n{subtitle}", fontsize=9.8)
    for ext in ("png", "pdf"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=160 if ext == "png" else None)
        print(f"  wrote {os.path.relpath(path, ROOT_DIR)}")
    plt.close(fig)


def _panels():
    return plt.subplots(len(TASKS), 1, figsize=(7.6, 14.0), sharex=True, constrained_layout=True)


def _excluded_note(df: pd.DataFrame, task: str, ax) -> None:
    meta = df[df["task"] == task]
    note = (f"not evaluated: B = {int(meta['B'].iloc[0])} > 32\n(excluded from Q3 by the spec)"
            if len(meta) else "not run\n(pass --tasks with this task)")
    ax.text(0.5, 0.5, note, transform=ax.transAxes, ha="center", va="center", fontsize=9, color="0.35")


def plot_absolute(df: pd.DataFrame, stem: pathlib.Path) -> None:
    """Figure 1: observation-only AUROC against normalized episode time, one panel per task."""
    lo = float(np.nanmin(df[df["features"] == "obs"]["auroc"].to_numpy(dtype=float)))
    ylim = (max(0.0, min(0.45, lo) - 0.05), 1.02)

    fig, axes = _panels()
    drawn = None
    for ax, task in zip(axes, TASKS):
        sub = df[(df["task"] == task) & (df["features"] == "obs")]
        ax.axhline(0.5, color="0.5", lw=0.9, ls=":", zorder=1)
        if sub.empty:
            _excluded_note(df, task, ax)
        else:
            for comp, colour in CURVES:
                s = sub[sub["comparison"] == comp].sort_values("t")
                # A spread of AUROCs cannot leave [0, 1] even where mean +- std would.
                lo_b = np.clip(s["fold_mean"] - s["fold_std"], 0.0, 1.0)
                hi_b = np.clip(s["fold_mean"] + s["fold_std"], 0.0, 1.0)
                ax.fill_between(s["t"], lo_b, hi_b, color=colour, alpha=0.15, lw=0, zorder=2)
                ax.plot(s["t"], s["auroc"], color=colour, marker="o", ms=3.5, lw=1.6,
                        label=f"test {comp} vs nominal", zorder=3)
            drawn = drawn or ax
        ax.set_title(_panel_title(df, task), fontsize=9, loc="left")
        ax.set_ylim(*ylim)
        ax.set_ylabel("AUROC", fontsize=9)
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=8)

    _finish(fig, axes, drawn, stem,
            f"random forest, {N_FOLDS}-fold CV over episodes;  line = pooled out-of-fold "
            "AUROC,  band = per-fold mean ±1 std",
            "Oracle separability from observation embeddings alone  (Q1 / Q2)")


def plot_relative(df: pd.DataFrame, stem: pathlib.Path) -> None:
    """Figure 2: the difference-in-differences against the observation-only baseline (Q3).

    One curve per task, since the estimand is one number per grid point. y = 0 is "the action
    channels changed nothing"; a band spanning 0 means no effect detectable on these folds.
    """
    curves = {task: did_curve(df, task) for task in TASKS}
    baseline = _gap_frame(df, "obs").mean(axis=1)

    # Scale to the curves and the *typical* interval, not the widest one: pretzel's t=0.1 CI
    # is ±0.18 on a 20-row pool and would flatten every panel. Clipped ends are annotated.
    populated = [c for c in curves.values() if not c.empty]
    ends = np.abs(np.concatenate([np.concatenate([c["lo"], c["hi"]]) for c in populated])) \
        if populated else np.array([0.05])
    lines = np.abs(np.concatenate([c["did"] for c in populated])) if populated else np.array([0.0])
    m = max(0.05, 1.18 * max(float(np.nanmax(lines)), float(np.nanquantile(ends, 0.90))))

    fig, axes = _panels()
    drawn = None
    for ax, task in zip(axes, TASKS):
        # Heavier than figure 1's chance line: here it is what the panel is measured against.
        ax.axhline(0.0, color="0.25", lw=1.1, zorder=1)
        c = curves[task]
        if c.empty:
            _excluded_note(df, task, ax)
            extra = ""
        else:
            ax.fill_between(c["t"], c["lo"], c["hi"], color="tab:purple", alpha=0.15, lw=0,
                            zorder=2, label="95% CI over folds")
            ax.plot(c["t"], c["did"], color="tab:purple", marker="o", ms=3.5, lw=1.7,
                    label="DiD: change in the failure−success gap", zorder=3)
            drawn = drawn or ax

            g = float(baseline.get(task, np.nan))
            mean_did = float(c["did"].mean())
            n_sig = int((np.sign(c["lo"]) == np.sign(c["hi"])).sum())
            d_act = int(df[(df["task"] == task) & (df["features"] == "obs+act")]["d_act"].iloc[0])
            extra = (f" + d_act = {d_act}\nobs-only gap {g:+.3f};  mean DiD {mean_did:+.3f}"
                     f" = {100 * mean_did / g:+.0f}% of it;  CI excludes 0 at {n_sig}/{len(c)} points")

            # Same numbers as a fraction of the gap this task's actions are perturbing, so
            # the left axis has a denominator.
            rel = ax.twinx()
            rel.set_ylim(-100 * m / g, 100 * m / g)
            rel.set_ylabel("% of obs-only gap", fontsize=8, color="0.4")
            rel.tick_params(labelsize=7, colors="0.4")

            off = int((c["hi"] > m).sum() + (c["lo"] < -m).sum())
            if off:
                ax.text(0.012, 0.96, f"{off} CI end(s) beyond the axis", transform=ax.transAxes,
                        ha="left", va="top", fontsize=7, color="0.45")
        ax.set_title(_panel_title(df, task, extra), fontsize=9, loc="left")
        ax.set_ylim(-m, m)
        ax.set_ylabel("change in the failure−success\nAUROC gap", fontsize=9)
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=8)

    _finish(fig, axes, drawn, stem,
            "DiD = (fail − succ | obs+act) − (fail − succ | obs), in AUROC points, formed per fold\n"
            "band = 95% CI (Student-t, 4 df);  right axis rescales it to % of that task's obs-only gap",
            "Do the action channels help? Difference-in-differences vs the "
            "observation-only baseline  (Q3)")


# -------------------------------------------------------------------------------- reports
def _pivot(df: pd.DataFrame, features: str, value: str) -> pd.DataFrame:
    s = df[df["features"] == features]
    out = s.pivot_table(index=["task", "comparison"], columns="t", values=value)
    order = [(t, c) for t in TASKS for c in COMPARISONS if (t, c) in out.index]
    return out.reindex(order)


def report(df: pd.DataFrame) -> None:
    """The tables that actually answer Q1-Q3."""
    fmt = lambda v: f"{v:.3f}"  # noqa: E731

    for features in FEATURE_SETS:
        if df[df["features"] == features].empty:
            continue
        print(f"\n{'=' * 88}\n=== AUROC by normalized time -- features: {features}\n{'=' * 88}")
        print(_pivot(df, features, "auroc").to_string(float_format=fmt, na_rep="  --  "))

    print(f"\n{'=' * 88}\n=== Q1/Q2: failure AUROC minus success AUROC (obs only)\n{'=' * 88}")
    print("  the control-adjusted signal. ~0 means failure steps are no more distinguishable")
    print("  from nominal than success steps are, i.e. the one-class problem is hopeless.\n")
    gap = _gap_frame(df, "obs")
    print(gap.to_string(float_format=lambda v: f"{v:+.3f}", na_rep="  --  "))
    print(f"\n  mean over t and tasks: {np.nanmean(gap.to_numpy(dtype=float)):+.4f}")
    late = gap[[c for c in gap.columns if c >= 0.7]]
    print(f"  mean over the late window t >= 0.7: {np.nanmean(late.to_numpy(dtype=float)):+.4f}")

    weak = df[(df["features"] == "obs") & (df["auroc"] < 0.45)]
    if len(weak):
        print("\n  cells below 0.45 -- the forest ranks *held-out* episodes worse than chance. That is")
        print("  the grouped-CV signature of a class whose own episodes are more heterogeneous among")
        print("  themselves than they are different from the other class: a held-out nominal episode")
        print("  lands where no training nominal episode did, so it is called 'test' more confidently")
        print("  than the held-out test episodes are, and the ranking inverts. Read these as 'no")
        print("  separation that generalises to an unseen episode'. Note the fold std: on stacking it")
        print("  is 0.03 over thousands of rows, so this is reproducible, not sampling noise.")
        for _, r in weak.iterrows():
            print(f"    {r['task']:<11} {r['comparison']:<8} t={r['t']:.1f}  AUROC {r['auroc']:.3f}  "
                  f"rows {int(r['n_rows']):>5}  episodes {int(r['n_ep_nominal'])}+{int(r['n_ep_pos'])}"
                  f"  fold std {r['fold_std']:.3f}")

    if not df[df["features"] == "obs+act"].empty:
        print(f"\n{'=' * 88}\n=== Q3: do the action channels help? (obs+act minus obs)\n{'=' * 88}")
        signed = lambda v: f"{v:+.3f}"  # noqa: E731
        a = _pivot(df, "obs+act", "auroc")
        b = _pivot(df, "obs", "auroc").reindex(a.index)
        print("\n  -- change in AUROC per comparison (positive = actions help), pooled out-of-fold --")
        print("  context only; neither row is the estimand, because both absorb any shift that has")
        print("  nothing to do with failing. The contrast of the two is what Q3 asks about.\n")
        print((a - b).to_string(float_format=signed, na_rep="  --  "))

        dids = {task: did_curve(df, task) for task in TASKS}
        dids = {k: v for k, v in dids.items() if not v.empty}

        def did_table(column):
            return pd.DataFrame({t: dict(zip(c["t"], c[column])) for t, c in dids.items()}).T

        print("\n  -- DIFFERENCE-IN-DIFFERENCES: the change in the failure−success gap. Figure 2. --")
        print("  formed per fold, then averaged over the 5 folds.\n")
        print(did_table("did").to_string(float_format=signed, na_rep="  --  "))
        print("\n  -- 95% CI half-width on that DiD (Student-t over folds, 4 df) --")
        print(did_table("half").to_string(float_format=fmt, na_rep="  --  "))
        print("\n  -- per task. Units are AUROC points; the gap the DiD perturbs is the scale. --")
        base = _gap_frame(df, "obs").mean(axis=1)
        for task, c in dids.items():
            n_sig = int((np.sign(c["lo"]) == np.sign(c["hi"])).sum())
            g, mean_did = float(base.get(task, np.nan)), float(c["did"].mean())
            print(f"    {task:<11} obs-only gap {g:+.4f} (mean over t)   mean DiD {mean_did:+.4f}"
                  f" = {100 * mean_did / g:+.0f}% of it   "
                  f"typical CI half-width {np.nanmedian(c['half']):.4f}   "
                  f"CI excludes 0 at {n_sig}/{len(c)} grid points   k = {int(c['k'].min())} folds")
        allmean = float(np.nanmean(did_table("did").to_numpy(dtype=float)))
        print(f"\n  mean DiD over t and tasks: {allmean:+.4f}")
        print("\n  -- cross-check: the same DiD from the pooled AUROCs rather than the fold mean --")
        print("  (the plotted line uses the fold mean so that it and its CI come from the same numbers)")
        print(did_table("pooled").to_string(float_format=signed, na_rep="  --  "))

    print(f"\n{'=' * 88}\n=== per-fold spread (obs only), to read the curves against\n{'=' * 88}")
    print(_pivot(df, "obs", "fold_std").to_string(float_format=fmt, na_rep="  --  "))


def _gap_frame(df: pd.DataFrame, features: str) -> pd.DataFrame:
    """``task x t`` frame of (failure AUROC − success AUROC)."""
    piv = _pivot(df, features, "auroc")
    tasks = [t for t in TASKS if (t, "failure") in piv.index and (t, "success") in piv.index]
    return pd.DataFrame(
        {t: piv.loc[(t, "failure")] - piv.loc[(t, "success")] for t in tasks}
    ).T.reindex(tasks)


# ------------------------------------------------------------------------------------ run
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="separability", description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="recompute instead of reading the cache")
    parser.add_argument("--plot-only", action="store_true", help="redraw from the cache and exit")
    args = parser.parse_args(argv)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for task in args.tasks:
        cache = CACHE_DIR / f"{task}__seed{args.seed}.csv"
        if cache.exists() and not args.force:
            frames.append(pd.read_csv(cache))
            print(f"loaded cached scores for {task} ({os.path.relpath(cache, ROOT_DIR)})")
            continue
        if args.plot_only:
            print(f"no cache for {task}; skipping (--plot-only)", file=sys.stderr)
            continue
        frame = run_task(task, args.seed)
        frame.to_csv(cache, index=False)
        frames.append(frame)

    if not frames:
        print("nothing to plot", file=sys.stderr)
        return 1
    df = pd.concat(frames, ignore_index=True)

    report(df)

    print(f"\n{'=' * 88}\n=== figures\n{'=' * 88}")
    plot_absolute(df, HERE / "separability_obs")
    plot_relative(df, HERE / "separability_obs_act")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
