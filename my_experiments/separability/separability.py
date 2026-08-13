"""Oracle separability of FIPER's nominal data from test success / failure steps over normalized episode time -- ``specs/experiments/separability.md``."""

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
from cd.utils.misc import median_distance  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.kernel_approximation import RBFSampler  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

CACHE_DIR = HERE / "cache"
DATA_DIR = pathlib.Path(ROOT_DIR) / "data"

ALL_TASKS = ["push_t", "pretzel", "push_chair", "sorting", "stacking"]
#: nominal = FIPER's calibration split; the other two are the labelled test split.
GROUPS = ("nominal", "success", "failure")
FEATURE_SETS = ("obs", "obs+act")
COMPARISONS = ("success", "failure")

#: Normalized times at which the pooled classifier is fitted.
TIMES = np.round(np.arange(1, 11) / 10.0, 3)
#: Steps kept per group per task; what bounds the fit sets is this, not the data.
CAP = 1500
#: Random Fourier Features in the action-chunk kernel mean embedding.
M_RFF = 256
#: Elements per RFF block, bounding the [rows, B, M] intermediate.
_RFF_ELEM_BUDGET = 30_000_000

MODELS = {
    "rf": lambda seed: RandomForestClassifier(
        n_estimators=200, class_weight="balanced", n_jobs=1, random_state=seed
    ),
    "logreg": lambda seed: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
    ),
}


# ---------------------------------------------------------------------------- the data
def load_metadata(task: str) -> dict:
    """Read one task's processed-rollout metadata."""
    with open(DATA_DIR / task / "processed_rollouts" / "metadata.pkl", "rb") as f:
        return pickle.load(f)


def episodes(md: dict) -> pd.DataFrame:
    """Per-episode start index, length and separability group."""
    start = np.asarray(md["episode_start_indices"], dtype=int)
    length = np.asarray(md["episode_end_indices"], dtype=int) - start
    calib = np.asarray(md["calibration_rollout_labels"], dtype=bool)
    test = np.asarray(md["test_rollout_labels"], dtype=bool)
    ok = np.asarray(md["successful_rollout_labels"], dtype=bool)
    group = np.full(len(start), "", dtype=object)
    # push_t's failed calibration episodes stay in "nominal": the spec takes FIPER's train
    # split as the nominal class and explicitly ignores that contamination.
    group[calib] = "nominal"
    group[test & ok] = "success"
    group[test & ~ok] = "failure"
    return pd.DataFrame({"episode": np.arange(len(start)), "start": start,
                         "ep_len": length, "successful": ok, "group": group})


def select_steps(eps: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Evenly strided steps per episode, one stride per group so each group keeps ~cap steps."""
    rows = []
    for g in GROUPS:
        sub = eps[eps.group == g]
        stride = max(1, int(np.ceil(sub.ep_len.sum() / cap)))
        for e in sub.itertuples():
            # Strided from step 0, so every episode contributes at every normalized time and
            # the pools stay nested in time (a later t only ever adds steps).
            local = np.arange(0, e.ep_len, stride)
            rows.append(pd.DataFrame({"episode": e.episode, "group": g, "local": local,
                                      "ep_len": e.ep_len, "stride": stride,
                                      "index": e.start + local}))
    return pd.concat(rows, ignore_index=True)


def assign_folds(sel: pd.DataFrame, k: int, seed: int) -> np.ndarray:
    """CV fold of every selected step, k-folding each group's *episodes* separately.

    Folding the groups separately is what lets one fold index hold out episodes of all three
    at once, so the four AUROCs a DiD is built from share their held-out nominal episodes.
    """
    fold = {}
    for g in GROUPS:
        eps = np.unique(sel.episode[sel.group == g])
        for j, (_, held) in enumerate(KFold(n_splits=k, shuffle=True, random_state=seed).split(eps)):
            fold.update({int(e): j for e in eps[held]})
    return sel.episode.map(fold).to_numpy(dtype=int)


def pool_mask(t: float, local: np.ndarray, ep_len: np.ndarray) -> np.ndarray:
    """Steps whose normalized time ``(local+1)/ep_len`` is at most ``t``; never empty."""
    return local < np.maximum(1, np.floor(t * ep_len + 1e-9))


# ------------------------------------------------------------------------- the features
def build_features(task: str, md: dict, sel: pd.DataFrame, seed: int, m_rff: int) -> dict:
    """Observation embeddings and action-chunk kernel mean embeddings for the selected steps."""
    idx = torch.from_numpy(sel["index"].to_numpy().copy())
    root = DATA_DIR / task / "processed_rollouts"
    # mmap: only the selected rows are faulted in, ~7% of stacking's 1.45 GB of action_preds.
    obs = torch.load(root / "obs_embeddings.pt", weights_only=True, mmap=True)[idx].numpy()
    chunks = torch.load(root / "action_preds.pt", weights_only=True, mmap=True)[idx].numpy()
    if chunks.ndim == 5:  # [N, robots, B, H, D]
        assert chunks.shape[1] == 1, f"{task}: multi-robot action_preds not supported"
        chunks = chunks[:, 0]

    # mu_A = mean_b phi(vec(a_b)) over the batch of chunks, on the D=3 task-space coordinates.
    pos = np.asarray(md["actions"]["action_mapping"]["position"], dtype=int)
    parts = np.ascontiguousarray(chunks[..., pos]).reshape(len(idx), chunks.shape[1], -1)
    sigma = median_distance(parts.reshape(-1, parts.shape[-1]), rng=seed)
    sampler = RBFSampler(gamma=1.0 / (2.0 * sigma**2), n_components=m_rff, random_state=seed)
    sampler.fit(np.zeros((1, parts.shape[-1])))
    W = sampler.random_weights_.astype(np.float32)
    b = sampler.random_offset_.astype(np.float32)

    mu = np.empty((len(idx), m_rff), dtype=np.float32)
    block = max(1, _RFF_ELEM_BUDGET // (parts.shape[1] * m_rff))
    scale = np.float32(np.sqrt(2.0 / m_rff))
    for s in range(0, len(idx), block):
        mu[s : s + block] = (scale * np.cos(parts[s : s + block] @ W + b)).mean(axis=1)
    return {"obs": obs.astype(np.float32), "kme": mu, "sigma_phi": float(sigma),
            "n_chunks": int(parts.shape[1]), "part_dim": int(parts.shape[-1])}


def get_features(task: str, md: dict, sel: pd.DataFrame, seed: int, cap: int,
                 m_rff: int, force: bool) -> dict:
    """Cached :func:`build_features`."""
    path = CACHE_DIR / f"{task}__features__seed{seed}__cap{cap}__M{m_rff}.npz"
    if path.exists() and not force:
        with np.load(path) as z:
            loaded = {k: z[k] for k in z.files}
        return {k: (v if v.ndim else v.item()) for k, v in loaded.items()}
    feats = build_features(task, md, sel, seed, m_rff)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, **feats)
    return feats


# -------------------------------------------------------------------------- the scoring
def score_cell(t: float, j: int, code: np.ndarray, local: np.ndarray, ep_len: np.ndarray,
               fold: np.ndarray, X_obs: np.ndarray, X_oa: np.ndarray, seed: int,
               models: list[str]) -> list[dict]:
    """AUROC of every (feature set, comparison, model) at one normalized time and CV fold."""
    keep = pool_mask(t, local, ep_len)
    rows = []
    for c, comparison in enumerate(COMPARISONS, start=1):
        pool = keep & ((code == 0) | (code == c))
        y = (code == c).astype(int)
        tr, te = pool & (fold != j), pool & (fold == j)
        for fname, X in (("obs", X_obs), ("obs+act", X_oa)):
            for mname in models:
                clf = MODELS[mname](seed).fit(X[tr], y[tr])
                p = clf.predict_proba(X[te])[:, 1]
                auc = roc_auc_score(y[te], p) if len(np.unique(y[te])) == 2 else np.nan
                rows.append({"t": t, "fold": j, "comparison": comparison, "features": fname,
                             "model": mname, "auroc": float(auc), "n_train": int(tr.sum()),
                             "n_test": int(te.sum()), "n_test_pos": int(y[te].sum())})
    return rows


def run_task(task: str, seed: int, k: int, cap: int, m_rff: int, models: list[str],
             n_jobs: int, force: bool) -> pd.DataFrame:
    """Selection, features and cross-validated AUROCs for one task, with a shape report."""
    path = CACHE_DIR / f"{task}__scores__seed{seed}__k{k}__cap{cap}__M{m_rff}.csv"
    md = load_metadata(task)
    eps = episodes(md)
    sel = select_steps(eps, cap)
    fold = assign_folds(sel, k, seed)
    feats = get_features(task, md, sel, seed, cap, m_rff, force)
    report_task(task, md, eps, sel, fold, feats, k)

    if path.exists() and not force:
        df = pd.read_csv(path)
        # The cache key has no model in it, so a cache written for a narrower --models is
        # only reusable when it already holds every model this run asks for.
        if set(models) <= set(df.model.unique()):
            print(f"  scores       cached {df.shape} <- {os.path.relpath(path, ROOT_DIR)}")
            return df[df.model.isin(models)]
        print(f"  scores       cache holds only {sorted(df.model.unique())}, re-running")

    X_obs = feats["obs"]
    X_oa = np.hstack([X_obs, feats["kme"]])
    # Integer group codes rather than the object column: these arrays cross a process
    # boundary once per (time, fold) job.
    code = np.array([GROUPS.index(g) for g in sel.group], dtype=np.int8)
    local, ep_len = sel.local.to_numpy(), sel.ep_len.to_numpy()
    started = time.time()
    batches = Parallel(n_jobs=n_jobs)(
        delayed(score_cell)(t, j, code, local, ep_len, fold, X_obs, X_oa, seed, models)
        for t in TIMES for j in range(k)
    )
    df = pd.DataFrame([r for b in batches for r in b])
    df.insert(0, "task", task)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  scores       {len(df)} fits in {time.time() - started:.0f}s, {df.shape} -> "
          f"{os.path.relpath(path, ROOT_DIR)}")
    return df


# ------------------------------------------------------------------------ the reporting
def report_task(task: str, md: dict, eps: pd.DataFrame, sel: pd.DataFrame, fold: np.ndarray,
                feats: dict, k: int) -> None:
    """Print the shapes and summary stats of everything this task's AUROCs are built from."""
    print(f"\n===== {task}   ({md['num_rollouts']} rollouts, {md['num_steps']} steps, "
          f"action channels {md['actions']['action_mapping']['position']} of "
          f"{md['actions']['action_dim']})")
    per_ep = sel.assign(fold=fold).drop_duplicates("episode")
    rows = []
    for g in GROUPS:
        e, s = eps[eps.group == g], sel[sel.group == g]
        rows.append({"group": g, "episodes": len(e),
                     "ep_len min/med/max": f"{e.ep_len.min()}/{int(e.ep_len.median())}/{e.ep_len.max()}",
                     "steps": int(e.ep_len.sum()), "stride": int(s.stride.iloc[0]), "kept": len(s),
                     "episodes/fold": np.bincount(per_ep.fold[per_ep.group == g],
                                                  minlength=k).tolist()})
    print(pd.DataFrame(rows).to_string(index=False))
    n_fail = int((~eps.successful[eps.group == "nominal"]).sum())
    if n_fail:
        print(f"  note: {n_fail} nominal episodes are failures, kept as nominal per the spec")

    for name, arr in (("obs_embeddings", feats["obs"]), ("KME mu_A", feats["kme"])):
        print(f"  {name:<15} {str(arr.shape):<14} {arr.dtype}  mean {arr.mean():+.4g}  "
              f"std {arr.std():.4g}  min {arr.min():+.4g}  max {arr.max():+.4g}")
    print(f"  action parts    ({len(sel)}, {feats['n_chunks']}, {feats['part_dim']})   "
          f"B x vec(a_b), sigma_phi {float(feats['sigma_phi']):.4g} (median heuristic)")

    pools = []
    for t in TIMES:
        keep = pool_mask(t, sel.local.to_numpy(), sel.ep_len.to_numpy())
        row = {"t": t}
        row.update({g: int((keep & (sel.group == g).to_numpy()).sum()) for g in GROUPS})
        pools.append(row)
    print("  pooled steps by normalized time:")
    print("    " + pd.DataFrame(pools).to_string(index=False).replace("\n", "\n    "))


def summarise(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Fold-mean AUROCs, the per-fold difference-in-differences, and their t confidence intervals."""
    wide = df.pivot_table(index=["task", "model", "t", "fold"],
                          columns=["features", "comparison"], values="auroc")
    # The DiD is formed inside each fold, so its interval is over folds of the DiD itself
    # rather than assembled from the spreads of the four AUROCs.
    wide["did"] = ((wide[("obs+act", "failure")] - wide[("obs+act", "success")])
                   - (wide[("obs", "failure")] - wide[("obs", "success")]))
    wide.columns = ["__".join(p for p in c if p) if isinstance(c, tuple) else c
                    for c in wide.columns]
    stats = wide.groupby(level=["task", "model", "t"]).agg(["mean", "std", "count"])
    crit = student_t.ppf(0.975, k - 1)
    curves = pd.DataFrame(index=stats.index)
    for col in wide.columns:
        mean = stats[(col, "mean")]
        half = crit * stats[(col, "std")] / np.sqrt(stats[(col, "count")])
        curves[col], curves[col + "__lo"], curves[col + "__hi"] = mean, mean - half, mean + half
    return curves.reset_index()


def report_curves(curves: pd.DataFrame, model: str) -> None:
    """Print the two plotted quantities as a table per task."""
    for task, g in curves[curves.model == model].groupby("task", sort=False):
        print(f"\n--- {model} on {task}:  AUROC vs nominal (+- = half-width of the 95% CI over folds)")
        print(pd.DataFrame({
            "t": g.t,
            "succ|obs": g["obs__success"], "+-": g["obs__success__hi"] - g["obs__success"],
            "fail|obs": g["obs__failure"], "+-.": g["obs__failure__hi"] - g["obs__failure"],
            "gap|obs": g["obs__failure"] - g["obs__success"],
            "gap|obs+act": g["obs+act__failure"] - g["obs+act__success"],
            "DiD": g["did"], "+-..": g["did__hi"] - g["did"],
        }).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        signif = int(((g["did__lo"] > 0) | (g["did__hi"] < 0)).sum())
        print(f"  obs-only gap (failure - success) averages "
              f"{(g['obs__failure'] - g['obs__success']).mean():+.4f};  "
              f"DiD averages {g['did'].mean():+.4f}, CI excludes 0 at {signif}/{len(g)} times")


# ---------------------------------------------------------------------------- the plots
#: Chart chrome and the two mandated series hues (dataviz reference palette). Green/red sit
#: in the 6-8 CVD band, so identity is carried by dash pattern, marker and end label too.
_SURFACE, _INK, _SECOND, _MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
_GRID, _AXIS = "#e1e0d9", "#c3c2b7"
_GREEN, _RED, _BLUE = "#008300", "#e34948", "#2a78d6"
_SERIES = (("obs__success", "test success", _GREEN, "-", "o"),
           ("obs__failure", "test failure", _RED, "--", "s"))


def _panel(ax, task: str, ylabel: str, bottom: bool, xmax: float) -> None:
    """Recessive grid, hairline axes and shared x treatment for one subplot."""
    ax.set_facecolor(_SURFACE)
    ax.set_title(task, color=_INK, fontsize=11, loc="left", pad=6)
    ax.set_ylabel(ylabel, color=_SECOND, fontsize=9.5)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.set_xlim(0.0, xmax)
    ax.set_xticks(np.arange(0, 6) / 5.0)
    if bottom:
        ax.set_xlabel("normalized episode time t   (the pool is every step up to t)",
                      color=_SECOND, fontsize=10)


def _empty_panel(ax, task: str, reason: str, bottom: bool, xmax: float) -> None:
    """An out-of-scope task keeps its slot in the 5-panel layout, annotated with why."""
    _panel(ax, task, "", bottom, xmax)
    ax.set_yticks([])
    ax.text(0.5, 0.5, f"out of scope -- {reason}", transform=ax.transAxes, ha="center",
            va="center", color=_MUTED, fontsize=10, style="italic")


def plot_separability(curves: pd.DataFrame, model: str, scope: dict, stem: pathlib.Path) -> None:
    """Plot 1: observation-only separability of nominal from test success and test failure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(len(ALL_TASKS), 1, figsize=(8.0, 14.0), sharex=True)
    for ax, task in zip(axes, ALL_TASKS):
        bottom = task == ALL_TASKS[-1]
        if scope.get(task) is not None:
            _empty_panel(ax, task, scope[task], bottom, 1.16)
            continue
        g = curves[(curves.model == model) & (curves.task == task)].sort_values("t")
        _panel(ax, task, "AUROC   (0.5 = chance)", bottom, 1.16)
        ax.axhline(0.5, color=_AXIS, linewidth=1.0)
        ends = [float(g[col].iloc[-1]) for col, *_ in _SERIES]
        # Direct end labels, pushed apart vertically: green vs red alone is not a
        # sufficient identity channel, and the two curves can end on top of each other.
        offsets = (8, -8) if ends[0] >= ends[1] else (-8, 8)
        for (col, label, color, dash, marker), end, dy in zip(_SERIES, ends, offsets):
            ax.fill_between(g.t, g[col + "__lo"], g[col + "__hi"], color=color, alpha=0.13,
                            linewidth=0)
            ax.plot(g.t, g[col], color=color, linewidth=2.0, linestyle=dash, marker=marker,
                    markersize=4.5, markeredgecolor=_SURFACE, markeredgewidth=1.0, label=label)
            ax.annotate(label, (g.t.iloc[-1], end), xytext=(5, dy), va="center",
                        textcoords="offset points", color=color, fontsize=8.5)
        lo = float(g[[c[0] + "__lo" for c in _SERIES]].to_numpy().min())
        hi = float(g[[c[0] + "__hi" for c in _SERIES]].to_numpy().max())
        ax.set_ylim(min(0.45, lo - 0.03), max(1.0, hi + 0.03))
    handles = [Line2D([], [], color=c, linestyle=d, marker=m, markersize=4.5, linewidth=2.0,
                      label=lab) for _, lab, c, d, m in _SERIES]
    _finish(fig, f"How separable is FIPER's nominal data from its test episodes?   ({model})",
            f"Q1 / Q2, observation embeddings only: AUROC of an oracle {model} that has the "
            "success/failure labels, cross-validated over episodes; the band is a 95% t "
            "interval over the folds. 0.5 means the oracle cannot tell the two apart -- which "
            "is the regime the one-class FIPER benchmark has to work in.", stem, handles)


def plot_did(curves: pd.DataFrame, model: str, scope: dict, stem: pathlib.Path) -> None:
    """Plot 2: what the action channels add, as a difference-in-differences over time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(ALL_TASKS), 1, figsize=(8.0, 14.0), sharex=True)
    first = next((t for t in ALL_TASKS if scope.get(t) is None), None)
    for ax, task in zip(axes, ALL_TASKS):
        bottom = task == ALL_TASKS[-1]
        if scope.get(task) is not None:
            _empty_panel(ax, task, scope[task], bottom, 1.04)
            continue
        g = curves[(curves.model == model) & (curves.task == task)].sort_values("t")
        _panel(ax, task, "DiD in AUROC", bottom, 1.04)
        ax.axhline(0.0, color=_AXIS, linewidth=1.0)
        ax.fill_between(g.t, g["did__lo"], g["did__hi"], color=_BLUE, alpha=0.13, linewidth=0)
        ax.plot(g.t, g["did"], color=_BLUE, linewidth=2.0, marker="o", markersize=4.5,
                markeredgecolor=_SURFACE, markeredgewidth=1.0)
        span = float(np.nanmax(np.abs(g[["did__lo", "did__hi"]].to_numpy())))
        ax.set_ylim(-1.25 * span, 1.25 * span)
        if task == first:  # one orientation label, on the topmost panel that has data
            ax.annotate("above 0: the action channels widen the failure-minus-success gap",
                        (0.015, 0.95), xycoords="axes fraction", color=_MUTED, fontsize=8.5,
                        va="top")
    _finish(fig, f"Do the action channels separate failures any better?   ({model})",
            "Q3, difference-in-differences:  (failure - success | obs+act) - (failure - "
            "success | obs), formed within each CV fold and averaged; the band is a 95% t "
            "interval over the folds. y = 0 is the observation-only baseline of plot 1, and "
            "the success curve controls for calibration-vs-test shift unrelated to failure.",
            stem)


def _finish(fig, title: str, subtitle: str, stem: pathlib.Path, handles=None) -> None:
    """Header block (title, wrapped subtitle, optional legend), spacing, and the png/pdf pair."""
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
    if handles is not None:
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, y), ncol=len(handles),
                   frameon=False, fontsize=9, labelcolor=_SECOND, handlelength=2.6)
        y -= 20.0 / height
    fig.tight_layout(rect=(0.0, 0.0, 1.0, y - 6.0 / height))
    fig.subplots_adjust(hspace=0.30)
    for ext in ("png", "pdf"):
        path = stem.with_suffix("." + ext)
        fig.savefig(path, dpi=150, facecolor=_SURFACE)
        print(f"  -> {os.path.relpath(path, ROOT_DIR)}")
    plt.close(fig)


# ----------------------------------------------------------------------------- the main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="separability", description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--folds", type=int, default=5, help="k for the CV over episodes")
    parser.add_argument("--cap", type=int, default=CAP, help="steps kept per group per task")
    parser.add_argument("--rff", type=int, default=M_RFF, help="Random Fourier Features in phi")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=-1, help="parallel (time, fold) workers")
    parser.add_argument("--force", action="store_true", help="ignore the cache")
    args = parser.parse_args(argv)

    started = time.time()
    print(f"normalized times {TIMES.tolist()}\nk = {args.folds} folds over episodes, cap "
          f"{args.cap} steps/group, M = {args.rff} RFF, seed {args.seed}, models {args.models}")

    scope, frames = {}, []
    for task in ALL_TASKS:
        if task not in args.tasks:
            scope[task] = "not requested on this run"
            continue
        counts = episodes(load_metadata(task)).groupby("group").size()
        thin = [g for g in GROUPS if counts.get(g, 0) < args.folds]
        if thin:
            scope[task] = f"fewer than k={args.folds} episodes in {', '.join(thin)}"
            print(f"\n===== {task}: skipped -- {scope[task]}")
            continue
        scope[task] = None
        frames.append(run_task(task, args.seed, args.folds, args.cap, args.rff, args.models,
                               args.jobs, args.force))

    if not frames:
        print("nothing in scope", file=sys.stderr)
        return 1
    df = pd.concat(frames, ignore_index=True)
    print(f"\n=== AUROC table {df.shape}: {len(df)} fits = {df.task.nunique()} tasks x "
          f"{len(TIMES)} times x {args.folds} folds x {len(FEATURE_SETS)} feature sets x "
          f"{len(COMPARISONS)} comparisons x {len(args.models)} models")
    print(df.groupby(["model", "features", "comparison"])["auroc"]
          .agg(["count", "mean", "std", "min", "max"]).to_string(float_format=lambda v: f"{v:.4f}"))
    print(df.groupby("task")[["n_train", "n_test", "n_test_pos"]]
          .agg(["min", "max"]).to_string())
    if df.auroc.isna().any():
        print(f"  !! {int(df.auroc.isna().sum())} folds held a single class and scored NaN")

    curves = summarise(df, args.folds)
    print(f"\n=== curve table {curves.shape}, columns {list(curves.columns)}")

    for model in args.models:
        report_curves(curves, model)
        print(f"\n=== plots, {model}")
        plot_separability(curves, model, scope, HERE / f"separability_obs__{model}")
        plot_did(curves, model, scope, HERE / f"separability_did__{model}")

    print(f"\ntotal wall clock: {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
