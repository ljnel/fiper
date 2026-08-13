"""Compute cost (Q1) and data efficiency (Q2) of the FIPER baselines and two kern_cd methods -- ``specs/experiments/efficiency.md``.

    python my_experiments/efficiency/efficiency.py --run              # fill the cache
    python my_experiments/efficiency/efficiency.py --plot             # both figures

One row per (task, method, number of calibration episodes): the wall clock of every
stage of the real FIPER evaluation, plus the metrics that evaluation produces. Q1 reads
the full-data rows, Q2 reads the whole grid.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import pathlib
import shutil
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT_DIR = str(HERE.parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from my_methods.base import REGISTRY, discover  # noqa: E402
from my_methods.methods.kern_cd_flat import KernCDFlat  # noqa: E402
from my_methods.methods.kern_cd_obs import KernCDObs  # noqa: E402
from my_methods.run import _RegistryEvaluationManager  # noqa: E402
from rnd import RNDTrainer  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402
from shared_utils.hydra_utils import load_config  # noqa: E402
from shared_utils.utility_functions import get_required_tensors, set_seed  # noqa: E402
from tasks import TaskManager  # noqa: E402

CACHE_DIR = HERE / "cache"
#: RND checkpoints and eval artefacts are written here, never into ``data/``.
SCRATCH_DIR = CACHE_DIR / "scratch"
BASE_CONFIG_PATH = os.path.join(ROOT_DIR, "configs")
BASE_DATA_PATH = os.path.join(ROOT_DIR, "data")

ALL_TASKS = ["push_t", "sorting", "stacking", "pretzel", "push_chair"]

#: Every method of the FIPER paper, plus the two the spec keeps from specs/methods.md.
BASELINES = ["entropy", "tc", "similarity", "logpzo", "rnd_a", "rnd_oe"]
KERN_CD = ["kern_cd_obs", "kern_cd_flat"]
METHODS = BASELINES + KERN_CD
RND_MODELS = ["rnd_oe", "rnd_a"]

#: FIPER's headline detector is the logical AND of RND-OE and ACE (configs/default.yaml).
COMBO_NAME = "rnd_oe_and_entropy"
COMBO = {
    "m1": {"name": "rnd_oe", "quantiles": None, "window_sizes": None},
    "m2": {"name": "entropy", "quantiles": None, "window_sizes": None},
    "operation": "and",
}

LABEL = {
    "entropy": "ACE",
    "tc": "STAC",
    "similarity": "PCA-kmeans",
    "logpzo": "logpZO",
    "rnd_a": "RND-A",
    "rnd_oe": "RND-OE",
    "rnd_oe_and_entropy": "FIPER",
    "kern_cd_obs": "KernCD-Obs",
    "kern_cd_flat": "KernCD-Flat",
}
#: Episode-LOO variants, defined below. Same method and same kernel, calibrated by holding
#: out whole episodes, so each inherits the label and colour of the step-LOO original.
EPLOO_OF = {"kern_cd_obs": "kern_cd_obs_eploo", "kern_cd_flat": "kern_cd_flat_eploo"}
EPLOO_METHODS = list(EPLOO_OF.values())
LABEL.update({new: LABEL[old] for old, new in EPLOO_OF.items()})

#: Plot order: published baselines, then ours.
PLOT_ORDER = ["entropy", "tc", "similarity", "logpzo", "rnd_a", "rnd_oe", COMBO_NAME] + KERN_CD
#: Q2 draws only these: ours against FIPER's own detector. Nine curves per panel over five
#: panels is unreadable, and every method's numbers are in the summary csv either way.
Q2_METHODS = [COMBO_NAME] + EPLOO_METHODS

TASK_LABEL = {"push_t": "Push-T", "sorting": "Sorting", "stacking": "Stacking",
              "pretzel": "Pretzel", "push_chair": "Push-Chair"}

#: Calibration episodes per point. push_chair's episodes are ~5 steps, so two of them
#: leave RND's 10% validation split empty; its grid starts at three.
TRAIN_EPISODES = {
    "push_t": [5, 10, 20, 35, 50],
    "sorting": [5, 10, 20, 35, 50],
    "stacking": [5, 10, 20, 35, 50],
    "pretzel": [2, 4, 6, 8, 10],
    "push_chair": [3, 5, 7, 10],
}

#: Timed stages. "train" is RND's training loop, run outside the eval class; "fit" is
#: FIPER's ``_execute_preprocessing`` (logpZO's flow, PCA-kmeans' clustering, kern_cd's
#: Cholesky); "load" is model construction and checkpoint loading. The two "window"
#: stages are the moving-window sums FIPER applies to a pass's scores afterwards -- the
#: same 17 windows for every method, and no part of what a method costs.
STAGES = ("load", "train", "fit", "score_calibration", "score_test", "thresholds", "metrics",
          "window_calibration", "window_test")


# --------------------------------------------------------------------------- timing
def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def timed(store: dict, key: str):
    """Accumulate wall clock (and process CPU time) of a stage into ``store``."""
    _sync()
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        yield
    finally:
        _sync()
        store[key] = store.get(key, 0.0) + time.perf_counter() - t0
        store[key + "__cpu"] = store.get(key + "__cpu", 0.0) + time.process_time() - c0


def instrument(obj, store: dict) -> None:
    """Wrap one eval object's stages so each lands in ``store``; drop its pickle dump.

    Wrapping the *instance* rather than ``BaseEvalClass`` is what makes the kern_cd
    numbers honest: those methods score a whole subset ahead of the per-step loop in an
    adapter override, which a patch on the base class would time only the inside of.
    """
    for attr, key in (("_execute_preprocessing", "fit"), ("_get_thresholds", "thresholds"),
                      ("_get_metrics", "metrics")):
        original = getattr(obj, attr)

        def wrapper(*args, __original=original, __key=key, **kwargs):
            with timed(store, __key):
                return __original(*args, **kwargs)

        setattr(obj, attr, wrapper)

    original_process, current = obj._process_rollouts, {"subset": ""}

    def process_rollouts(subset: str):
        current["subset"] = subset
        with timed(store, f"score_{subset}"):
            return original_process(subset)

    obj._process_rollouts = process_rollouts
    original_window = obj._apply_window_size

    def apply_window_size(*args, **kwargs):
        with timed(store, f"window_{current['subset']}"):
            return original_window(*args, **kwargs)

    obj._apply_window_size = apply_window_size
    # eval_results.pkl is up to ~300 MB per (task, method) and nothing here reads it.
    obj._save_pickle = lambda *args, **kwargs: None


class _TimedManager(_RegistryEvaluationManager):
    """``EvaluationManager`` that times every stage of every method it evaluates."""

    def __init__(self, *args, timings: dict, **kwargs):
        self.timings = timings
        super().__init__(*args, **kwargs)

    def _get_method_eval_class(self, method_name: str, cfg):
        store = self.timings.setdefault(method_name, {})
        with timed(store, "load"):
            obj = super()._get_method_eval_class(method_name, cfg)
        instrument(obj, store)
        return obj


# ------------------------------------------- leave-one-episode-out calibration scores
class EpisodeLOOMixin:
    """kern_cd calibrated by holding out whole episodes rather than single steps.

    The scores that set FIPER's threshold come from ``KernCDMethod._loo_scores``, which
    leaves one *step* out. Steps within an episode are strongly correlated, so a held-out
    step keeps its own temporal neighbours in the fit set and scores low however small
    that set is, while a test step comes from an episode the fit set has never seen. The
    threshold is therefore optimistic, and increasingly so as the calibration split
    shrinks -- which is exactly the regime a data-efficiency curve measures.

    Holding out the whole episode closes that leak, and stays closed-form. With
    ``M = K + lam*m*I``, ``B = M^-1``, and ``E`` the block of one episode,

        s_i = K_ii + v_i' m_i + v_i' P^-1 v_i,   P = B_EE,  m_i = M[E, i],
        v_i = e_i - P m_i,

    which is the ordinary ``1/B_ii - lam*m`` when the block is a single step.
    """

    def _blocks(self) -> list[tuple[int, int]]:
        """Half-open [start, end) row ranges of the fit set, one per fitted episode."""
        lengths = np.asarray(self._fit_episode_lengths)[np.asarray(self._fit_episode_kept)]
        bounds = np.concatenate([[0], np.cumsum(lengths)])
        return list(zip(bounds[:-1], bounds[1:]))

    def _loo_scores(self) -> np.ndarray:
        Z = np.asarray(self.model.data)
        m = len(Z)
        lam_m = self.model.lam_ * m
        L_inv = solve_triangular(self.model.L, np.eye(m), lower=True)
        B = L_inv.T @ L_inv
        out = np.empty(m)
        for a, b in self._blocks():
            M_EE = self.model.kernel(Z[a:b]) + lam_m * np.eye(b - a)
            P = B[a:b, a:b]
            V = np.eye(b - a) - P @ M_EE
            out[a:b] = (np.diag(M_EE) - lam_m
                        + np.einsum("ki,ki->i", V, M_EE)
                        + np.einsum("ki,ki->i", V, np.linalg.solve(P, V)))
        return out

    def _validate_loo(self, Z: np.ndarray) -> None:
        """Refitting with each episode left out (same lam*m ridge) must match the closed form."""
        m = len(Z)
        lam_m = self.model.lam_ * m
        K = self.model.kernel(Z)
        brute = np.empty(m)
        for a, b in self._blocks():
            keep = np.r_[0:a, b:m]
            M = K[np.ix_(keep, keep)] + lam_m * np.eye(len(keep))
            for i in range(a, b):
                k_vec = K[keep, i]
                brute[i] = K[i, i] - k_vec @ np.linalg.solve(M, k_vec)
        max_diff = np.abs(brute - self.loo_scores).max()
        assert np.allclose(brute, self.loo_scores, atol=1e-6, rtol=1e-5), (
            f"episode-LOO closed form mismatch vs brute force: max abs diff {max_diff:.2e}"
        )


class KernCDObsEpisodeLOO(EpisodeLOOMixin, KernCDObs):
    name = "kern_cd_obs_eploo"


class KernCDFlatEpisodeLOO(EpisodeLOOMixin, KernCDFlat):
    name = "kern_cd_flat_eploo"


# ----------------------------------------------------------------------------- data
def restrict_calibration(dataset, n_episodes: int, seed: int) -> dict:
    """Keep ``n_episodes`` calibration episodes; the test split is left untouched.

    Dropped episodes leave the calibration mask without entering the test mask, so every
    point of a data-efficiency curve is scored on exactly the same test episodes. The
    subsets are nested prefixes of one seeded permutation per success label, so the
    success ratio of the split (which only push_t has) is held roughly fixed.
    """
    md = dataset.data["metadata"]
    calib = np.asarray(md["calibration_rollout_labels"], dtype=bool)
    successful = np.asarray(md["successful_rollout_labels"], dtype=bool)
    rng = np.random.default_rng(seed)
    keep = []
    n_total = int(calib.sum())
    n_ok = int((calib & successful).sum())
    # At least one successful episode: FIPER's thresholds are quantiles over those alone.
    n_keep_ok = max(1, int(round(n_episodes * n_ok / n_total))) if n_ok else 0
    for mask, n_keep in ((calib & successful, n_keep_ok), (calib & ~successful, n_episodes - n_keep_ok)):
        idx = np.where(mask)[0]
        if len(idx) == 0 or n_keep <= 0:
            continue
        order = rng.permutation(len(idx))[: min(n_keep, len(idx))]
        keep.extend(idx[np.sort(order)].tolist())

    new = np.zeros_like(calib)
    new[np.asarray(sorted(keep), dtype=int)] = True
    md["calibration_rollout_labels"] = new
    lengths = np.asarray(md["episode_end_indices"]) - np.asarray(md["episode_start_indices"])
    return {
        "train_episodes": int(new.sum()),
        "train_episodes_successful": int((new & successful).sum()),
        "train_steps": int(lengths[new].sum()),
        "test_episodes": int(np.asarray(md["test_rollout_labels"]).sum()),
        "test_steps": int(lengths[np.asarray(md["test_rollout_labels"], dtype=bool)].sum()),
    }


def build_dataset(task: str, methods: list[str], device: str):
    """The processed rollout dataset, with the union of the tensors the methods declare."""
    cfg = load_config("task", task, return_only_subdict=False)
    builtins = [m for m in methods if m not in REGISTRY]
    required, optional = get_required_tensors(builtins, BASE_CONFIG_PATH) if builtins else ([], [])
    for name in methods:
        if name in REGISTRY:
            required.extend(REGISTRY[name].tensors)
            optional.extend(REGISTRY[name].optional_tensors)
    required = list(dict.fromkeys(required))
    optional = [t for t in dict.fromkeys(optional) if t not in required]

    taskmanager = TaskManager(
        cfg, task, BASE_CONFIG_PATH, os.path.join(BASE_DATA_PATH, task),
        required_tensors=required, optional_tensors=optional, device=device,
    )
    return cfg, taskmanager.get_rollout_dataset(load_dataset_if_exists=False)


# -------------------------------------------------------------------------- metrics
def metric_rows(method_results: dict) -> pd.DataFrame:
    """Flatten ``test_metrics`` into one row per (threshold style, quantile, window)."""
    rows = []
    for style, by_quantile in method_results["test_metrics"].items():
        for quantile, by_window in by_quantile.items():
            for window, metrics in by_window.items():
                rows.append({"Threshold": str(style), "Quantile": float(quantile),
                             "Window": str(window), "TWA": float(metrics["TWA"]),
                             "Accuracy": float(metrics["balanced_accuracy"]),
                             "Det. Time": float(metrics["avg_detection_time"]),
                             "TPR": float(metrics["TPR"]), "TNR": float(metrics["TNR"])})
    return pd.DataFrame(rows)


def best_configuration(rows: pd.DataFrame) -> dict:
    """Quantile-averaged metrics of the best (threshold style, window), selected by TWA.

    Same footing as configs/results/base.yaml and my_methods/paper_table.py: average over
    quantiles first, then take the best window and threshold style.
    """
    grouped = rows.groupby(["Threshold", "Window"], as_index=False)[
        ["TWA", "Accuracy", "Det. Time", "TPR", "TNR"]
    ].mean()
    best = grouped.loc[grouped["TWA"].idxmax()]
    return {"TWA": float(best["TWA"]), "Accuracy": float(best["Accuracy"]),
            "Det. Time": float(best["Det. Time"]), "TPR": float(best["TPR"]),
            "TNR": float(best["TNR"]), "Window": str(best["Window"]),
            "ThresholdStyle": str(best["Threshold"])}


# ------------------------------------------------------------------------ one point
def run_point(task: str, n_episodes: int, seed: int, methods: list[str], with_combo: bool = True) -> pd.DataFrame:
    """Evaluate every method on one (task, training-set size) point and time every stage."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    cfg, dataset = build_dataset(task, methods, device)
    sizes = restrict_calibration(dataset, n_episodes, seed)
    print(f"  {sizes['train_episodes']} train episodes "
          f"({sizes['train_episodes_successful']} successful, {sizes['train_steps']} steps), "
          f"{sizes['test_episodes']} test episodes ({sizes['test_steps']} steps)")

    scratch = SCRATCH_DIR / task
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)

    timings: dict[str, dict] = {}
    # RND is the only method whose training does not run inside its eval class. It is
    # retrained here rather than read from data/{task}/rnd_models so that the training
    # cost is measured and so that it sees only this point's calibration episodes.
    for name in [m for m in methods if m in RND_MODELS]:
        trainer = RNDTrainer(BASE_CONFIG_PATH, str(scratch), dataset, device=device, task_cfg=cfg, seed=seed)
        with timed(timings.setdefault(name, {}), "train"):
            trainer.train([name])

    manager = _TimedManager(BASE_CONFIG_PATH, str(scratch), dataset, device=device, seed=seed,
                            timings=timings)
    # One method at a time: at the small end of the grid FIPER's threshold machinery can
    # fail outright (a calibration split of two successful episodes gives a conformal band
    # that is not positive), and that must cost one row rather than the whole point.
    results = {}
    for name in methods:
        try:
            results[name] = manager.evaluate([name], combine_methods=False)[name]
        except Exception as error:  # noqa: BLE001 -- any method-side failure is one lost row
            print(f"  !! {name} failed: {type(error).__name__}: {error}", flush=True)

    if with_combo and {"rnd_oe", "entropy"} <= set(results):
        with timed(timings.setdefault(COMBO_NAME, {}), "metrics"):
            results = manager._combine_two_methods(COMBO, results)

    rows = []
    for name in [m for m in methods if m in results] + ([COMBO_NAME] if COMBO_NAME in results else []):
        store = timings.get(name, {})
        row = {"task": task, "method": name, "seed": seed, **sizes}
        row.update({stage: store.get(stage, 0.0) for stage in STAGES})
        row.update({stage + "__cpu": store.get(stage + "__cpu", 0.0) for stage in STAGES})
        row.update(best_configuration(metric_rows(results[name])))
        rows.append(row)

    del results, manager
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    # RND checkpoints are ~130 MB each and are of no use once the timings are recorded.
    shutil.rmtree(scratch, ignore_errors=True)
    return pd.DataFrame(rows)


def probe_single_step(task: str, seed: int, methods: list[str], n_steps: int = 200) -> pd.DataFrame:
    """Latency of a kern_cd score computed one step at a time, without batching.

    The kern_cd methods implement ``score_subset``, so the evaluation above scores a whole
    pass in blocks and their online cost is amortised over it. A deployed detector scores
    one step at a time, so this re-times the same fitted model that way: embed one step's
    chunk batch, then one triangular solve against the m x m Cholesky factor.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    cfg, dataset = build_dataset(task, methods, device)
    manager = _TimedManager(BASE_CONFIG_PATH, str(SCRATCH_DIR / task), dataset, device=device,
                            seed=seed, timings={})
    rows = []
    for name in methods:
        obj = manager._get_method_eval_class(name, manager._load_config(name))
        obj._execute_preprocessing()
        obj.impl.subset = "test"
        data = obj._get_subset("test")
        keys = [k for k in ("obs_embeddings", "action_preds") if k in data]
        steps = np.linspace(0, len(data[keys[0]]) - 1, n_steps, dtype=int)
        _sync()
        t0 = time.perf_counter()
        for i in steps:
            Z = obj.impl._features({k: data[k][i : i + 1] for k in keys})
            obj.impl.model.score(Z)
        _sync()
        rows.append({"task": task, "method": name, "seed": seed, "fit_points": len(obj.impl.model.data),
                     "single_step_ms": 1e3 * (time.perf_counter() - t0) / len(steps)})
        print(f"  {name}: {rows[-1]['single_step_ms']:.3f} ms/step unbatched", flush=True)
    return pd.DataFrame(rows)


def point_path(task: str, n_episodes: int, seed: int, tag: str = "") -> pathlib.Path:
    return CACHE_DIR / f"{tag}{task}__n{n_episodes:03d}__seed{seed}.csv"


def run_grid(tasks: list[str], methods: list[str], seed: int, full_only: bool, overwrite: bool,
             tag: str = "") -> None:
    """Run every (task, training-set size) point that is not already cached.

    ``tag`` gives a run its own cache namespace, so a subset of methods can be re-run
    without disturbing the points already measured for the others.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        grid = TRAIN_EPISODES[task][-1:] if full_only else TRAIN_EPISODES[task]
        for n_episodes in grid:
            path = point_path(task, n_episodes, seed, tag)
            if path.exists() and not overwrite:
                print(f"[cached] {path.name}")
                continue
            print(f"\n=== {task}, {n_episodes} calibration episodes, seed {seed} ===", flush=True)
            t0 = time.perf_counter()
            df = run_point(task, n_episodes, seed, methods)
            if df.empty:
                print("  !! every method failed on this point -- not cached", flush=True)
                continue
            df.to_csv(path, index=False)
            print(f"  -> {path.name}  ({time.perf_counter() - t0:.0f} s)", flush=True)


def load_cache(seed: int) -> pd.DataFrame:
    """Every cached point, with the derived cost columns Q1 and Q2 plot."""
    frames = [pd.read_csv(p) for p in sorted(CACHE_DIR.glob(f"*__seed{seed}.csv"))
              if not p.name.startswith("probe__")]
    if not frames:
        raise SystemExit("empty cache -- run with --run first")
    df = pd.concat(frames, ignore_index=True)
    # FIPER's combination is derived from two methods it has to run first, so its cost is
    # theirs plus the cost of the combination itself.
    parts = df[df.method.isin(["rnd_oe", "entropy"])].groupby(["task", "train_episodes"], as_index=False)[
        list(STAGES)].sum()
    combo = df.method == COMBO_NAME
    df = df.merge(parts, on=["task", "train_episodes"], how="left", suffixes=("", "__parts"))
    for stage in STAGES:
        df.loc[combo, stage] = df.loc[combo, stage] + df.loc[combo, stage + "__parts"]
    df = df.drop(columns=[s + "__parts" for s in STAGES])

    # Offline: fitting the model and scoring the calibration episodes the threshold is
    # read off. FIPER's sweep over 17 windows x 10 quantiles x 3 threshold styles is a
    # benchmark cost, not a method cost, so the window sums and "thresholds" come out.
    df["fit_s"] = df[["load", "train", "fit"]].sum(axis=1)
    df["offline_s"] = df["fit_s"] + df["score_calibration"] - df["window_calibration"]
    df["online_ms"] = 1e3 * (df["score_test"] - df["window_test"]) / df["test_steps"]
    df["label"] = df.method.map(LABEL)

    probes = [pd.read_csv(p) for p in sorted(CACHE_DIR.glob(f"probe__*__seed{seed}.csv"))]
    if probes:
        probe = pd.concat(probes, ignore_index=True)[["task", "method", "single_step_ms"]]
        df = df.merge(probe, on=["task", "method"], how="left")
    else:
        df["single_step_ms"] = np.nan
    return df


# ---------------------------------------------------------------------------- plots
#: Tasks are an ordered variable here -- they differ by dataset size -- so they take an
#: ordinal ramp of one hue rather than categorical slots: light = small, dark = large.
TASK_COLOR = {"push_chair": "#86b6ef", "pretzel": "#5598e7", "push_t": "#2a78d6",
              "sorting": "#1c5cab", "stacking": "#0d366b"}
#: Categorical slots 1-8, in the fixed order that keeps adjacent pairs colourblind-safe.
#: FIPER's detector is the AND of two of these rather than a method in its own right, so
#: it takes neutral ink and a dashed line instead of a ninth hue.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
COMBO_STYLE = {"color": "#52514e", "linestyle": (0, (4, 2)), "marker": "h"}
INK, SECOND, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e0dfdb"


def method_style() -> dict:
    """Colour, marker and dash of every method; identity is never colour alone."""
    styles, slot = {}, 0
    for name in PLOT_ORDER:
        if name == COMBO_NAME:
            styles[name] = dict(COMBO_STYLE)
            continue
        styles[name] = {"color": SERIES[slot], "marker": MARKERS[slot], "linestyle": "-"}
        slot += 1
    # Recalibrating a method does not make it a different entity, so it keeps its colour.
    styles.update({new: dict(styles[old]) for old, new in EPLOO_OF.items()})
    return styles


def paper_style() -> None:
    """Matplotlib defaults for a single-column figure."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": SECOND, "ytick.color": SECOND,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "legend.frameon": False, "figure.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
    })


def _save(fig, stem: pathlib.Path) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix("." + ext), dpi=400)
        print(f"  -> {os.path.relpath(stem.with_suffix('.' + ext), ROOT_DIR)}")


def plot_cost(df: pd.DataFrame, stem: pathlib.Path) -> None:
    """Q1: offline and online cost of every method, one dot per task."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    paper_style()
    full = df[df.train_episodes == df.groupby("task").train_episodes.transform("max")]
    methods = [m for m in PLOT_ORDER if m in set(full.method)]
    tasks = [t for t in TASK_COLOR if t in set(full.task)]
    y_of = {m: len(methods) - 1 - i for i, m in enumerate(methods)}

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.45), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    for ax, column, xlabel in ((axes[0], "offline_s", "offline cost (s)"),
                               (axes[1], "online_ms", "online cost (ms / rollout step)")):
        for method in methods:
            g = full[full.method == method]
            y = y_of[method]
            ax.plot([g[column].min(), g[column].max()], [y, y], color=GRID, linewidth=2.5,
                    solid_capstyle="round", zorder=1)
            for task in tasks:
                v = g[g.task == task][column]
                if len(v):
                    ax.plot(float(v.iloc[0]), y, marker="o", markersize=3.4, zorder=3,
                            color=TASK_COLOR[task], markeredgecolor="white", markeredgewidth=0.4)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        # The rule marks where the published baselines end and specs/methods.md begins.
        ax.axhline(len(KERN_CD) - 0.5, color=MUTED, linewidth=0.5, linestyle=(0, (3, 3)))

    axes[0].set_yticks(range(len(methods)), [LABEL[m] for m in reversed(methods)], color=INK)
    axes[0].set_ylim(-0.7, len(methods) - 0.3)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=3.4, color=TASK_COLOR[t],
                      label=f"{TASK_LABEL[t]}") for t in tasks]
    # Ordered light -> dark, so the legend row itself says "colour depth = dataset size".
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, 0.98), ncol=5,
               columnspacing=0.8, handletextpad=0.2, labelcolor=SECOND,
               title="task, by dataset size", title_fontsize=6.5)
    _save(fig, stem)
    plt.close(fig)


def plot_data_efficiency(df: pd.DataFrame, stem: pathlib.Path) -> None:
    """Q2: TWA against the number of training episodes, one panel per task."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

    paper_style()
    # Styles are assigned over the full method list, so dropping series here never
    # repaints the ones that remain -- a curve keeps its colour across both figures.
    styles = method_style()
    methods = [m for m in Q2_METHODS if m in set(df.method)]
    tasks = [t for t in ALL_TASKS if t in set(df.task)]

    # No shared x: the tasks have different calibration splits (2-10 episodes on the
    # real-world tasks, 5-50 on the simulated ones), so each panel carries its own ticks.
    fig, axes = plt.subplots(len(tasks), 1, figsize=(3.4, 4.8))
    fig.subplots_adjust(top=0.94, hspace=0.55)
    for ax, task in zip(np.atleast_1d(axes), tasks):
        g = df[df.task == task]
        for method in methods:
            h = g[g.method == method].sort_values("train_episodes")
            ax.plot(h.train_episodes, h.TWA, linewidth=1.1, markersize=2.6,
                    markeredgecolor="white", markeredgewidth=0.3, **styles[method])
        ax.set_title(f"{TASK_LABEL[task]}", loc="left", pad=2.5, color=INK)
        ax.set_ylabel("TWA", color=INK, labelpad=2)
        ax.grid(color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.margins(x=0.04, y=0.10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ticks = sorted(g.train_episodes.unique())
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", length=0)
    np.atleast_1d(axes)[-1].set_xlabel("training episodes  (FIPER's calibration split)", color=INK)

    handles = [Line2D([], [], linewidth=1.1, markersize=2.6, label=LABEL[m], **styles[m])
               for m in methods]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.955), ncol=3,
               columnspacing=1.4, handletextpad=0.4, labelcolor=SECOND)
    _save(fig, stem)
    plt.close(fig)


def write_summary(df: pd.DataFrame, path: pathlib.Path) -> pd.DataFrame:
    """Save the tidy per-point table and return the full-data slice Q1 reads."""
    columns = ["task", "method", "train_episodes", "train_steps", "test_episodes", "test_steps",
               "fit_s", "offline_s", "online_ms", "TWA", "Accuracy", "Det. Time",
               "Window", "ThresholdStyle", "single_step_ms", *STAGES]
    df[columns].sort_values(["task", "method", "train_episodes"]).to_csv(path, index=False)
    print(f"  -> {os.path.relpath(path, ROOT_DIR)}")
    return df[df.train_episodes == df.groupby("task").train_episodes.transform("max")]


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="fill the cache")
    parser.add_argument("--plot", action="store_true", help="draw both figures from the cache")
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full-only", action="store_true", help="only the full-data point (Q1)")
    parser.add_argument("--overwrite", action="store_true", help="re-run cached points")
    parser.add_argument("--probe", action="store_true",
                        help="time the kern_cd methods one step at a time (no batching)")
    parser.add_argument("--eploo", action="store_true",
                        help="run only the episode-LOO kern_cd variants, into their own cache")
    args = parser.parse_args(argv)

    if not (args.run or args.plot or args.probe or args.eploo):
        parser.error("nothing to do: pass --run, --eploo, --probe and/or --plot")

    if args.run:
        discover()
        run_grid(args.tasks, list(args.methods), args.seed, args.full_only, args.overwrite)
    if args.eploo:
        discover()
        run_grid(args.tasks, EPLOO_METHODS, args.seed, args.full_only, args.overwrite, tag="eploo__")
    if args.probe:
        discover()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for task in args.tasks:
            print(f"\n=== probe: {task} ===", flush=True)
            probe_single_step(task, args.seed, KERN_CD).to_csv(
                CACHE_DIR / f"probe__{task}__seed{args.seed}.csv", index=False)
    if args.plot:
        df = load_cache(args.seed)
        full = write_summary(df, HERE / "efficiency__summary.csv")
        plot_cost(df, HERE / "efficiency_cost")
        plot_data_efficiency(df, HERE / "efficiency_data")
        pivot = full.pivot_table(index="label", columns="task", values=["offline_s", "online_ms"])
        print("\nfull-data cost per task:")
        print(pivot.round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
