"""Time-and-memory profile of the kern_cd family -- ``specs/experiments/profiling.md``.

The five methods in ``my_methods/methods/`` share every line of code except ``_parts``,
and ``kern_cd_obs`` has no action path at all. So the question is not "how long does each
method take" but **how total cost splits into a shared floor and a per-method delta**. The
obs bar *is* that floor, measured rather than estimated, and every other method minus obs
is the exact cost of its action channel.

What this file does
-------------------
It runs the real FIPER evaluation (the same path ``my_methods/run.py`` takes) for the 5
methods on 2 tasks, with the ten seams of ``kern_cd_core.py`` instrumented by
monkey-patching -- nothing in ``my_methods/`` is edited, and the profiled code is byte-for-
byte the code that runs in an ordinary evaluation. Timings are **exclusive** (a seam is
charged only for the work it does outside its instrumented children), so the segments sum
exactly and nothing is double counted; ``torch.cuda.synchronize()`` brackets every seam, so
GPU asynchrony cannot smear the boundaries.

Measure fine, display coarse: every seam is written to the raw CSV, only the four-segment
roll-up is plotted. The residual (evaluate wall clock minus the sum of the seams) is drawn
as its own segment rather than hidden -- a large one means the segmentation is wrong.

Two things the numbers forced. The sweep over all methods is run several times and the
first is discarded, because whichever method goes first pays the process's one-off warm-up
and the method that goes first is the obs baseline, i.e. the reference floor itself. And
Deliverable 1 is drawn twice on two scales: with the harness bar in, the methods are 3% of
the width, which answers "is this worth optimising"; with it out, the segments are legible,
which answers "where does the time go".

Running it
----------
    streamlit run my_experiments/profiling/profiling.py        # app; "Run profile" fills the cache
    python my_experiments/profiling/profiling.py --run         # headless (re)generation of the cache

Results are cached as CSVs under ``my_experiments/profiling/cache/`` and are not
regenerated on repeated app runs. ``--run --force`` overwrites; the app has the same button.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import os
import pathlib
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = str(HERE.parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CACHE_DIR = HERE / "cache"
SEAMS_CSV = CACHE_DIR / "seams.csv"
META_CSV = CACHE_DIR / "meta.csv"

# ---------------------------------------------------------------------------- scope

#: The two profiled tasks. B / H / d_o / episode counts are re-measured at run time; the
#: values here are the spec's, kept only so the app can show a table before a run exists.
TASKS = {
    "pretzel": dict(kind="real-world", d_o=512, B=30, H=16, calib_ep=10, test_ep=20),
    "push_t": dict(kind="simulation", d_o=64, B=256, H=16, calib_ep=50, test_ep=300),
}
METHODS = ["kern_cd_obs", "kern_cd_disp", "kern_cd_sum", "kern_cd_flat", "kern_cd_sig"]
FLOOR_METHOD = "kern_cd_obs"
SEED = 0

#: FIPER's threshold grid, from ``configs/eval/base.yaml`` -- 17 windows x 10 quantiles x
#: 3 threshold styles. Quoted in the harness-context caption.
THRESHOLD_GRID = (17, 10, 3)

# ------------------------------------------------------------------- segment roll-up

SEG_FEAT_FIT = "Features (fit)"
SEG_EST = "Estimator fit"
SEG_FEAT_TEST = "Features (test)"
SEG_SCORE = "Scoring"
SEG_RESID = "Unattributed"
SEG_VALID = "LOO validation"
SEG_HARNESS = "FIPER harness"

#: Display order == stacking order == categorical slot order (see PALETTE).
SEGMENTS = [SEG_FEAT_FIT, SEG_EST, SEG_FEAT_TEST, SEG_SCORE, SEG_RESID]

#: Seams that turn rollout tensors into Z. Everything else inside a method is estimator
#: work. ``fit`` and ``_features`` appear because their *exclusive* time is the glue
#: (the success mask, the boolean indexing of the chunk tensor) that is genuinely feature
#: construction and would otherwise vanish into the residual.
FEATURE_SEAMS = {
    "fit",
    "_features",
    "_obs",
    "sigma_o",
    "_chunks",
    "_fit_parts",
    "_parts",
    "_fit_phi",
    "sigma_phi",
    "_embed",
    "sigma_A",
    "_pack",
}

#: Seams that belong to FIPER rather than to the method under test.
HARNESS_SEAMS = {
    "dataset_build",
    "get_subset",
    "harness_loop",
    "get_thresholds",
    "get_metrics",
    "save_pickle",
}


def segment_of(phase: str, seam: str) -> str:
    """Map one instrumented seam onto the four displayed segments.

    ``fit`` runs once on the calibration split; ``score_subset`` then runs twice, once per
    subset. The fit/test split inside Features is kept because the two halves mean
    different things -- fit-set features are a one-off, test-set features are a per-step
    deployment cost -- and the calibration scoring pass is folded into Scoring.
    """
    if seam in HARNESS_SEAMS:
        return SEG_HARNESS
    if seam == "evaluate":
        return SEG_RESID  # exclusive time of evaluate() *is* the unattributed residual
    if phase == "validate":
        return SEG_VALID
    if phase == "fit":
        return SEG_FEAT_FIT if seam in FEATURE_SEAMS else SEG_EST
    if phase == "score_test":
        return SEG_FEAT_TEST if seam in FEATURE_SEAMS else SEG_SCORE
    return SEG_SCORE  # score_calibration, features and solves alike


# ------------------------------------------------------------------------- recorder


def _rss_bytes() -> int:
    """Resident set size from ``/proc/self/statm`` -- cheap enough to poll at 200 Hz."""
    with open("/proc/self/statm", "rb") as fh:
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


@dataclass
class _Frame:
    seam: str
    phase: str
    depth: int
    t0: float
    rss_in: int
    cuda_in: int
    child_s: float = 0.0
    #: Peak CUDA bytes of already-closed children, which the allocator's own counter has
    #: been reset past (see ``Recorder._exit``).
    child_cuda_peak: int = 0
    child_cuda_reserved: int = 0


@dataclass
class Recorder:
    """Exclusive wall clock + memory per seam, with CUDA boundaries synchronised.

    Nesting is the whole difficulty: ``_fit_phi`` calls ``_median``, ``_fit_parts`` calls
    ``_parts``, ``model.score`` calls the kernel. A seam is therefore charged
    ``elapsed - sum(children)``, which makes the seam table a partition of the run rather
    than an overlapping set of intervals -- so segment sums are meaningful and the residual
    is real.

    CUDA peaks nest the same way. ``torch.cuda`` exposes one running maximum, so on entry
    a seam resets it and on exit it folds its own maximum up into its parent by hand; the
    reset on exit re-bases the counter at the currently-allocated bytes, which is what the
    parent needs to keep accumulating.
    """

    device: str = "cpu"
    task: str = ""
    method: str = ""
    phase: str = "harness"
    sweep: int = 0
    rows: list[dict] = field(default_factory=list)
    stack: list[_Frame] = field(default_factory=list)
    _median_calls: int = 0
    _rss_t: list[float] = field(default_factory=list)
    _rss_v: list[int] = field(default_factory=list)
    _stop: Any = None
    _thread: Any = None

    @property
    def cuda(self) -> bool:
        return str(self.device).startswith("cuda")

    # -- host RSS sampler ------------------------------------------------------
    def start_sampler(self, interval: float = 0.005) -> None:
        self._stop = threading.Event()

        def loop():
            while not self._stop.wait(interval):
                self._rss_t.append(time.perf_counter())
                self._rss_v.append(_rss_bytes())

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_sampler(self) -> None:
        if self._stop is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)

    def _rss_peak(self, t0: float, t1: float) -> int:
        lo = np.searchsorted(self._rss_t, t0)
        hi = np.searchsorted(self._rss_t, t1)
        window = self._rss_v[lo:hi]
        return max(window) if window else 0

    # -- seams -----------------------------------------------------------------
    def _enter(self, seam: str) -> _Frame:
        import torch

        if self.cuda:
            torch.cuda.synchronize()
            cuda_in = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        else:
            cuda_in = 0
        f = _Frame(seam, self.phase, len(self.stack), time.perf_counter(), _rss_bytes(), cuda_in)
        self.stack.append(f)
        return f

    def _exit(self, f: _Frame, args: tuple = (), out: Any = None) -> None:
        import torch

        if self.cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        elapsed = t1 - f.t0
        if self.cuda:
            cuda_peak = max(torch.cuda.max_memory_allocated(), f.child_cuda_peak)
            cuda_res = max(torch.cuda.max_memory_reserved(), f.child_cuda_reserved)
            cuda_out = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        else:
            cuda_peak = cuda_res = cuda_out = 0

        assert self.stack and self.stack[-1] is f, f"seam stack corrupted at {f.seam}"
        self.stack.pop()
        if self.stack:
            parent = self.stack[-1]
            parent.child_s += elapsed
            parent.child_cuda_peak = max(parent.child_cuda_peak, cuda_peak)
            parent.child_cuda_reserved = max(parent.child_cuda_reserved, cuda_res)

        row = dict(
            task=self.task,
            method=self.method,
            sweep=self.sweep,
            phase=f.phase,
            seam=f.seam,
            depth=f.depth,
            inclusive_s=elapsed,
            exclusive_s=elapsed - f.child_s,
            rss_in_b=f.rss_in,
            rss_peak_b=max(self._rss_peak(f.t0, t1), f.rss_in, _rss_bytes()),
            cuda_in_b=f.cuda_in,
            cuda_out_b=cuda_out,
            cuda_peak_b=cuda_peak,
            cuda_reserved_peak_b=cuda_res,
        )
        row.update(_array_info(_first_array(args), "in"))
        row.update(_array_info(out, "out"))
        self.rows.append(row)

    @contextlib.contextmanager
    def seam(self, name: str):
        f = self._enter(name)
        try:
            yield f
        finally:
            self._exit(f)

    @contextlib.contextmanager
    def in_phase(self, name: str):
        previous = self.phase
        self.phase = name
        try:
            yield
        finally:
            self.phase = previous

    def median_seam(self) -> str:
        """Name the three median-heuristic calls apart: sigma_o, sigma_phi, sigma_A.

        ``_median`` is one method called three times per fit, in the order the bandwidths
        must be fixed. sigma_phi is the one nested inside ``_fit_phi``; of the two
        top-level calls the first is sigma_o (on observations) and the second sigma_A (on
        the mean embeddings, which only exist once phi is drawn).
        """
        if any(fr.seam == "_fit_phi" for fr in self.stack):
            return "sigma_phi"
        self._median_calls += 1
        return "sigma_o" if self._median_calls == 1 else "sigma_A"


def _first_array(args: tuple):
    import torch

    for a in args:
        if isinstance(a, (np.ndarray, torch.Tensor)):
            return a
    return None


def _array_info(x: Any, tag: str) -> dict:
    """Shape / dtype / bytes of a seam's principal input or output, if it has one.

    ``is_view`` is the point of the exercise for ``_parts``: ``sum`` and ``flat`` are a
    reshape of a contiguous tensor and should allocate nothing, which is a fact about the
    returned object, not something to be asserted in a table.
    """
    import torch

    empty = {f"{tag}_shape": "", f"{tag}_dtype": "", f"{tag}_bytes": 0, f"{tag}_is_view": ""}
    if isinstance(x, np.ndarray):
        return {
            f"{tag}_shape": str(tuple(x.shape)),
            f"{tag}_dtype": str(x.dtype),
            f"{tag}_bytes": int(x.nbytes),
            f"{tag}_is_view": str(x.base is not None),
        }
    if isinstance(x, torch.Tensor):
        return {
            f"{tag}_shape": str(tuple(x.shape)),
            f"{tag}_dtype": str(x.dtype).replace("torch.", ""),
            f"{tag}_bytes": int(x.element_size() * x.nelement()),
            f"{tag}_is_view": str(x._base is not None),
        }
    return empty


# -------------------------------------------------------------------- instrumentation

_REC: Recorder | None = None


def _patch(store: list, owner: type, attr: str, seam: Any = None, phase: str | None = None, reset_medians: bool = False):
    """Wrap ``owner.attr`` in a seam, remembering the original for restoration.

    ``seam`` may be a string or a callable taking the recorder (``_median`` needs the
    latter: one method, three bandwidths).
    """
    original = getattr(owner, attr)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        rec = _REC
        if rec is None:
            return original(*args, **kwargs)
        name = seam(rec) if callable(seam) else (seam or attr)
        if reset_medians:
            rec._median_calls = 0
        frame = rec._enter(name)
        out = None
        try:
            with rec.in_phase(phase) if phase else contextlib.nullcontext():
                out = original(*args, **kwargs)
            return out
        finally:
            rec._exit(frame, args[1:], out)

    setattr(owner, attr, wrapper)
    store.append((owner, attr, original))


@contextlib.contextmanager
def instrumented(rec: Recorder):
    """Install the seams for the duration of a profile run, then take them back out.

    The ten seams named in the spec, plus four that make the residual interpretable: the
    kernel gram (which separates the O(m^2 d) gram from the O(m^3) Cholesky inside
    ``KernCD.fit``, and the gram from the triangular solve inside ``model.score``),
    ``score_subset``, ``_features`` and ``fit`` itself.
    """
    global _REC
    import my_methods.base as mb
    import my_methods.run as mr
    from cd.algs.kern_cd import KernCD
    from cd.algs.kernels import RBF
    from evaluation.method_eval_classes import BaseEvalClass

    from my_methods.kern_cd_core import ChunkPartsMixin, KernCDMethod
    from my_methods.methods.kern_cd_sig import KernCDSig
    from my_methods.methods.kern_cd_sum import KernCDSum

    _REC = rec
    undo: list = []
    original_factory = mb.make_eval_class
    try:
        # -- the method under test
        _patch(undo, KernCDMethod, "fit", "fit", reset_medians=True)
        _patch(undo, KernCDMethod, "_obs")
        _patch(undo, KernCDMethod, "_chunks")
        _patch(undo, KernCDMethod, "_median", seam=lambda r: r.median_seam())
        _patch(undo, KernCDMethod, "_fit_parts")
        _patch(undo, KernCDMethod, "_fit_phi")
        _patch(undo, KernCDMethod, "_embed")
        _patch(undo, KernCDMethod, "_pack")
        _patch(undo, KernCDMethod, "_loo_scores")
        # The brute-force LOO cross-check is O(m^4) and only runs at m <= 300; it is
        # debug scaffolding, not method cost, so it gets its own phase and its own bar.
        _patch(undo, KernCDMethod, "_validate_loo", phase="validate")
        _patch(undo, KernCDMethod, "_features")
        _patch(undo, KernCDMethod, "score_subset")
        _patch(undo, KernCDSig, "_fit_parts")
        _patch(undo, ChunkPartsMixin, "_parts", "_parts")
        _patch(undo, KernCDSum, "_parts", "_parts")

        # -- the estimator
        _patch(undo, KernCD, "fit", "KernCD.fit")
        _patch(undo, KernCD, "score", "model.score")
        _patch(undo, RBF, "__call__", "kernel.gram")
        _patch(undo, RBF, "diag", "kernel.diag")

        # -- FIPER's own costs
        _patch(undo, BaseEvalClass, "_process_rollouts", "harness_loop")
        _patch(undo, BaseEvalClass, "_get_thresholds", "get_thresholds")
        _patch(undo, BaseEvalClass, "_get_metrics", "get_metrics")
        _patch(undo, BaseEvalClass, "_save_pickle", "save_pickle")

        # Both bindings: ``my_methods.run`` imported the factory by value, so patching
        # only the module it came from would leave the runner using the original.
        factory = _profiled_factory(rec, original_factory)
        mb.make_eval_class = factory
        mr.make_eval_class = factory
        yield
    finally:
        mb.make_eval_class = original_factory
        mr.make_eval_class = original_factory
        for owner, attr, original in reversed(undo):
            setattr(owner, attr, original)
        _REC = None


def _profiled_factory(rec: Recorder, original_factory: Callable) -> Callable:
    """``make_eval_class`` returning an adapter that tags phases and times ``evaluate``.

    Subclassing the generated adapter rather than patching it keeps the adapter's own
    logic untouched: this only brackets ``evaluate`` (whose *exclusive* time is the
    residual, since every child of consequence is a seam), names the three phases, and
    charges ``_get_subset`` -- dataset slicing and normalisation, upstream of the method --
    to the harness.
    """

    def make(method_cls, params):
        base_cls = original_factory(method_cls, params)

        class Profiled(base_cls):  # type: ignore[valid-type, misc]
            def evaluate(self):
                import torch

                rec.method = method_cls.name
                if rec.cuda:
                    torch.cuda.empty_cache()
                with rec.seam("evaluate"):
                    return super().evaluate()

            def _get_subset(self, subset):
                with rec.seam("get_subset"):
                    return super()._get_subset(subset)

            def _execute_preprocessing(self):
                with rec.in_phase("fit"):
                    return super()._execute_preprocessing()

            def _process_rollouts(self, subset):
                with rec.in_phase(f"score_{subset}"):
                    return super()._process_rollouts(subset)

        Profiled.__name__ = f"Profiled{base_cls.__name__}"
        return Profiled

    return make


# ------------------------------------------------------------------------- the run


def run_profile(
    tasks: list[str],
    methods: list[str],
    seed: int = SEED,
    repeats: int = 6,
    progress: Callable | None = None,
) -> pd.DataFrame:
    """Evaluate ``methods`` on ``tasks`` through the real FIPER pipeline, instrumented.

    The dataset is built once per task and shared across the methods, exactly as
    ``my_methods/run.py`` does, so the rebuild is charged once to the harness bar rather
    than once per method.

    The first of ``repeats`` sweeps is discarded as warm-up, and that is not optional:
    whichever method runs first pays for the BLAS thread pool, the cuBLAS handle and the
    first-touch of every kernel, which on pretzel made the observation-only baseline --
    doing strictly the least work of the five -- the most expensive to score. That
    baseline *is* the reference floor, so contaminating it contaminates every comparison
    drawn against it.

    The remaining sweeps are all recorded and reduced by minimum downstream (see
    :func:`segment_table`). At these task sizes a whole method costs tens of milliseconds,
    which is well inside the noise of a single measurement on a shared machine, so more
    sweeps buy a better minimum and the spread across them is reported alongside it rather
    than averaged away.
    """
    import torch

    from shared_utils.hydra_utils import load_config
    from shared_utils.utility_functions import set_seed
    from tasks import TaskManager

    from my_methods.base import discover
    from my_methods.run import BASE_CONFIG_PATH, BASE_DATA_PATH, _RegistryEvaluationManager, _union

    discover()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rec = Recorder(device=device)
    rec.start_sampler()
    try:
        with instrumented(rec):
            for task in tasks:
                rec.task, rec.method = task, "__harness__"
                if progress:
                    progress(f"{task}: building dataset")
                set_seed(seed)
                cfg = load_config("task", task, return_only_subdict=False)
                task_data_path = os.path.join(BASE_DATA_PATH, task)
                with rec.seam("dataset_build"):
                    taskmanager = TaskManager(
                        cfg,
                        task,
                        BASE_CONFIG_PATH,
                        task_data_path,
                        required_tensors=_union(methods, "tensors"),
                        optional_tensors=_union(methods, "optional_tensors"),
                        device=device,
                    )
                    dataset = taskmanager.get_rollout_dataset(load_dataset_if_exists=False)

                manager = _RegistryEvaluationManager(
                    BASE_CONFIG_PATH, task_data_path, dataset, device=device, seed=seed
                )
                for sweep in range(repeats):
                    keep = sweep > 0
                    rec.sweep = sweep
                    recorded, rec.rows = rec.rows, (rec.rows if keep else [])
                    for method in methods:
                        if progress:
                            progress(f"{task}: {method}" + (f" (sweep {sweep})" if keep else " (warm-up)"))
                        manager.evaluate([method], combine_methods=False)
                    rec.rows = recorded
                del dataset, taskmanager, manager
                if device == "cuda":
                    torch.cuda.empty_cache()
    finally:
        rec.stop_sampler()

    df = pd.DataFrame(rec.rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SEAMS_CSV, index=False)
    pd.DataFrame(
        [
            dict(
                generated=pd.Timestamp.now().isoformat(timespec="seconds"),
                device=device,
                gpu=torch.cuda.get_device_name(0) if device == "cuda" else "",
                torch=torch.__version__,
                seed=seed,
                warmup_sweeps=1,
                recorded_sweeps=max(1, repeats - 1),
                tasks=",".join(tasks),
                methods=",".join(methods),
            )
        ]
    ).to_csv(META_CSV, index=False)
    return df


# ------------------------------------------------------------------- derived tables


def load_cache() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if not SEAMS_CSV.exists():
        return None, None
    seams = pd.read_csv(SEAMS_CSV, keep_default_na=False)
    for col in ("inclusive_s", "exclusive_s"):
        seams[col] = pd.to_numeric(seams[col])
    meta = pd.read_csv(META_CSV) if META_CSV.exists() else None
    return seams, meta


def add_segments(seams: pd.DataFrame) -> pd.DataFrame:
    out = seams.copy()
    out["segment"] = [segment_of(p, s) for p, s in zip(out["phase"], out["seam"])]
    return out


def segment_table(seams: pd.DataFrame) -> pd.DataFrame:
    """Seconds per (task, method, segment) -- the roll-up that Deliverable 1 plots.

    Each sweep is summed first and the sweeps are then reduced by **minimum**, not mean or
    median. Contention only ever adds time: a sweep that lands while another process owns
    the cores measures the scheduler, not the code. The minimum is the closest estimate of
    the work itself that repeated sampling can give, and ``spread`` -- the sweep-to-sweep
    range -- is what says how contended the box was while measuring. Read every number
    here against its spread; this machine is shared.
    """
    df = add_segments(seams)
    method_rows = df[(df["method"] != "__harness__") & (df["segment"] != SEG_HARNESS)]
    per_sweep = method_rows.groupby(["task", "method", "segment", "sweep"], as_index=False)["exclusive_s"].sum()
    return (
        per_sweep.groupby(["task", "method", "segment"])["exclusive_s"]
        .agg(seconds="min", spread=lambda s: float(s.max() - s.min()), sweeps="size")
        .reset_index()
    )


def harness_table(seams: pd.DataFrame) -> pd.DataFrame:
    """FIPER's own per-evaluation costs, per task.

    The dataset rebuild is paid once per task no matter how many methods run, so it is
    reported as itself; every other harness seam is averaged over the methods, which is
    what a single-method run would actually pay.
    """
    df = add_segments(seams)
    harness = df[df["segment"] == SEG_HARNESS]
    build = harness[harness["seam"] == "dataset_build"].groupby("task", as_index=False)["exclusive_s"].sum()
    build["seam"] = "dataset_build"

    per_sweep = (
        harness[harness["seam"] != "dataset_build"]
        .groupby(["task", "seam", "sweep"], as_index=False)
        .agg(exclusive_s=("exclusive_s", "sum"), n_methods=("method", "nunique"))
    )
    per_sweep["exclusive_s"] /= per_sweep["n_methods"]
    rest = per_sweep.groupby(["task", "seam"], as_index=False)["exclusive_s"].min()

    out = pd.concat([build[["task", "seam", "exclusive_s"]], rest], ignore_index=True)
    return out.rename(columns={"exclusive_s": "seconds"}).sort_values(["task", "seconds"], ascending=[True, False])


def total_table(seams: pd.DataFrame) -> pd.DataFrame:
    """Best-case total per method and the sweep-to-sweep range around it (see segment_table)."""
    df = add_segments(seams)
    rows = df[(df["method"] != "__harness__") & (df["segment"] != SEG_HARNESS)]
    per_sweep = rows.groupby(["task", "method", "sweep"], as_index=False)["exclusive_s"].sum()
    return (
        per_sweep.groupby(["task", "method"])["exclusive_s"]
        .agg(total_min="min", spread=lambda s: float(s.max() - s.min()))
        .reset_index()
    )


def _parse_shape(s: str) -> tuple[int, ...]:
    s = str(s).strip()
    if not s or s == "()":
        return ()
    return tuple(int(x) for x in s.strip("()").rstrip(",").split(",") if x.strip())


def dims_table(seams: pd.DataFrame) -> pd.DataFrame:
    """The shapes that drive every cost, read back off the measured seams.

    Nothing here is declared: ``d_o`` comes from what ``_obs`` returned, ``B``/``H`` from
    what ``_chunks`` returned, ``P``/``F`` from what ``_parts`` returned, ``m`` from the
    matrix handed to ``KernCD.fit``.
    """
    rows = []
    for (task, method), g in seams[seams["method"] != "__harness__"].groupby(["task", "method"]):
        rec: dict[str, Any] = dict(task=task, method=method)
        obs_fit = _parse_shape(_last(g, phase="fit", seam="_obs", col="out_shape"))
        obs_test = _parse_shape(_last(g, phase="score_test", seam="_obs", col="out_shape"))
        obs_cal = _parse_shape(_last(g, phase="score_calibration", seam="_obs", col="out_shape"))
        chunks = _parse_shape(_last(g, phase="score_test", seam="_chunks", col="out_shape"))
        parts = _parse_shape(_last(g, phase="score_test", seam="_parts", col="out_shape"))
        zfit = _parse_shape(_last(g, phase="fit", seam="KernCD.fit", col="in_shape"))
        rec["m"] = zfit[0] if zfit else (obs_fit[0] if obs_fit else 0)
        rec["d_packed"] = zfit[1] if len(zfit) > 1 else 0
        rec["N_test"] = obs_test[0] if obs_test else 0
        # The calibration scoring pass re-derives features for the *whole* calibration
        # split, failed episodes included, so it is wider than the m rows that were fitted.
        rec["N_cal"] = obs_cal[0] if obs_cal else 0
        rec["d_o"] = obs_fit[1] if len(obs_fit) > 1 else 0
        rec["B"] = chunks[1] if len(chunks) > 2 else 0
        rec["H"] = chunks[2] if len(chunks) > 2 else 0
        rec["D"] = chunks[3] if len(chunks) > 3 else 0
        rec["P"] = parts[1] if len(parts) > 2 else 0
        rec["F"] = parts[2] if len(parts) > 2 else 0
        rec["n_components"] = 128
        rows.append(rec)
    return pd.DataFrame(rows)


def _last(g: pd.DataFrame, phase: str, seam: str, col: str) -> str:
    sel = g[(g["phase"] == phase) & (g["seam"] == seam)]
    return str(sel[col].iloc[-1]) if len(sel) else ""


# ------------------------------------------------ Deliverable 2: the array inventory


@dataclass
class ArrayEntry:
    """One array big enough to matter, with a formula rather than a number.

    ``symbolic`` says what breaks as tasks grow; ``nbytes`` pins what is true today;
    ``lifetime`` is what governs peak memory, since three m x m matrices alive at once is
    a fact no single peak number conveys.
    """

    name: str
    site: str
    symbolic: str
    dtype: str
    device: str
    lifetime: str
    segment: str
    nbytes: Callable[[dict], float]
    applies: Callable[[str], bool] = lambda m: True
    #: The moments at which this array is simultaneously reachable. A peak is a moment,
    #: not a segment, so "what is live" has to be answered per moment or the sum is a
    #: fiction: summing every array a segment ever allocates double counts everything that
    #: was already freed. See PEAK_MOMENTS.
    groups: Callable[[dict], tuple[str, ...]] = lambda D: ("build", "rff")
    note: str = ""
    #: Feature arrays exist twice -- once at m rows in ``fit``, once at N rows in the test
    #: pass -- and are charged to whichever segment built them. Everything else is
    #: described once, on the pass where it is created.
    per_pass: bool = False


#: The device-side moments inside one features pass, in the order they occur.
PEAK_MOMENTS = {
    "cat": "inside `_chunk_feature`, while the time channel is concatenated on",
    "build": "inside `_parts`, with every path intermediate alive",
    "rff": "inside `_embed`, the RFF batch loop",
}

F8, F4 = 8, 4
_ACTION = lambda m: m != "kern_cd_obs"  # noqa: E731
_SIG = lambda m: m == "kern_cd_sig"  # noqa: E731
_COPY_PARTS = lambda m: m in ("kern_cd_disp", "kern_cd_sig")  # noqa: E731
_VIEW_PARTS = lambda m: m in ("kern_cd_sum", "kern_cd_flat")  # noqa: E731


def _embed_rows(D: dict) -> int:
    """Rows per RFF batch: ``_RFF_ELEM_BUDGET // (P * n_components)``, from kern_cd_core."""
    from my_methods.kern_cd_core import _RFF_ELEM_BUDGET

    denom = max(1, D["P"] * D["n_components"])
    return max(1, min(D["n"], _RFF_ELEM_BUDGET // denom))


def _rff_one(D: dict, rows: int) -> float:
    return rows * D["P"] * D["n_components"] * F4


def _rff_peak(D: dict) -> float:
    """Peak device bytes of the ``_embed`` batch loop, over its iterations.

    ``_RFF_ELEM_BUDGET`` caps *one* intermediate, but three are reachable at once. Two are
    the temporaries of ``scale * cos(blk @ W + b)`` -- the matmul result is still alive
    when the broadcast add allocates its output. The third is the *previous* iteration's
    ``phi``, which the loop only rebinds after the next one has been allocated:

        for s in range(0, N, rows):
            phi = self._phi_scale * torch.cos(blk @ self._phi_W + self._phi_b)

    So the cap bounds peak memory at 3x its value, not 1x, whenever the loop runs to a
    second full block. Measured at 3.03x and 3.00x on standalone replays of the loop at
    push_t's two shapes.
    """
    n, rows = D["n"], _embed_rows(D)
    sizes = [min(rows, n - s) for s in range(0, n, rows)]
    return max(
        _rff_one(D, prev) + 2 * _rff_one(D, cur)
        for prev, cur in zip([0, *sizes], sizes)
    )


#: The catalogue. ``n`` is the rows of the pass being described (m for the fit pass, N for
#: the test pass); everything else comes from ``dims_table``.
INVENTORY: list[ArrayEntry] = [
    ArrayEntry(
        "action_preds (subset)", "rollout_datasets.get_subset -> _chunks", "[n, B, H, D] host",
        "float32/64", "host", "held", SEG_FEAT_TEST,
        lambda D: D["n"] * D["B"] * D["H"] * D["D"] * D["src_action_itemsize"], _ACTION,
        note="the whole subset's action tensor, materialised before any batching", per_pass=True,
    ),
    ArrayEntry(
        "chunks", "kern_cd_core._chunks", "[n, B, H, D] device", "float32", "cuda", "held", SEG_FEAT_TEST,
        lambda D: D["n"] * D["B"] * D["H"] * D["D"] * F4, _ACTION, per_pass=True,
        # In fit() chunks is a named local held across _embed. In _features it is an
        # expression temporary -- `_embed(_parts(_chunks(x)))` -- so it survives into the
        # RFF loop only when _parts handed back a view of it.
        groups=lambda D: ("cat", "build", "rff") if (D["parts_is_view"] or D["is_fit"]) else ("cat", "build"),
        note="freed before _embed on the test pass unless _parts returned a view",
    ),
    ArrayEntry(
        "parts (view)", "kern_cd_{sum,flat}._parts", "[n, P, F] view of chunks", "float32", "cuda", "held",
        SEG_FEAT_TEST, lambda D: 0.0, _VIEW_PARTS, groups=lambda D: (),
        note="a reshape of a contiguous tensor: allocates nothing", per_pass=True,
    ),
    ArrayEntry(
        "parts", "kern_cd_{disp,sig}._parts", "[n, P, F] device", "float32", "cuda", "held", SEG_FEAT_TEST,
        lambda D: D["n"] * D["P"] * D["F"] * F4, _COPY_PARTS, per_pass=True,
    ),
    ArrayEntry(
        "sig: paths (dilated)", "kern_cd_sig._chunk_feature", "[nB, H, D] device", "float32", "cuda", "transient",
        SEG_FEAT_TEST, lambda D: D["n"] * D["B"] * D["H"] * D["D"] * F4, _SIG, per_pass=True,
        groups=lambda D: ("cat",), note="rebound by the cat, so it dies before inc/pre exist",
    ),
    ArrayEntry(
        "sig: paths + time channel", "kern_cd_sig._chunk_feature", "[nB, H, D+1] device", "float32", "cuda",
        "transient", SEG_FEAT_TEST, lambda D: D["n"] * D["B"] * D["H"] * (D["D"] + 1) * F4, _SIG, per_pass=True,
        groups=lambda D: ("cat", "build"),
    ),
    ArrayEntry(
        "sig: inc", "kern_cd_sig._chunk_feature", "[nB, H-1, D+1] device", "float32", "cuda", "transient",
        SEG_FEAT_TEST, lambda D: D["n"] * D["B"] * (D["H"] - 1) * (D["D"] + 1) * F4, _SIG, per_pass=True,
        groups=lambda D: ("build",),
    ),
    ArrayEntry(
        "sig: pre", "kern_cd_sig._chunk_feature", "[nB, H-1, D+1] device", "float32", "cuda", "transient",
        SEG_FEAT_TEST, lambda D: D["n"] * D["B"] * (D["H"] - 1) * (D["D"] + 1) * F4, _SIG, per_pass=True,
        groups=lambda D: ("build",),
    ),
    ArrayEntry(
        "RFF: blk @ W", "kern_cd_core._embed", "[rows, P, 128] device", "float32", "cuda", "transient",
        SEG_FEAT_TEST, lambda D: _rff_one(D, _embed_rows(D)), _ACTION,
        groups=lambda D: ("rff",), note="one _RFF_ELEM_BUDGET worth", per_pass=True,
    ),
    ArrayEntry(
        "RFF: cos(...)", "kern_cd_core._embed", "[rows, P, 128] device", "float32", "cuda", "transient",
        SEG_FEAT_TEST, lambda D: _rff_one(D, _embed_rows(D)), _ACTION,
        groups=lambda D: ("rff",), note="alive alongside the matmul result it consumes", per_pass=True,
    ),
    ArrayEntry(
        "RFF: previous block's phi", "kern_cd_core._embed", "[rows, P, 128] device", "float32", "cuda",
        "transient", SEG_FEAT_TEST,
        lambda D: max(0.0, _rff_peak(D) - 2 * _rff_one(D, _embed_rows(D))), _ACTION,
        groups=lambda D: ("rff",), per_pass=True,
        note="the loop rebinds phi only after the next block is allocated: peak is 3x the budget",
    ),
    ArrayEntry(
        "mu (mean embeddings)", "kern_cd_core._embed", "[n, 128] host", "float64", "host", "held", SEG_FEAT_TEST,
        lambda D: D["n"] * D["n_components"] * F8, _ACTION, per_pass=True,
    ),
    ArrayEntry(
        "obs (float64)", "kern_cd_core._obs", "[n, d_o] host", "float64", "host", "held", SEG_FEAT_TEST,
        lambda D: D["n"] * D["d_o"] * F8, per_pass=True,
    ),
    ArrayEntry(
        "Z (packed)", "kern_cd_core._pack", "[n, d_o + 128] host", "float64", "host", "held", SEG_FEAT_TEST,
        lambda D: D["n"] * (D["d_o"] + (D["n_components"] if D["uses_actions"] else 0)) * F8, per_pass=True,
    ),
    ArrayEntry(
        "median heuristic: pdist", "cd.utils.misc.median_distance", "[c(c-1)/2], c = min(n, 3000)", "float64",
        "host", "transient", SEG_FEAT_FIT, lambda D: D["c"] * (D["c"] - 1) / 2 * F8,
    ),
    ArrayEntry(
        "K (gram)", "KernCD._fit_exact", "[m, m] host", "float64", "host", "transient", SEG_EST,
        lambda D: D["m"] ** 2 * F8,
    ),
    ArrayEntry(
        "eye(m)", "KernCD._fit_exact", "[m, m] host", "float64", "host", "transient", SEG_EST,
        lambda D: D["m"] ** 2 * F8,
    ),
    ArrayEntry(
        "K + lam*m*I", "KernCD._fit_exact", "[m, m] host", "float64", "host", "held", SEG_EST,
        lambda D: D["m"] ** 2 * F8,
    ),
    ArrayEntry(
        "L (Cholesky)", "KernCD._fit_exact", "[m, m] host", "float64", "host", "held", SEG_EST,
        lambda D: D["m"] ** 2 * F8,
    ),
    ArrayEntry(
        "L_inv", "kern_cd_core._loo_scores", "[m, m] host", "float64", "host", "transient", SEG_EST,
        lambda D: D["m"] ** 2 * F8, note="alive with eye(m) and the held L",
    ),
    ArrayEntry(
        "eye(m) (LOO)", "kern_cd_core._loo_scores", "[m, m] host", "float64", "host", "transient", SEG_EST,
        lambda D: D["m"] ** 2 * F8,
    ),
    ArrayEntry(
        "kx (query gram)", "KernCD._score_exact", "[block, m] host", "float64", "host", "transient", SEG_SCORE,
        lambda D: D["block"] * D["m"] * F8,
    ),
    ArrayEntry(
        "y (triangular solve)", "KernCD._score_exact", "[block, m] host", "float64", "host", "transient",
        SEG_SCORE, lambda D: D["block"] * D["m"] * F8,
    ),
    ArrayEntry(
        "euclidean_distances temp", "sklearn.metrics.pairwise.rbf_kernel", "[block, m] host", "float64", "host",
        "transient", SEG_SCORE, lambda D: D["block"] * D["m"] * F8,
    ),
]

#: Decimal MB (10^6 bytes), not MiB. The spec quotes 465 MB for push_t's
#: ``[9458, 256, 16, 3]`` float32 chunk tensor, which is 464,879,616 bytes; reporting that
#: as 443 MiB would make every prediction in the spec look wrong by 5%.
MB = 1e6
INVENTORY_FLOOR_MB = 10.0


#: The three passes that build features, and the segment each is charged to. The
#: calibration pass is a third pass the stage table does not name -- it re-derives features
#: for the whole calibration split, failed episodes included -- and the spec folds it into
#: Scoring, so that is where its arrays are charged.
FEATURE_PASSES = (
    ("fit (m rows)", "m", SEG_FEAT_FIT),
    ("calibration (n_cal rows)", "N_cal", SEG_SCORE),
    ("test (N rows)", "N_test", SEG_FEAT_TEST),
)


def inventory_table(dims: pd.DataFrame, seams: pd.DataFrame, floor_mb: float = INVENTORY_FLOOR_MB) -> pd.DataFrame:
    """Evaluate the catalogue at each task's measured dimensions; keep what exceeds ~10 MB."""
    from my_methods.kern_cd_core import _MEDIAN_MAX_POINTS, _SCORE_BLOCK

    rows = []
    for _, d in dims.iterrows():
        method = d["method"]
        for which, dim_key, pass_segment in FEATURE_PASSES:
            n = int(d[dim_key])
            if not n:
                continue
            src_itemsize = _source_action_itemsize(seams, d["task"], method)
            D = dict(
                n=n,
                m=int(d["m"]),
                N=int(d["N_test"]),
                B=int(d["B"]),
                H=int(d["H"]),
                D=int(d["D"]) or 3,
                d_o=int(d["d_o"]),
                P=int(d["P"]),
                F=int(d["F"]),
                n_components=int(d["n_components"]),
                block=_SCORE_BLOCK,
                c=min(n, _MEDIAN_MAX_POINTS),
                uses_actions=method != "kern_cd_obs",
                src_action_itemsize=src_itemsize,
                parts_is_view=method in ("kern_cd_sum", "kern_cd_flat"),
                is_fit=which == "fit (m rows)",
            )
            for e in INVENTORY:
                if not e.applies(method):
                    continue
                # Estimator and scoring arrays are sized by m and the query block, not by
                # the pass, so they are described once -- on the fit pass -- rather than
                # duplicated. Feature arrays exist on both passes at different n.
                if not e.per_pass and which != "fit (m rows)":
                    continue
                segment = pass_segment if e.per_pass else e.segment
                nb = float(e.nbytes(D))
                if nb / MB < floor_mb and not e.symbolic.endswith("view of chunks"):
                    continue
                rows.append(
                    dict(
                        task=d["task"],
                        method=method,
                        pass_=which if e.per_pass else e.segment,
                        name=e.name,
                        site=e.site,
                        symbolic=e.symbolic,
                        concrete=_concretise(e.symbolic, D),
                        dtype=e.dtype,
                        device=e.device,
                        MB=round(nb / MB, 1),
                        lifetime=e.lifetime,
                        segment=segment,
                        live_at=" + ".join(e.groups(D)),
                        note=e.note,
                    )
                )
    return pd.DataFrame(rows)


def _concretise(symbolic: str, D: dict) -> str:
    """Substitute the measured dimensions into a symbolic shape string."""
    body = symbolic.split("]")[0].strip("[")
    parts = []
    for token in body.split(","):
        t = token.strip()
        expr = {
            "n": D["n"], "m": D["m"], "N": D["N"], "B": D["B"], "H": D["H"], "D": D["D"],
            "P": D["P"], "F": D["F"], "d_o": D["d_o"], "128": D["n_components"],
            "block": D["block"], "nB": D["n"] * D["B"], "H-1": D["H"] - 1,
            "D+1": D["D"] + 1, "d_o + 128": D["d_o"] + D["n_components"],
            "rows": _embed_rows(D),
        }.get(t)
        parts.append(str(expr) if expr is not None else t)
    return "[" + ", ".join(parts) + "]"


def _source_action_itemsize(seams: pd.DataFrame, task: str, method: str) -> int:
    """Itemsize of the host action tensor handed to ``_chunks`` (float32 vs float64)."""
    sel = seams[(seams["task"] == task) & (seams["method"] == method) & (seams["seam"] == "_chunks")]
    sel = sel[sel["in_dtype"] != ""]
    if not len(sel):
        return 4
    return 8 if "64" in str(sel["in_dtype"].iloc[-1]) else 4


#: What the spec predicts each method's ``_parts`` allocates on the test pass, in MB.
#: ``0`` means "returns a view of a contiguous tensor and allocates nothing".
_PARTS_PREDICTION = {"kern_cd_sum": 0.0, "kern_cd_flat": 0.0, "kern_cd_disp": 29.0, "kern_cd_sig": None}


def predictions_table(seams: pd.DataFrame, dims: pd.DataFrame, task: str = "push_t") -> pd.DataFrame:
    """The spec's predictions for the test pass, against what was measured.

    The verdict keys on *net allocation across the seam*, not on whether the returned
    object is a view. Those two come apart for ``sig``, which returns a view -- of the
    ``S2`` tensor it just built -- while allocating 148 MB and peaking at 2.2 GB. A
    ``_base`` check alone would have called that "allocates nothing".
    """
    rows = []
    d = dims[dims["task"] == task]
    s = seams[seams["task"] == task]

    for method, predicted in _PARTS_PREDICTION.items():
        sel = s[(s["method"] == method) & (s["seam"] == "_parts") & (s["phase"] == "score_test")]
        if not len(sel):
            continue
        is_view = str(sel["out_is_view"].iloc[-1]) == "True"
        net = float(sel["cuda_out_b"].iloc[-1] - sel["cuda_in_b"].iloc[-1]) / MB
        peak = float(sel["cuda_peak_b"].iloc[-1] - sel["cuda_in_b"].iloc[-1]) / MB
        if predicted is None:
            claim, ok = "sig is the outlier, peaking near 2.3 GB", 2000 <= peak <= 2600
        elif predicted == 0.0:
            claim, ok = "returns a view, allocates nothing", net < 1.0
        else:
            claim, ok = f"allocates {predicted:.0f} MB", abs(net - predicted) < 0.1 * predicted
        rows.append(
            dict(
                prediction=f"`{method}._parts`: {claim}",
                measured=f"net {net:,.1f} MB, peak +{peak:,.0f} MB, out {sel['out_shape'].iloc[-1]}, is_view={is_view}",
                verdict="confirmed" if ok else "refuted",
            )
        )

    chunks = s[(s["seam"] == "_chunks") & (s["phase"] == "score_test")]
    if len(chunks):
        dev_mb = float(chunks["out_bytes"].iloc[-1]) / MB
        host_mb = float(chunks["in_bytes"].iloc[-1]) / MB
        f64 = "64" in str(chunks["in_dtype"].iloc[-1])
        rows.append(
            dict(
                prediction="`_chunks` materialises the whole subset's action tensor: 465 MB on the "
                "device, and 930 MB on the host first if the source is float64",
                measured=f"device {dev_mb:,.0f} MB {chunks['out_shape'].iloc[-1]}, "
                f"host source {host_mb:,.0f} MB ({chunks['in_dtype'].iloc[-1]})",
                verdict="confirmed" if abs(dev_mb - 465) < 25 else "refuted",
            )
        )
        rows.append(
            dict(
                prediction="...and the host source is float64, so 930 MB precedes it",
                measured=f"source dtype is {chunks['in_dtype'].iloc[-1]} ({host_mb:,.0f} MB)",
                verdict="confirmed" if f64 else "refuted -- the source is float32, so the host copy is 465 MB",
            )
        )

    parts_rows = s[(s["seam"] == "_parts") & (s["phase"] == "score_test")]
    if len(parts_rows) and len(d):
        n_test = int(d["N_test"].max())
        got = {_parse_shape(x)[0] for x in parts_rows["out_shape"] if _parse_shape(x)}
        rows.append(
            dict(
                prediction=f"`_parts` runs on all N={n_test:,} rows before `_embed` batches, so the "
                "_RFF_ELEM_BUDGET cap bounds the RFF intermediate but not the parts tensor",
                measured=f"_parts output row counts: {sorted(got)}",
                verdict="confirmed" if got == {n_test} else "refuted",
            )
        )
    return pd.DataFrame(rows)


# ------------------------------------------- Deliverable 4: the scaling knobs / model

#: stage -> (displayed complexity, dominant term as a function of the measured dims).
#: The Features term is the RFF matmul (n P F x 128 MACs), which dominates the cos
#: evaluations the spec's ``N P . 128`` counts. The scoring solve is O(N m^2) in total --
#: the block size sets arithmetic intensity, not flop count -- so that is what is fitted.
COST_MODEL = {
    SEG_FEAT_FIT: (
        "O(m P F + m P . 128)",
        "m P F . 128",
        lambda D: D["m"] * D["P"] * D["F"] * D["n_components"],
    ),
    SEG_EST: ("O(m^2 d + m^3)", "m^3 / 3", lambda D: D["m"] ** 3 / 3.0),
    SEG_FEAT_TEST: (
        "O(N P F + N P . 128)",
        "N P F . 128",
        lambda D: D["N"] * D["P"] * D["F"] * D["n_components"],
    ),
    SEG_SCORE: ("O(N m d + N m^2 / block)", "N m^2", lambda D: D["N"] * D["m"] ** 2),
}


def cost_model_table(dims: pd.DataFrame, segs: pd.DataFrame) -> pd.DataFrame:
    """Stage -> complexity -> dominant symbol -> measured seconds -> implied constant.

    Two tasks give two independent estimates of each constant. Agreement is the check:
    a constant that moves by an order of magnitude between tasks means the stated
    complexity is not what the stage is actually doing.
    """
    rows = []
    for _, d in dims.iterrows():
        # P and F are 0 for the obs-only baseline, which has no RFF map at all: its
        # feature terms are then 0 and the implied constant is undefined, which is the
        # honest answer rather than a number fitted to work that does not exist.
        D = dict(
            m=int(d["m"]), N=int(d["N_test"]), P=int(d["P"]), F=int(d["F"]),
            d_o=int(d["d_o"]), n_components=int(d["n_components"]),
        )
        for seg, (complexity, symbol, term) in COST_MODEL.items():
            sec = segs[(segs["task"] == d["task"]) & (segs["method"] == d["method"]) & (segs["segment"] == seg)]
            seconds = float(sec["seconds"].sum())
            work = float(term(D))
            rows.append(
                dict(
                    task=d["task"], method=d["method"], stage=seg, complexity=complexity,
                    dominant=symbol, work=work, seconds=round(seconds, 3),
                    ns_per_unit=(seconds / work * 1e9) if work else float("nan"),
                )
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ the charts

#: Categorical slots 1-5 in fixed order, one per segment, plus green for the debug bar.
#: The dark column is the same six hues re-stepped for a dark surface, not a second
#: palette; both were run through the six-check validator (worst adjacent CVD dE 9.1 light
#: / 8.4 dark, normal-vision 19.6 / 19.3) against their own surface. Assigned by segment,
#: never cycled, so a segment keeps its colour when another one drops out.
_PALETTES = {
    "light": {
        SEG_FEAT_FIT: "#2a78d6",
        SEG_EST: "#eb6834",
        SEG_FEAT_TEST: "#1baf7a",
        SEG_SCORE: "#eda100",
        SEG_RESID: "#e87ba4",
        SEG_VALID: "#008300",
    },
    "dark": {
        SEG_FEAT_FIT: "#3987e5",
        SEG_EST: "#d95926",
        SEG_FEAT_TEST: "#199e70",
        SEG_SCORE: "#c98500",
        SEG_RESID: "#d55181",
        SEG_VALID: "#008300",
    },
}
_INK = {
    "light": dict(ink="#0b0b0b", muted="#52514e", grid="#e6e5e0", neutral="#b3b2a8", gap="#fcfcfb"),
    "dark": dict(ink="#ffffff", muted="#c3c2b7", grid="#3a3a38", neutral="#6f6e66", gap="#0e1117"),
}


def _theme() -> tuple[dict, dict]:
    """The palette and ink for the viewer's Streamlit theme.

    The chart surface is left transparent so it sits on whatever Streamlit painted; only
    the marks, text and grid switch. A light chart embedded in a dark app reads as a bug,
    and flipping the hues rather than re-stepping them for the dark surface would fail the
    contrast check -- hence two selected columns rather than one inverted.
    """
    import streamlit as st

    mode = "light"
    with contextlib.suppress(Exception):
        mode = st.context.theme.type or "light"
    mode = mode if mode in _PALETTES else "light"
    return _PALETTES[mode], _INK[mode]


def deliverable1_figure(
    segs: pd.DataFrame,
    harness: pd.DataFrame,
    tasks: list[str],
    methods: list[str],
    include_harness: bool = True,
):
    """Stacked horizontal bars, method-major, with the obs floor drawn as a reference line.

    Drawn twice, because one linear axis cannot do both jobs: with the harness bar in, the
    five methods are 3% of the width (which is the answer to "is this worth optimising?"),
    and with it out the segments are legible (which is the answer to "where does the time
    go?"). A log axis would make the stack lengths non-additive, so it is not an option.
    """
    import plotly.graph_objects as go

    palette, ink = _theme()
    order = [m for m in methods if m in set(segs["method"])]
    groups = [(m, t) for m in order for t in tasks]
    labels_outer = [m.replace("kern_cd_", "") for m, _ in groups]
    labels_inner = [t for _, t in groups]
    if include_harness:
        labels_outer = labels_outer + ["harness"] * len(tasks)
        labels_inner = labels_inner + list(tasks)
    pad = [0.0] * (len(labels_outer) - len(groups))

    present = [s for s in SEGMENTS + [SEG_VALID] if segs[segs["segment"] == s]["seconds"].sum() > 1e-9]
    fig = go.Figure()
    for seg in present:
        values = [
            float(segs[(segs["method"] == m) & (segs["task"] == t) & (segs["segment"] == seg)]["seconds"].sum())
            for m, t in groups
        ]
        fig.add_bar(
            y=[labels_outer, labels_inner],
            x=values + pad,
            name=seg,
            orientation="h",
            marker=dict(color=palette[seg], line=dict(color=ink["gap"], width=2)),
            hovertemplate="%{y}<br>" + seg + ": %{x:.3f} s<extra></extra>",
        )

    totals = [float(segs[(segs["method"] == m) & (segs["task"] == t)]["seconds"].sum()) for m, t in groups]
    if include_harness:
        harness_totals = [float(harness[harness["task"] == t]["seconds"].sum()) for t in tasks]
        fig.add_bar(
            y=[labels_outer, labels_inner],
            x=[0.0] * len(groups) + harness_totals,
            name="FIPER harness (context)",
            orientation="h",
            marker=dict(color=ink["neutral"], line=dict(color=ink["gap"], width=2)),
            hovertemplate="%{y}<br>harness: %{x:.2f} s<extra></extra>",
        )
        totals = totals + harness_totals

    span = max(totals) if totals else 1.0
    digits = 1 if span >= 1.0 else 2
    fig.add_scatter(
        y=[labels_outer, labels_inner],
        x=[v + span * 0.012 for v in totals],
        mode="text",
        text=[f"{v:,.{digits}f}s" for v in totals],
        textposition="middle right",
        textfont=dict(color=ink["muted"], size=11),
        showlegend=False,
        hoverinfo="skip",
    )

    if not include_harness:
        # Above the plot, not inside it: the two floors land within a hair of each other,
        # so an in-plot label would sit on top of both the bars and the other label.
        for i, task in enumerate(tasks):
            floor = float(segs[(segs["method"] == FLOOR_METHOD) & (segs["task"] == task)]["seconds"].sum())
            if floor <= 0:
                continue
            fig.add_vline(
                x=floor,
                line=dict(color=ink["muted"], width=1, dash="dot" if i else "dash"),
                annotation=dict(
                    text=f"obs floor, {task} ({floor:.{digits}f}s)",
                    yref="paper",
                    y=1.0 + 0.055 * i,
                    yanchor="bottom",
                    xanchor="left",
                    showarrow=False,
                    font=dict(color=ink["muted"], size=11),
                ),
            )

    fig.update_layout(
        barmode="stack",
        height=140 + 46 * len(labels_outer),
        # automargin (below) grows the left margin to fit the two-level category labels;
        # a fixed one clips "pretzel"/"push_t" against the method group they sit under.
        margin=dict(l=10, r=90, t=30 if include_harness else 66, b=90),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=ink["ink"], size=12),
        # traceorder="normal" so the legend reads in the same order the segments stack.
        legend=dict(orientation="h", yanchor="top", y=-0.16, x=0, traceorder="normal", font=dict(size=11)),
        bargap=0.28,
        bargroupgap=0.0,
    )
    fig.update_xaxes(
        title_text="seconds", gridcolor=ink["grid"], zerolinecolor=ink["grid"], linecolor=ink["grid"],
        range=[0, span * 1.14], title_font=dict(color=ink["muted"], size=11),
    )
    fig.update_yaxes(autorange="reversed", linecolor=ink["grid"], tickfont=dict(size=11), automargin=True)
    return fig


def harness_figure(harness: pd.DataFrame, tasks: list[str]):
    """Where FIPER's own time goes, per task -- the denominator the methods sit inside."""
    import plotly.graph_objects as go

    palette, ink = _theme()
    seams = list(harness.groupby("seam")["seconds"].sum().sort_values(ascending=False).index)
    colors = [palette[s] for s in SEGMENTS] + [palette[SEG_VALID], ink["neutral"]]
    fig = go.Figure()
    for i, seam in enumerate(seams):
        fig.add_bar(
            y=list(tasks),
            x=[float(harness[(harness["task"] == t) & (harness["seam"] == seam)]["seconds"].sum()) for t in tasks],
            name=seam,
            orientation="h",
            marker=dict(color=colors[i % len(colors)], line=dict(color=ink["gap"], width=2)),
            hovertemplate="%{y}<br>" + seam + ": %{x:.2f} s<extra></extra>",
        )
    fig.update_layout(
        barmode="stack",
        height=150 + 56 * len(tasks),
        margin=dict(l=10, r=30, t=20, b=100),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=ink["ink"], size=12),
        legend=dict(orientation="h", yanchor="top", y=-0.3, x=0, traceorder="normal", font=dict(size=11)),
        bargap=0.4,
    )
    fig.update_xaxes(title_text="seconds", gridcolor=ink["grid"], zerolinecolor=ink["grid"],
                     linecolor=ink["grid"], title_font=dict(color=ink["muted"], size=11))
    fig.update_yaxes(autorange="reversed", linecolor=ink["grid"], automargin=True)
    return fig


# ---------------------------------------------------------------------------- the app


def main_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="kern_cd profiling", layout="wide")
    st.title("Profiling the kern_cd family")
    st.caption(
        "The five methods share every line of code except `_parts`. This measures how total cost "
        "splits into a shared floor (`kern_cd_obs`, which runs the identical path with the action "
        "block removed) and a per-method delta -- only the delta is under the kernel design's control."
    )

    seams, meta = load_cache()

    with st.sidebar:
        st.header("Cache")
        if seams is None:
            st.warning("No cached profile.")
        else:
            st.success(f"{len(seams):,} seam rows")
            if meta is not None and len(meta):
                m = meta.iloc[0]
                st.caption(f"{m['generated']} · {m['gpu'] or m['device']} · torch {m['torch']} · seed {m['seed']}")
        tasks = st.multiselect("Tasks", list(TASKS), default=list(TASKS))
        methods = st.multiselect("Methods", METHODS, default=METHODS)
        if st.button("Run profile", type="primary", disabled=not (tasks and methods)):
            box = st.empty()
            with st.spinner("Running the real evaluation, instrumented..."):
                run_profile(tasks, methods, progress=lambda s: box.write(s))
            st.rerun()
        st.caption(
            "A run evaluates each method on each task through the real FIPER pipeline. Expect "
            "minutes, not seconds. Results are cached to CSV under `my_experiments/profiling/cache/`."
        )

    if seams is None:
        st.info("Press **Run profile** in the sidebar to generate the cache.")
        st.subheader("Scope")
        st.dataframe(pd.DataFrame(TASKS).T.rename_axis("task").reset_index(), width="stretch")
        return

    tasks_present = [t for t in TASKS if t in set(seams["task"])]
    segs = segment_table(seams)
    harness = harness_table(seams)
    dims = dims_table(seams)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "1 · Where the time goes",
            "2 · What is big",
            "3 · Peak memory",
            "4 · Scaling knobs",
            "Raw seams",
        ]
    )

    # ---------------------------------------------------------------- deliverable 1
    with tab1:
        st.subheader("Deliverable 1 — where the time goes")

        method_total = float(segs["seconds"].sum()) / max(1, len(segs.groupby(["task", "method"])))
        harness_total = float(harness["seconds"].sum()) / max(1, harness["task"].nunique())
        share = 100 * method_total / (method_total + harness_total)
        # Totals are the sum of the per-segment minima, matching the bars and the table
        # below rather than the min of the per-sweep totals, which is a different number.
        totals = segs.groupby(["task", "method"], as_index=False)["seconds"].sum()
        deltas = [
            (t, m, tot - float(segs[(segs["method"] == FLOOR_METHOD) & (segs["task"] == t)]["seconds"].sum()))
            for t, m, tot in zip(totals["task"], totals["method"], totals["seconds"])
            if m != FLOOR_METHOD
        ]
        worst = max(deltas, key=lambda r: r[2]) if deltas else ("", "", 0.0)

        c1, c2 = st.columns(2)
        c1.metric(
            "Method share of one evaluation",
            f"{share:.1f}%",
            help="averaged over the profiled (task, method) pairs",
        )
        c2.metric(
            "Costliest action channel, vs the obs floor",
            f"{worst[2] * 1e3:,.0f} ms",
            help=f"{worst[1]} on {worst[0]}",
        )
        st.markdown(
            f"The most expensive action channel of the five costs **{worst[2] * 1e3:,.0f} ms** more than "
            f"the observation-only floor ({worst[1]} on {worst[0]}), inside an evaluation that takes "
            f"**{method_total + harness_total:,.1f} s**. On these two tasks the choice of part is free, "
            "and the question is closed -- see Deliverable 4 for what changes at stacking's dimensions."
        )

        st.plotly_chart(
            deliverable1_figure(segs, harness, tasks_present, METHODS, include_harness=True),
            width="stretch",
            theme=None,
        )
        st.caption(
            "**In context.** The grey **harness** bar is FIPER's own cost -- dataset rebuild, the "
            f"{THRESHOLD_GRID[0]}x{THRESHOLD_GRID[1]}x{THRESHOLD_GRID[2]} threshold sweep, metrics. "
            "Without it the chart flatters the methods; with it, they are barely visible, which is "
            "itself the answer to whether optimising them is worth doing."
        )

        st.plotly_chart(
            deliverable1_figure(segs, harness, tasks_present, METHODS, include_harness=False),
            width="stretch",
            theme=None,
        )
        st.caption(
            "**The same bars, zoomed to the methods.** Method-major: each method's two tasks side "
            "by side, against the `kern_cd_obs` reference line -- the shared floor, measured rather "
            "than estimated, since obs runs the identical code path with the action block removed. "
            "**Unattributed** is wall clock minus the sum of the seams, shown rather than hidden: a "
            "large one would mean the segmentation is wrong."
        )

        wide = segs.pivot_table(index=["task", "method"], columns="segment", values="seconds", aggfunc="sum").fillna(0.0)
        wide = wide[[c for c in SEGMENTS + [SEG_VALID] if c in wide.columns]]
        wide["total"] = wide.sum(axis=1)
        wide = wide.reset_index()
        floors = {
            t: float(segs[(segs["method"] == FLOOR_METHOD) & (segs["task"] == t)]["seconds"].sum())
            for t in tasks_present
        }
        wide["vs obs floor"] = [tot - floors.get(t, float("nan")) for t, tot in zip(wide["task"], wide["total"])]
        wide = wide.merge(total_table(seams)[["task", "method", "spread"]], on=["task", "method"], how="left")
        st.dataframe(wide.round(3), width="stretch")
        st.caption(
            "`vs obs floor` is the cost of the action channel: the method's total minus the "
            "observation-only baseline on the same task. `spread` is the sweep-to-sweep range of "
            "the total -- read `vs obs floor` against it before believing any ordering."
        )

        st.markdown("**FIPER harness, per single-method evaluation**")
        st.plotly_chart(harness_figure(harness, tasks_present), width="stretch", theme=None)
        st.dataframe(
            harness.pivot_table(index="task", columns="seam", values="seconds", aggfunc="sum").round(2).reset_index(),
            width="stretch",
        )
        st.caption(
            "`dataset_build` is paid once per task however many methods run; the rest is averaged "
            "over the profiled methods, which is what a single-method run pays."
        )

        with st.expander("Drill-down: every seam (measure fine, display coarse)"):
            fine = add_segments(seams)
            fine = fine[fine["method"] != "__harness__"]
            keys = ["task", "method", "segment", "phase", "seam"]
            per_sweep = fine.groupby([*keys, "sweep"], as_index=False).agg(
                calls=("exclusive_s", "size"), seconds=("exclusive_s", "sum")
            )
            fine = (
                per_sweep.groupby(keys, as_index=False)
                .agg(calls=("calls", "median"), seconds=("seconds", "min"))
                .sort_values(["task", "method", "seconds"], ascending=[True, True, False])
            )
            st.dataframe(fine.round(4), width="stretch", height=420)

    # ---------------------------------------------------------------- deliverable 2
    with tab2:
        st.subheader("Deliverable 2 — what is big")
        st.caption(
            f"Every array above ~{INVENTORY_FLOOR_MB:.0f} MB. Symbolic shape says what breaks as tasks "
            "grow; concrete shape pins what is true today; lifetime is what governs peak memory. "
            "`live_at` names the moments at which the array is reachable, which is what makes the "
            "Deliverable 3 cross-check a sum over a *moment* rather than over a segment. MB here is "
            "10^6 bytes, matching the spec's own figures."
        )
        inv = inventory_table(dims, seams)
        c1, c2 = st.columns(2)
        task_sel = c1.selectbox("Task", tasks_present, key="inv_task")
        method_sel = c2.multiselect("Methods", METHODS, default=METHODS, key="inv_methods")
        view = inv[(inv["task"] == task_sel) & (inv["method"].isin(method_sel))]
        st.dataframe(
            view[["method", "pass_", "name", "site", "symbolic", "concrete", "dtype", "device", "MB", "lifetime", "note"]]
            .sort_values(["method", "MB"], ascending=[True, False]),
            width="stretch",
            height=520,
        )

        st.markdown("**Predictions from the spec, against what was measured**")
        for task in tasks_present:
            preds = predictions_table(seams, dims, task)
            if len(preds):
                st.caption(f"`{task}`")
                st.dataframe(preds, width="stretch")

        st.markdown("**Measured shapes at the seams**")
        io = seams[(seams["method"] != "__harness__") & (seams["out_shape"] != "")]
        io = io.groupby(["task", "method", "phase", "seam"], as_index=False).agg(
            in_shape=("in_shape", "last"), in_dtype=("in_dtype", "last"),
            out_shape=("out_shape", "last"), out_dtype=("out_dtype", "last"),
            out_is_view=("out_is_view", "last"),
            out_MB=("out_bytes", lambda s: round(float(s.iloc[-1]) / MB, 1)),
        )
        st.dataframe(io[io["out_MB"] > 1.0].sort_values("out_MB", ascending=False), width="stretch", height=340)

    # ---------------------------------------------------------------- deliverable 3
    with tab3:
        st.subheader("Deliverable 3 — peak memory")
        st.caption(
            "On its own a peak number supports no decision; its purpose is as a cross-check on "
            "Deliverable 2. Deltas are over the segment's entry watermark, so the resident dataset "
            "and the CUDA context do not swamp the comparison."
        )
        peaks = _peak_table(seams)
        st.dataframe(peaks.round(1), width="stretch", height=380)

        st.markdown("**Cross-check: named bytes vs measured CUDA peak**")
        st.dataframe(_crosscheck_table(seams, dims).round(1), width="stretch", height=340)
        st.caption(
            "`named_MB` is the sum of Deliverable 2's entries that are live at the segment's worst "
            f"moment ({', '.join(f'**{k}** -- {v}' for k, v in PEAK_MOMENTS.items())})."
        )
        st.caption(
            "The gap is memory allocated without being named, i.e. copies. Named suspects: the "
            "float64 host array behind `_chunks`, the `.cpu().numpy().astype(np.float64)` in "
            "`_embed`, and whatever `KernCD.fit` allocates internally. A large gap is reported as "
            "unaccounted-for, not explained away: without allocation-site tracking it can be "
            "measured but not attributed."
        )

    # ---------------------------------------------------------------- deliverable 4
    with tab4:
        st.subheader("Deliverable 4 — the scaling knobs")
        st.caption(
            "Two profiled tasks give two independent estimates of each stage's constant. Agreement "
            "is the check: a constant that moves by an order of magnitude between tasks means the "
            "stated complexity is not what the stage is doing."
        )
        st.dataframe(dims, width="stretch")
        model = cost_model_table(dims, segs)
        pivot = model.pivot_table(
            index=["stage", "complexity", "dominant", "method"], columns="task", values="ns_per_unit"
        )
        task_cols = [c for c in pivot.columns if c in tasks_present]
        if len(task_cols) > 1:
            ratio = pivot[task_cols].max(axis=1) / pivot[task_cols].min(axis=1)
            pivot["agreement"] = [
                "n/a" if not np.isfinite(r) else ("consistent" if r < 2 else f"{r:,.0f}x apart")
                for r in ratio
            ]
        st.dataframe(pivot.round(4).reset_index(), width="stretch", height=460)
        st.caption(
            "Values are nanoseconds per unit of the dominant term, so a stage whose stated "
            "complexity is right has the *same* constant on both tasks. Two caveats on the spec's "
            "formulae, adopted here: the Features term fitted is `n P F . 128` (the RFF matmul), "
            "which dominates the `n P . 128` cosine evaluations; and the scoring solve is O(N m^2) "
            "in total -- the 8192-row block sets arithmetic intensity, not flop count."
        )
        st.warning(
            "Read the `agreement` column before using any of these constants. At m of a few "
            "hundred and N of a few thousand, none of the four stages is anywhere near its "
            "asymptotic regime: what is being measured is Python, kernel-launch and BLAS-entry "
            "overhead, which is constant per call rather than proportional to the dominant term. "
            "Constants that disagree across the two tasks are not extrapolatable, and that is a "
            "result about the tasks, not a defect of the measurement."
        )

        st.markdown("**Extrapolation**")
        st.caption(
            "Predict the three unprofiled tasks by entering their dimensions. This is what makes "
            "the profile falsifiable: predict stacking from push_t, then run it and check."
        )
        base_task = st.selectbox("Fit constants on", tasks_present, index=len(tasks_present) - 1)
        base_method = st.selectbox("For method", sorted(set(dims["method"])), index=0)
        editable = pd.DataFrame(
            [dict(task=t, m=0, N=0, P=0, F=0, d_o=0) for t in ["push_chair", "sorting", "stacking"]]
        )
        entered = st.data_editor(editable, width="stretch", hide_index=True, key="extrapolate")
        st.dataframe(
            _extrapolate(model, entered, base_task, base_method).round(2), width="stretch"
        )

    # -------------------------------------------------------------------- raw seams
    with tab5:
        st.subheader("Raw seams")
        st.caption(
            "Everything measured, retained. Timings are exclusive of instrumented children, so the "
            "rows partition the run; `inclusive_s` is the same seam including them."
        )
        st.dataframe(seams, width="stretch", height=620)
        st.download_button("Download seams.csv", seams.to_csv(index=False), "seams.csv", "text/csv")


#: Seams whose subtree crosses a segment boundary: ``evaluate`` parents the entire run,
#: and ``score_subset`` (Scoring) parents ``_features`` (Features). A memory peak is
#: inclusive of children by construction -- that is what makes it a peak -- so charging
#: these two to their own segment would report the features peak twice, once under the
#: wrong name. They are dropped from the peak roll-up and kept in the raw table.
SPANNING_SEAMS = {"evaluate", "score_subset"}


def _peak_table(seams: pd.DataFrame) -> pd.DataFrame:
    """Peak host RSS and peak CUDA allocated/reserved per (task, method, segment)."""
    df = add_segments(seams)
    df = df[(df["method"] != "__harness__") & (~df["seam"].isin(SPANNING_SEAMS))]
    df["cuda_peak_MB"] = df["cuda_peak_b"] / MB
    df["cuda_delta_MB"] = (df["cuda_peak_b"] - df["cuda_in_b"]) / MB
    df["cuda_reserved_MB"] = df["cuda_reserved_peak_b"] / MB
    df["rss_peak_MB"] = df["rss_peak_b"] / MB
    df["rss_delta_MB"] = (df["rss_peak_b"] - df["rss_in_b"]) / MB
    return (
        df.groupby(["task", "method", "segment"], as_index=False)
        .agg(
            cuda_peak_MB=("cuda_peak_MB", "max"),
            cuda_delta_MB=("cuda_delta_MB", "max"),
            cuda_reserved_MB=("cuda_reserved_MB", "max"),
            rss_peak_MB=("rss_peak_MB", "max"),
            rss_delta_MB=("rss_delta_MB", "max"),
        )
        .sort_values(["task", "method", "segment"])
    )


def _crosscheck_table(seams: pd.DataFrame, dims: pd.DataFrame) -> pd.DataFrame:
    """Sum the catalogue's live device bytes *at each moment* and compare with the measurement.

    A peak is a moment, so the accounted number is the largest of the per-moment sums, not
    the sum of everything the segment ever allocated -- that would count freed arrays.
    ``worst moment`` names which one wins, which is where an optimisation would have to bite.
    """
    inv = inventory_table(seams=seams, dims=dims, floor_mb=0.0)
    device = inv[(inv["device"] == "cuda") & (inv["live_at"] != "")].copy()
    per_moment = []
    for (task, method, segment), g in device.groupby(["task", "method", "segment"]):
        moments = {m for row in g["live_at"] for m in row.split(" + ")}
        totals = {
            m: float(g[[m in row.split(" + ") for row in g["live_at"]]]["MB"].sum()) for m in moments
        }
        if not totals:
            continue
        worst = max(totals, key=totals.get)
        per_moment.append(
            dict(task=task, method=method, segment=segment, named_MB=totals[worst], worst_moment=worst)
        )
    named = pd.DataFrame(per_moment)

    peaks = _peak_table(seams)[["task", "method", "segment", "cuda_delta_MB"]]
    out = peaks.merge(named, on=["task", "method", "segment"], how="left").fillna(
        {"named_MB": 0.0, "worst_moment": ""}
    )
    out = out[out["cuda_delta_MB"] > 1.0]
    out["unaccounted_MB"] = out["cuda_delta_MB"] - out["named_MB"]
    return out.sort_values("cuda_delta_MB", ascending=False)


def _extrapolate(model: pd.DataFrame, entered: pd.DataFrame, base_task: str, base_method: str) -> pd.DataFrame:
    """Apply the constants fitted on one (task, method) to hand-entered dimensions."""
    base = model[(model["task"] == base_task) & (model["method"] == base_method)]
    consts = {r["stage"]: r["ns_per_unit"] for _, r in base.iterrows()}
    rows = []
    for _, e in entered.iterrows():
        if not (e["m"] and e["N"]):
            continue
        D = dict(m=int(e["m"]), N=int(e["N"]), P=int(e["P"]) or 1, F=int(e["F"]) or 1,
                 d_o=int(e["d_o"]), n_components=128)
        row = dict(task=e["task"])
        total = 0.0
        for stage, (_, _, term) in COST_MODEL.items():
            seconds = consts.get(stage, float("nan")) * term(D) / 1e9
            row[stage] = seconds
            total += 0.0 if np.isnan(seconds) else seconds
        row["predicted total (s)"] = total
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ CLI


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Headless (re)generation of the profiling cache.")
    parser.add_argument("--run", action="store_true", help="run the profile and write the CSV cache")
    parser.add_argument("--force", action="store_true", help="overwrite an existing cache")
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--repeats", type=int, default=6, help="sweeps per task; the first is discarded as warm-up")
    args = parser.parse_args(argv)

    if not args.run:
        parser.print_help()
        print("\nFor the app:  streamlit run my_experiments/profiling/profiling.py")
        return 0
    if SEAMS_CSV.exists() and not args.force:
        print(f"{SEAMS_CSV} exists; pass --force to overwrite.")
        return 0

    t0 = time.perf_counter()
    df = run_profile(
        args.tasks, args.methods, repeats=args.repeats, progress=lambda s: print(f"  {s}", flush=True)
    )
    print(f"\n{len(df):,} seam rows -> {SEAMS_CSV}  ({time.perf_counter() - t0:.1f}s)")
    print(segment_table(df).pivot_table(index=["task", "method"], columns="segment", values="seconds").round(2))
    return 0


if __name__ == "__main__":
    # `streamlit run` executes this file as __main__ with no extra argv, so the presence
    # of a flag is what separates the headless regeneration from the app.
    if any(a.startswith("--") for a in sys.argv[1:]):
        raise SystemExit(main())
    main_app()
