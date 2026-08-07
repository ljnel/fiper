"""Runner for methods defined in ``my_methods/methods/``.

    python -m my_methods.run <name> --task push_t        # quick: one task, one seed
    python -m my_methods.run <name> [<name> ...] --full  # all tasks, all seeds
    python -m my_methods.run --list                      # show registered methods

Quick mode is the iteration loop: it loads the cached processed dataset, evaluates one
task with one seed, and prints the metric table directly from the returned results. It
deliberately does not touch ``data/results/`` -- writing there triggers a rescan of the
whole (multi-GB) results store.

``--full`` reproduces what ``scripts/run_fiper.py`` does: every task, every seed from
``eval/base.yaml``, results merged into ``data/results/`` so the methods show up in
``complete_results.csv`` alongside the built-in ones.

Naming several methods at once is much cheaper than running them one at a time. The
processed dataset is built once per task and shared (~12 s per avoided rebuild), and
``combine_results`` -- which unpickles the entire results store to rebuild the CSV --
runs once instead of once per method.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import pandas as pd  # noqa: E402

from evaluation import EvaluationManager, ResultsManager  # noqa: E402
from shared_utils.hydra_utils import load_config  # noqa: E402
from shared_utils.utility_functions import set_seed  # noqa: E402
from tasks import TaskManager  # noqa: E402

from my_methods.base import REGISTRY, build_cfg, discover, make_eval_class  # noqa: E402

BASE_CONFIG_PATH = os.path.join(ROOT_DIR, "configs")
BASE_DATA_PATH = os.path.join(ROOT_DIR, "data")

ALL_TASKS = ["push_t", "pretzel", "push_chair", "sorting", "stacking"]


class _RegistryEvaluationManager(EvaluationManager):
    """``EvaluationManager`` that resolves methods from ``REGISTRY`` instead of disk.

    Overriding these two hooks is what removes the need for a ``configs/eval/*.yaml``
    file and for the ``{METHOD}Eval`` class-naming convention. Built-in methods still
    resolve through the parent implementation.
    """

    def __init__(self, *args, param_overrides=None, **kwargs):
        self._param_overrides = param_overrides or {}
        super().__init__(*args, **kwargs)

    def _load_config(self, method_name: str = "base"):
        if method_name in REGISTRY:
            return build_cfg(REGISTRY[method_name], self.base_config_path, self._param_overrides)
        return super()._load_config(method_name)

    def _get_method_eval_class(self, method_name: str, cfg):
        if method_name not in REGISTRY:
            return super()._get_method_eval_class(method_name, cfg)
        method_cls = REGISTRY[method_name]
        params = dict(method_cls.params)
        params.update(self._param_overrides)
        # evaluate() injects the run seed into cfg.hparams.model when the method declares
        # a `seed` param (evaluation_manager.py:70). Read the params back from cfg so the
        # method sees the seed actually being run, not the declared placeholder. Every
        # other key round-trips unchanged -- build_cfg wrote hparams.model from params.
        model_hparams = cfg.get("hparams", {}).get("model", {})
        for key in params:
            if key in model_hparams:
                params[key] = model_hparams[key]
        eval_cls = make_eval_class(method_cls, params)
        return eval_cls(
            cfg=cfg,
            method_name=method_name,
            device=self.device,
            task_data_path=self.task_data_path,
            dataset=self.dataset,
            **self.kwargs,
        )


def _evaluate_task(names: list[str], task: str, seed: int, param_overrides: dict) -> dict:
    """Evaluate the named methods on one task with one seed.

    Returns FIPER's ``{method_name: results}`` dict. The dataset is built once and shared
    across the methods, so the tensors requested are the union of what they declare --
    each method's own config still selects from that, so an obs-only method is unaffected
    by an action-channel method being in the same call.
    """
    set_seed(seed)
    cfg = load_config("task", task, return_only_subdict=False)
    task_data_path = os.path.join(BASE_DATA_PATH, task)

    required = _union(names, "tensors")
    optional = _union(names, "optional_tensors")
    taskmanager = TaskManager(
        cfg,
        task,
        BASE_CONFIG_PATH,
        task_data_path,
        required_tensors=required,
        optional_tensors=optional,
        device=_device(),
    )
    # load_dataset_if_exists=False matches scripts/run_fiper.py. Loading the cached
    # processed_rollouts/*.pt looks tempting but is broken upstream: load_dataset()
    # appends ".pt" to the dict *key* (rollout_datasets.py:499-511), so tensors arrive
    # as "obs_embeddings.pt" and every lookup fails. Re-converting costs ~1s (sorting)
    # to ~5s (stacking), so there is nothing to gain by working around it.
    dataset = taskmanager.get_rollout_dataset(load_dataset_if_exists=False)

    manager = _RegistryEvaluationManager(
        BASE_CONFIG_PATH,
        task_data_path,
        dataset,
        device=_device(),
        seed=seed,
        param_overrides=param_overrides,
    )
    return manager.evaluate(list(names), combine_methods=False)


def _union(names: list[str], attr: str) -> list[str]:
    """Declared tensors across ``names``, in first-seen order."""
    out: list[str] = []
    for name in names:
        for item in getattr(REGISTRY[name], attr):
            if item not in out:
                out.append(item)
    return out


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _rows(name: str, task: str, results: dict) -> list[dict]:
    """Flatten test_metrics into rows using complete_results.csv's column names."""
    method_results = results[name]
    rows = []
    for threshold_style, by_quantile in method_results["test_metrics"].items():
        for quantile, by_window in by_quantile.items():
            for window_size, metrics in by_window.items():
                rows.append(
                    {
                        "Method": name,
                        "Task": task,
                        "TWA": round(float(metrics["TWA"]), 3),
                        "Accuracy": round(float(metrics["balanced_accuracy"]), 3),
                        "Det. Time": round(float(metrics["avg_detection_time"]), 3),
                        "TPR": round(float(metrics["TPR"]), 3),
                        "TNR": round(float(metrics["TNR"]), 3),
                        "Window": str(window_size),
                        "Quantile": float(quantile),
                        "Threshold": str(threshold_style),
                    }
                )
    return rows


def _report(name: str, task: str, results: dict) -> pd.DataFrame:
    """Print the best configuration per threshold style and save the full grid."""
    df = pd.DataFrame(_rows(name, task, results))
    best = df.sort_values("TWA", ascending=False).groupby("Threshold", as_index=False).first()
    best = best[["Threshold", "TWA", "Accuracy", "Det. Time", "TPR", "TNR", "Window", "Quantile"]]

    print(f"\n=== {name} on {task} -- best configuration per threshold style ===")
    print(best.sort_values("TWA", ascending=False).to_string(index=False))
    print(f"\navg inference time: {results[name]['avg_inference_time'] * 1e3:.3f} ms/step")

    out_dir = os.path.join(BASE_DATA_PATH, task, "results", name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "quick_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"full grid ({len(df)} rows) -> {os.path.relpath(out_path, ROOT_DIR)}")
    return df


def _parse_params(pairs: list[str]) -> dict:
    """Parse ``--param key=value`` overrides, coercing values with literal_eval."""
    import ast

    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--param expects key=value, got '{pair}'")
        key, _, raw = pair.partition("=")
        try:
            overrides[key.strip()] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            overrides[key.strip()] = raw  # plain string
    return overrides


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="my_methods.run", description=__doc__)
    parser.add_argument("methods", nargs="*", help="registered method name(s)")
    parser.add_argument("--task", default="push_t", choices=ALL_TASKS, help="task for quick mode")
    parser.add_argument("--full", action="store_true", help="all tasks, all seeds, into data/results/")
    parser.add_argument("--seed", type=int, default=0, help="seed for quick mode")
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        metavar="N",
        help="--full: use only the first N seeds (overrides the methods' `deterministic`)",
    )
    parser.add_argument("--param", action="append", default=[], metavar="K=V", help="override a params entry")
    parser.add_argument("--list", action="store_true", help="list registered methods and exit")
    args = parser.parse_args(argv)

    discover()

    if args.list or not args.methods:
        print("Registered methods:")
        for name, cls in sorted(REGISTRY.items()):
            print(f"  {name:<20} {cls.__module__}")
        return 0

    unknown = [m for m in args.methods if m not in REGISTRY]
    if unknown:
        print(f"Unknown method(s) {unknown}. Registered: {sorted(REGISTRY)}", file=sys.stderr)
        return 1

    overrides = _parse_params(args.param)
    if overrides:
        print(f"param overrides: {overrides}")

    if not args.full:
        results = _evaluate_task(args.methods, args.task, args.seed, overrides)
        for name in args.methods:
            _report(name, args.task, results)
        return 0

    # Full run: mirrors scripts/run_fiper.py.
    base_cfg = load_config("eval", "base", return_only_subdict=True, base_config_dir=BASE_CONFIG_PATH)
    seeds = list(base_cfg.get("random_seeds", [0, 1, 2]))
    if args.seeds is not None:
        seeds = seeds[: max(1, args.seeds)]
    elif all(REGISTRY[m].deterministic for m in args.methods):
        # Repeated seeds of a deterministic method reproduce the same numbers exactly, so
        # the remaining passes cost time and change nothing. One stochastic method in the
        # call is enough to need them all -- the seed loop is shared.
        seeds = seeds[:1]
        print(f"all methods are declared deterministic -- running 1 seed instead of {len(base_cfg.random_seeds)}")
    all_seed_results = []
    for seed in seeds:
        print(f"-------------- Seed: {seed} ----------------")
        per_task = {}
        for task in ALL_TASKS:
            print(f"-------------- Task: {task} ----------------")
            per_task[task] = _evaluate_task(args.methods, task, seed, overrides)
        all_seed_results.append(per_task)

    resultsmanager = ResultsManager(BASE_CONFIG_PATH, BASE_DATA_PATH)
    total_results = resultsmanager.accumulate_seed_results(all_seed_results)
    resultsmanager.combine_results(total_results, method_names=list(args.methods))
    resultsmanager.create_summary()
    print(f"\n{', '.join(args.methods)} merged into data/results/complete_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
