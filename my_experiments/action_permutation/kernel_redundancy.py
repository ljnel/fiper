"""Is the action kernel factor a second view, or a copy of the observation factor?

The permutation control (``action_permutation.py``) leaves two facts that look
contradictory: deleting the action channel costs nothing (every action method scores within
noise of ``kern_cd_obs``), yet scrambling the observation<->action pairing costs 0.12 TWA.
Both hold at once if the action factor is *informative but redundant* -- a near-duplicate
of the observation factor, so that removing it loses nothing while corrupting it damages a
kernel that was previously self-consistent.

That is a measurable claim about the two factors of the product kernel, not an
interpretation. This script fits each method and reports, over the same fit set:

* ``corr`` -- Pearson correlation between the off-diagonal entries of ``k_o`` and ``k_A``.
  Redundancy is exactly this being high.
* ``R2`` -- how much of ``k_A``'s variation over pairs a monotone-free linear fit on ``k_o``
  explains, i.e. the same thing in units of variance rather than correlation.
* ``k_o``, ``k_A`` -- the two factors' mean off-diagonal values, for scale.

The mechanism to expect: the action chunk is the policy's output *on that observation*, so
mu_A is close to a deterministic function of the observation embedding, and a kernel on it
is close to a reparameterised kernel on the observation.

20 evaluations (4 action methods x 5 tasks, model seed 0). Only the fit set is used, but
the harness runs a full evaluation to produce it.

    pixi run python my_experiments/action_permutation/kernel_redundancy.py

Stats are cached under ``my_experiments/action_permutation/cache/``; ``--force`` re-runs.
Litter is removable with ``rm -rf data/*/results/red_*``.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from my_methods import run as run_mod  # noqa: E402
from my_methods.base import REGISTRY, discover  # noqa: E402

CACHE_DIR = str(HERE / "cache")
ACTION_METHODS = ["kern_cd_flat", "kern_cd_sum", "kern_cd_disp", "kern_cd_sig"]
TASKS = ["push_t", "sorting", "stacking", "pretzel", "push_chair"]
SEED = 0
#: Pairs are capped so the two m x m factors stay small; 800 rows is 319,600 off-diagonal
#: pairs, far more than needed to pin a correlation.
CAP = 800

_STATS: list[dict] = []


def _factor(X: np.ndarray, sigma: float) -> np.ndarray:
    """Off-diagonal entries of ``exp(-||x-x'||^2 / 2 sigma^2)`` on the fit set, flattened."""
    from sklearn.metrics.pairwise import euclidean_distances

    A = np.ascontiguousarray(X[:CAP], dtype=np.float64)
    K = np.exp(-euclidean_distances(A, A, squared=True) / (2.0 * sigma**2))
    iu = np.triu_indices(len(A), k=1)
    return K[iu]


class _RedundancyMixin:
    """Records the two kernel factors' agreement once, on the fit set."""

    def _pack(self, obs, mu):
        if mu is not None and not any(s["arm"] == self.name for s in _STATS):
            ko, ka = _factor(obs, self._sigma_o), _factor(mu, self._sigma_A)
            corr = float(np.corrcoef(ko, ka)[0, 1])
            _STATS.append({"arm": self.name, "k_o": float(ko.mean()), "k_A": float(ka.mean()),
                           "corr": corr, "R2": corr**2, "m": int(min(len(obs), CAP))})
        return super()._pack(obs, mu)


def arm_name(base: str) -> str:
    return f"red_{base.removeprefix('kern_cd_')}"


def make_arm(base: str):
    """Register (once) the probe variant of one method."""
    name = arm_name(base)
    if name in REGISTRY:
        return REGISTRY[name]
    return type(f"Arm{name.title().replace('_', '')}", (_RedundancyMixin, REGISTRY[base]), {"name": name})


def _cache_path(task: str) -> str:
    return os.path.join(CACHE_DIR, f"{task}__redundancy__seed{SEED}.csv")


def collect(force: bool) -> pd.DataFrame:
    """Fit every action method on every task and return the per-(task, method) agreement stats."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    frames = []
    for task in TASKS:
        path = _cache_path(task)
        if os.path.exists(path) and not force:
            frames.append(pd.read_csv(path))
            continue
        print(f"\nfitting {len(ACTION_METHODS)} method(s) on {task} ...", flush=True)
        started = time.time()
        for base in ACTION_METHODS:
            make_arm(base)
        _STATS.clear()
        run_mod._evaluate_task([arm_name(b) for b in ACTION_METHODS], task, SEED, {})
        df = pd.DataFrame(_STATS).assign(task=task)
        df.to_csv(path, index=False)
        frames.append(df)
        print(f"  done in {time.time() - started:.0f}s", flush=True)
    return pd.concat(frames, ignore_index=True)


def report(df: pd.DataFrame) -> None:
    """Print the agreement between the two kernel factors, per method and per task."""
    df = df.assign(method=df["arm"].str.removeprefix("red_"))
    print("\n=== agreement between the observation factor k_o and the action factor k_A ===")
    print("    over the off-diagonal pairs of each method's own fit set, model seed 0")
    print("    corr ~ 1 means the action factor is a copy of the observation factor\n")
    piv = df.pivot_table(index="method", columns="task", values="corr")[TASKS]
    piv["mean"] = piv.mean(axis=1)
    print(piv.to_string(float_format=lambda v: f"{v:.3f}"))

    by_m = df.groupby("method").agg(k_o=("k_o", "mean"), k_A=("k_A", "mean"),
                                    corr=("corr", "mean"), R2=("R2", "mean"))
    print("\n    averaged over tasks:\n")
    print(by_m.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n  the action factor's variation over pairs is {100 * by_m['R2'].mean():.0f}% "
          f"linearly explained by the observation factor, on average")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-fit even if cached")
    args = parser.parse_args(argv)
    discover()
    df = collect(force=args.force)
    df.to_csv(str(HERE / "kernel_redundancy.csv"), index=False)
    report(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
