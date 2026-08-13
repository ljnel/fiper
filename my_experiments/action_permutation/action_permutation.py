"""Permutation control: does the action channel carry rollout-specific information?

The kern_cd action methods all score at or below ``kern_cd_obs``, which has no action
channel at all. That is consistent with two different situations, which TWA alone cannot
separate:

* **inert** -- the action factor ``exp(-||mu_A - mu_A'||^2 / 2 sigma_A^2)`` is ~1 for every
  pair, so the product kernel *is* the observation kernel and the action channel is
  decoration;
* **active but uninformative** -- the factor varies over pairs, so it genuinely reshapes
  the kernel, but what it encodes is either noise or already carried by the observations.

This script separates them with two measurements per method:

1. **A permutation control.** ``_pack`` is the one place where an observation row and an
   action row are joined, so overriding it to shuffle ``mu``'s rows breaks that pairing and
   nothing else: the multiset of mean embeddings is untouched, so sigma_A (a median over
   pairwise distances) is bit-identical, and the action factor keeps exactly the
   distribution of values it had. Only *which* observation each action is attached to
   changes. If shuffling costs nothing, the pairing carries nothing.
2. **The mean off-diagonal action factor** on the fit set, which says whether the factor
   was doing anything to shuffle in the first place. ~1.0 means inert.

Three independent permutations per method give the control its own error bar, and
``kern_cd_obs`` is evaluated alongside as the no-channel floor.

85 evaluations (4 action methods x (1 real + 3 permutations) + obs, on 5 tasks), model
seed 0 throughout; the seed noise floor for this footing is 0.0058 TWA, measured in
``my_experiments/rff_sensitivity/``. Arms are evaluated in one grouped
``run._evaluate_task`` call per task, which builds the processed dataset once.

    pixi run python my_experiments/action_permutation/action_permutation.py
    pixi run python my_experiments/action_permutation/action_permutation.py --report

Per-arm metric grids are cached under ``my_experiments/action_permutation/cache/``;
``--force`` re-runs them. Nothing in ``my_methods/`` or ``data/results/`` is modified, but
FIPER's eval class writes a pickle per arm into ``data/<task>/results/<arm>/``; every arm
here is named ``perm_*``, so that litter is removable with ``rm -rf data/*/results/perm_*``.
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
OBS = "kern_cd_obs"
ACTION_METHODS = ["kern_cd_flat", "kern_cd_sum", "kern_cd_disp", "kern_cd_sig"]
TASKS = ["push_t", "sorting", "stacking", "pretzel", "push_chair"]
PERMS = (0, 1, 2)
SEED = 0
#: Seed-to-seed range of this benchmark score at a fixed configuration, from
#: my_experiments/rff_sensitivity/ (kern_cd_flat, 7 dimensions x 3 seeds).
NOISE = 0.0058

#: arm name -> mean off-diagonal action factor on the fit set. Populated during fit.
_FACTOR: dict[str, float] = {}


def _mean_action_factor(mu: np.ndarray, sigma: float, cap: int = 800) -> float:
    """Mean off-diagonal ``exp(-||mu - mu'||^2 / 2 sigma^2)`` on the fit set.

    How much work the action factor does, in one number: the product kernel multiplies it
    into every entry, so a factor whose mean is ~1 has been switched off and the method has
    silently reverted to ``kern_cd_obs``.
    """
    from sklearn.metrics.pairwise import euclidean_distances

    A = np.ascontiguousarray(mu[:cap], dtype=np.float64)
    K = np.exp(-euclidean_distances(A, A, squared=True) / (2.0 * sigma**2))
    n = len(A)
    return float((K.sum() - np.trace(K)) / (n * (n - 1)))


class _ProbeMixin:
    """Records the action factor once, on the fit set."""

    def _pack(self, obs, mu):
        if mu is not None and self.name not in _FACTOR:
            _FACTOR[self.name] = _mean_action_factor(mu, self._sigma_A)
        return super()._pack(obs, mu)


class _ShuffleMixin(_ProbeMixin):
    """Breaks the observation<->action pairing by permuting ``mu``'s rows.

    ``_pack`` runs once on the fit set and once per scored subset, and each call sees that
    whole set at once (``score_subset`` scores a subset, not a rollout), so the permutation
    mixes across episodes rather than within one. It is drawn from the arm's own seed and
    the row count, so an arm is reproducible and the three permutation arms are genuinely
    three different permutations.

    Row order is the *only* thing that changes. sigma_A is a median over pairwise distances
    of the same multiset, so it is unchanged; the action factor keeps its distribution; the
    observation block is untouched. What the method loses is exclusively the association.
    """

    perm_seed = 0

    def _pack(self, obs, mu):
        if mu is not None:
            rng = np.random.default_rng([int(self.perm_seed), len(mu)])
            mu = mu[rng.permutation(len(mu))]
        return super()._pack(obs, mu)


def arm_name(base: str, perm: int | None) -> str:
    """``perm_flat__real`` for the method as specified, ``perm_flat__shuf0`` for a control."""
    tag = "real" if perm is None else f"shuf{perm}"
    return f"perm_{base.removeprefix('kern_cd_')}__{tag}"


def make_arm(base: str, perm: int | None):
    """Register (once) the variant class implementing one arm."""
    name = arm_name(base, perm)
    if name in REGISTRY:
        return REGISTRY[name]
    mixin = _ProbeMixin if perm is None else _ShuffleMixin
    attrs = {"name": name} if perm is None else {"name": name, "perm_seed": int(perm)}
    return type(f"Arm{name.title().replace('_', '')}", (mixin, REGISTRY[base]), attrs)


ARMS: list[tuple[str, int | None]] = (
    [(OBS, None)] + [(m, p) for m in ACTION_METHODS for p in [None, *PERMS]]
)


def headline(grids: dict[str, pd.DataFrame]) -> pd.Series:
    """Benchmark metrics for one arm at FIPER's footing, from its five per-task grids.

    ``configs/results/base.yaml`` averages over Quantile *and Task* before taking the best
    Window and Threshold, so a method gets one window for the whole benchmark rather than
    one per task. Verified against the results store: this reproduces kern_cd_flat's
    published 0.6845 and its per-task numbers exactly.
    """
    df = pd.concat(grids.values(), ignore_index=True)
    per_task = df.groupby(["Threshold", "Window", "Task"], as_index=False)[["TWA", "Accuracy", "Det. Time"]].mean()
    avg = per_task.groupby(["Threshold", "Window"], as_index=False)[["TWA", "Accuracy", "Det. Time"]].mean()
    best = avg.loc[avg["TWA"].idxmax()]
    chosen = per_task[(per_task["Threshold"] == best["Threshold"]) & (per_task["Window"] == best["Window"])]
    out = {"TWA": float(best["TWA"]), "Accuracy": float(best["Accuracy"]), "Det. Time": float(best["Det. Time"]),
           "TWA_gridmean": float(df["TWA"].mean())}
    out.update({f"TWA__{r.Task}": float(r.TWA) for r in chosen.itertuples()})
    return pd.Series(out)


def _cache_path(task: str, name: str) -> str:
    return os.path.join(CACHE_DIR, f"{task}__{name}__seed{SEED}.csv")


def _factor_path(task: str) -> str:
    return os.path.join(CACHE_DIR, f"{task}__factors__seed{SEED}.csv")


def evaluate(task: str, force: bool, cache_only: bool) -> dict[str, pd.DataFrame]:
    """Evaluate every arm not already cached on one task, and return all their grids."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    todo: list[tuple[str, int | None]] = []
    for base, perm in ARMS:
        name = arm_name(base, perm)
        path = _cache_path(task, name)
        if os.path.exists(path) and not force:
            # Window is a label: read back as int it stops matching the str a fresh grid
            # carries, and the task silently drops out of the group-by.
            out[name] = pd.read_csv(path, dtype={"Window": str})
        else:
            todo.append((base, perm))
    if cache_only or not todo:
        return out

    print(f"\nevaluating {len(todo)} arm(s) on {task} ({len(out)} cached) ...", flush=True)
    started = time.time()
    names = [arm_name(b, p) for b, p in todo]
    for base, perm in todo:
        make_arm(base, perm)
    try:
        results = run_mod._evaluate_task(names, task, SEED, {})
    except Exception as exc:  # noqa: BLE001 -- a group that cannot be evaluated is a result
        print(f"  !! {task} failed: {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
        return out
    for name in names:
        df = pd.DataFrame(run_mod._rows(name, task, results))
        df.to_csv(_cache_path(task, name), index=False)
        out[name] = df
    if _FACTOR:
        pd.DataFrame([{"arm": k, "k_A": v} for k, v in _FACTOR.items()]).to_csv(_factor_path(task), index=False)
        _FACTOR.clear()
    print(f"  done in {time.time() - started:.0f}s", flush=True)
    return out


def collect(force: bool = False, cache_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per arm with its benchmark headline, plus the per-task action factors."""
    grids: dict[str, dict[str, pd.DataFrame]] = {arm_name(b, p): {} for b, p in ARMS}
    for task in TASKS:
        for name, df in evaluate(task, force, cache_only).items():
            grids[name][task] = df
    rows = []
    for base, perm in ARMS:
        name = arm_name(base, perm)
        if len(grids[name]) == len(TASKS):  # an arm short of a task is not on the same footing
            rows.append({"arm": name, "base": base, "perm": -1 if perm is None else perm,
                         **headline(grids[name]).to_dict()})
    factors = pd.concat(
        [pd.read_csv(_factor_path(t)).assign(task=t) for t in TASKS if os.path.exists(_factor_path(t))],
        ignore_index=True) if any(os.path.exists(_factor_path(t)) for t in TASKS) else pd.DataFrame()
    return pd.DataFrame(rows), factors


def report(table: pd.DataFrame, factors: pd.DataFrame) -> None:
    """Print real vs permuted vs no-channel, per method, against the seed noise floor."""
    obs = table[table.base == OBS]
    obs_twa = float(obs["TWA"].iloc[0]) if not obs.empty else np.nan

    rows = []
    for base in ACTION_METHODS:
        real = table[(table.base == base) & (table.perm < 0)]["TWA"]
        shuf = table[(table.base == base) & (table.perm >= 0)]["TWA"]
        if real.empty or shuf.empty:
            continue
        k = factors[factors.arm == arm_name(base, None)]["k_A"] if not factors.empty else pd.Series(dtype=float)
        rows.append({
            "method": base.removeprefix("kern_cd_"),
            "k_A": k.mean() if not k.empty else np.nan,
            "real": float(real.iloc[0]),
            "shuffled": shuf.mean(),
            "shuf_range": shuf.max() - shuf.min(),
            "d_shuf": float(real.iloc[0]) - shuf.mean(),
            "d_obs": float(real.iloc[0]) - obs_twa,
        })
    df = pd.DataFrame(rows)

    print("\n=== benchmark TWA (mean over 5 tasks), model seed 0 ===")
    print(f"    kern_cd_obs, no action channel at all: {obs_twa:.4f}")
    print(f"    seed noise floor for this score: {NOISE:.4f} TWA\n")
    print("    k_A = mean off-diagonal action factor on the fit set; 1.0 = the factor is inert")
    print("    shuffled = mean over 3 independent permutations of the obs<->action pairing\n")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== reading ===")
    for r in df.itertuples():
        verdict = ("factor is inert (k_A ~ 1): the kernel is already the obs-only kernel"
                   if r.k_A > 0.99 else
                   ("pairing matters: shuffling costs more than seed noise" if abs(r.d_shuf) > NOISE else
                    "factor is active but the pairing carries nothing: shuffling is free"))
        print(f"  {r.method:<5} k_A {r.k_A:.3f}  real-shuf {r.d_shuf:+.4f}  ->  {verdict}")

    print("\n=== per task, real vs shuffled (mean over 3 permutations) ===")
    detail = []
    for base in ACTION_METHODS:
        for tag, sel in (("real", table.perm < 0), ("shuf", table.perm >= 0)):
            sub = table[(table.base == base) & sel]
            if sub.empty:
                continue
            row = {"method": base.removeprefix("kern_cd_"), "arm": tag}
            row.update({t: f"{sub[f'TWA__{t}'].mean():.3f}" for t in TASKS})
            row["AVG"] = f"{sub['TWA'].mean():.4f}"
            detail.append(row)
    if not obs.empty:
        detail.append({"method": "obs", "arm": "--",
                       **{t: f"{float(obs[f'TWA__{t}'].iloc[0]):.3f}" for t in TASKS},
                       "AVG": f"{obs_twa:.4f}"})
    print(pd.DataFrame(detail).to_string(index=False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-run arms even if cached")
    parser.add_argument("--report", action="store_true", help="report cached arms only, evaluate nothing")
    args = parser.parse_args(argv)

    discover()
    table, factors = collect(force=args.force, cache_only=args.report)
    if table.empty:
        print("nothing cached yet", file=sys.stderr)
        return 1
    table.to_csv(str(HERE / "action_permutation.csv"), index=False)
    report(table, factors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
