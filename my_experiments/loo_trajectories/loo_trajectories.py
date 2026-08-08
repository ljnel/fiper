"""Leave-one-*trajectory*-out calibration for kern_cd -- ``specs/experiments/loo_trajectories.md``.

The problem
-----------
``KernCDMethod._loo_scores`` (my_methods/kern_cd_core.py) calibrates with the textbook
leave-one-*point*-out identity

    s_i^{(-i)} = 1 / B_ii - lam*m,      B = (K + lam*m*I)^{-1}.

The fit set is a pool of *time steps*, so holding out step ``i`` leaves every other step of
the same rollout in the fit set. Consecutive steps of one trajectory are near-duplicates,
so the held-out point is still all but interpolated by its own neighbours: the LOO score is
barely larger than the in-sample score, the calibration quantile is too low, and the test
threshold is too permissive.

The fix
-------
Leave out the whole trajectory. For the block ``G`` of every step of one episode,

    s_i^{(-G)} = K_ii - K_{i,-G}^T (K_{-G,-G} + lam*m*I)^{-1} K_{i,-G},   i in G.

That has a closed form as cheap as the point version. With ``M = K + lam*m*I`` and
``B = M^{-1}``, the Schur complement of the block partition gives

    M_GG - M_{G,-G} M_{-G,-G}^{-1} M_{-G,G} = (B_GG)^{-1},

and since ``M_{i,-G} = K_{i,-G}`` (the ridge only touches the diagonal, and ``i`` is not in
``-G``), reading off the i-th diagonal entry gives

    s_i^{(-G)} = (B_GG^{-1})_{ii} - lam*m.

For a singleton block this is ``1/B_ii - lam*m``, the existing identity. So the whole change
is: invert the ``n_e x n_e`` diagonal block of ``B`` belonging to each episode instead of
taking one reciprocal per row. Both forms are checked against a brute-force refit below.

Nothing else moves: the estimator, the kernel, the bandwidths and the *test* scores are
untouched, because the trajectory grouping only exists inside the fit set. The only channel
through which this can change a headline number is the calibration threshold.

Running it
----------
    pixi run python my_experiments/loo_trajectories/loo_trajectories.py                  # push_chair, obs only
    pixi run python my_experiments/loo_trajectories/loo_trajectories.py --task pretzel
    pixi run python my_experiments/loo_trajectories/loo_trajectories.py --methods kern_cd_obs kern_cd_disp
    pixi run python my_experiments/loo_trajectories/loo_trajectories.py --check-only     # just the LOO algebra
    pixi run python my_experiments/loo_trajectories/loo_trajectories.py --summary        # roll up every cached grid

Per-(task, method) grids are cached as CSVs under ``my_experiments/loo_trajectories/cache/``
and reused; ``--force`` re-runs. Nothing under ``data/results/`` is touched.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.linalg import cho_factor, cho_solve, solve_triangular  # noqa: E402

from my_methods import run as run_mod  # noqa: E402
from my_methods.base import REGISTRY, discover  # noqa: E402
from my_methods.kern_cd_core import KernCDMethod  # noqa: E402

CACHE_DIR = str(HERE / "cache")

#: Suffix appended to a base method's name to get its trajectory-LOO twin.
SUFFIX = "_trajloo"


# --------------------------------------------------------------------------- the estimator
def block_loo_scores(L: np.ndarray, lam_m: float, group_sizes: np.ndarray) -> np.ndarray:
    """Leave-one-block-out support scores from the Cholesky factor of ``K + lam_m*I``.

    Args:
        L: lower Cholesky factor, ``[m, m]``.
        lam_m: the ridge actually on the diagonal, i.e. ``lam * m``.
        group_sizes: block sizes in fit-set row order; must sum to ``m``.

    Returns:
        ``[m]`` scores, each the support score of that row with its whole block held out.

    ``B = (L L^T)^{-1} = L^{-T} L^{-1}`` is formed once (one ``[m, m]`` triangular inverse
    and one symmetric product), then each block's ``n_e x n_e`` diagonal sub-matrix is
    inverted on its own. The block inverses cost ``sum_e n_e^3 << m^3``, so this is not
    measurably more expensive than the point version, which reads the diagonal of the same
    ``B``.
    """
    m = L.shape[0]
    assert int(np.sum(group_sizes)) == m, f"blocks sum to {np.sum(group_sizes)}, need {m}"
    L_inv = solve_triangular(L, np.eye(m), lower=True)
    B = L_inv.T @ L_inv

    out = np.empty(m)
    start = 0
    for n_e in group_sizes:
        stop = start + int(n_e)
        blk = B[start:stop, start:stop]
        # diag of the block inverse; cho_solve on I is the stable way to get it, and the
        # blocks are episode-sized (tens to hundreds), so the explicit inverse is fine.
        blk_inv = cho_solve(cho_factor(blk, lower=True), np.eye(stop - start))
        out[start:stop] = np.diag(blk_inv) - lam_m
        start = stop
    return out


def brute_block_loo(K: np.ndarray, lam_m: float, group_sizes: np.ndarray) -> np.ndarray:
    """The same thing by refitting from scratch with each block deleted. ``O(m^4)``."""
    m = K.shape[0]
    bounds = np.concatenate([[0], np.cumsum(group_sizes)]).astype(int)
    out = np.empty(m)
    for e in range(len(group_sizes)):
        mask = np.ones(m, dtype=bool)
        mask[bounds[e] : bounds[e + 1]] = False
        M = K[np.ix_(mask, mask)] + lam_m * np.eye(int(mask.sum()))
        M_chol = cho_factor(M, lower=True)
        for i in range(bounds[e], bounds[e + 1]):
            k_vec = K[mask, i]
            out[i] = K[i, i] - k_vec @ cho_solve(M_chol, k_vec)
    return out


def check_algebra(seed: int = 0) -> None:
    """Verify the closed form on a synthetic problem, including the singleton-block case."""
    rng = np.random.default_rng(seed)
    group_sizes = np.array([7, 3, 11, 5, 9])
    m = int(group_sizes.sum())
    X = rng.normal(size=(m, 4))
    K = np.exp(-0.5 * ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    lam_m = 1e-5 * m
    L = np.linalg.cholesky(K + lam_m * np.eye(m))

    closed = block_loo_scores(L, lam_m, group_sizes)
    brute = brute_block_loo(K, lam_m, group_sizes)
    assert np.allclose(closed, brute, atol=1e-9, rtol=1e-7), np.abs(closed - brute).max()

    # Singleton blocks must reproduce kern_cd_core's point-LOO identity exactly.
    singleton = block_loo_scores(L, lam_m, np.ones(m, dtype=int))
    L_inv = solve_triangular(L, np.eye(m), lower=True)
    point = 1.0 / np.einsum("ki,ki->i", L_inv, L_inv) - lam_m
    assert np.allclose(singleton, point, atol=1e-10, rtol=1e-9), np.abs(singleton - point).max()

    # And trajectory-LOO must be the more conservative of the two: deleting a superset of
    # the point-LOO holdout can only raise the posterior variance.
    assert (brute >= point - 1e-10).all()
    print(f"LOO algebra OK (m={m}, blocks={group_sizes.tolist()}): "
          f"closed form == brute force, singleton == point-LOO, block >= point elementwise")


# ------------------------------------------------------------------------------ the variant
class _TrajLOOMixin:
    """Swaps ``_loo_scores`` for its leave-one-episode-out counterpart.

    ``KernCDMethod.fit`` already records the fit set's episode structure
    (``_fit_episode_lengths`` / ``_fit_episode_kept``) before calling ``_loo_scores``, so
    the blocks are available without touching ``fit``. ``_splice_loo`` is inherited
    unchanged: the LOO array is still one score per fit-set row in the same order.
    """

    #: Refitting per block is O(n_episodes * m^3); affordable a good deal further up than
    #: the point version's O(m^4) brute force.
    _loo_validation_max_m = 1200

    def _loo_scores(self) -> np.ndarray:
        sizes = np.asarray(self._fit_episode_lengths, dtype=int)[np.asarray(self._fit_episode_kept, dtype=bool)]
        m = len(self.model.data)
        self._loo_group_sizes = sizes
        scores = block_loo_scores(self.model.L, self.model.lam_ * m, sizes)
        # Recorded for the shape report; costs one extra vector.
        L_inv = solve_triangular(self.model.L, np.eye(m), lower=True)
        self._point_loo_scores = 1.0 / np.einsum("ki,ki->i", L_inv, L_inv) - self.model.lam_ * m
        return scores

    def _validate_loo(self, Z: np.ndarray) -> None:
        m = len(Z)
        brute = brute_block_loo(self.model.kernel(Z), self.model.lam_ * m, self._loo_group_sizes)
        max_diff = np.abs(brute - self.loo_scores).max()
        assert np.allclose(brute, self.loo_scores, atol=1e-6, rtol=1e-5), (
            f"block-LOO closed form vs brute force: max abs diff {max_diff:.2e}"
        )
        print(f"  [{self.name}] block-LOO verified against brute-force refit "
              f"(m={m}, {len(self._loo_group_sizes)} episodes, max diff {max_diff:.2e})")


def make_variant(base_name: str) -> type[KernCDMethod]:
    """Register (once) the trajectory-LOO twin of a registered kern_cd method."""
    name = base_name + SUFFIX
    if name in REGISTRY:
        return REGISTRY[name]
    base = REGISTRY[base_name]
    cls = type(base.__name__ + "TrajLOO", (_TrajLOOMixin, base), {"name": name})
    cls.__doc__ = f"{base_name} with leave-one-trajectory-out calibration."
    return cls


# ----------------------------------------------------------------------------- the reporting
def grid(name: str, task: str, results: dict) -> pd.DataFrame:
    """The full (threshold style x quantile x window) metric grid for one method."""
    return pd.DataFrame(run_mod._rows(name, task, results))


def headline(df: pd.DataFrame) -> pd.Series:
    """The number the paper table reports: best (window, threshold) by quantile-averaged TWA.

    ``my_methods/paper_table.py`` averages over quantiles and then picks the best window and
    threshold style by TWA, so that is what a change has to move to matter. The per-config
    maximum over the raw grid is also reported, but it is the maximum of 510 noisy numbers
    and moves for reasons that have nothing to do with calibration.
    """
    avg = df.groupby(["Threshold", "Window"], as_index=False)[["TWA", "Accuracy", "Det. Time"]].mean()
    best = avg.loc[avg["TWA"].idxmax()]
    return pd.Series({
        "TWA": best["TWA"],
        "Accuracy": best["Accuracy"],
        "Det. Time": best["Det. Time"],
        "Threshold": best["Threshold"],
        "Window": best["Window"],
        "TWA_gridmax": df["TWA"].max(),
        "TWA_gridmean": df["TWA"].mean(),
    })


def report_shapes(task: str, impls: dict) -> None:
    """What trajectory grouping does to the shapes that carry meaning."""
    print(f"\n=== array shapes, {task} ===")
    for name, impl in impls.items():
        Z = np.asarray(impl.model.data)
        m, d = Z.shape
        sizes = getattr(impl, "_loo_group_sizes", np.ones(m, dtype=int))
        n_e = len(sizes)
        print(f"\n-- {name}")
        print(f"  fit array Z (the 'train array')       [{m}, {d}]   "
              f"= {n_e} kept episodes x ~{m / n_e:.0f} steps, d = d_obs"
              + ("" if not impl.uses_actions else " + 128 RFF"))
        print(f"  kernel matrix K / Cholesky L          [{m}, {m}]")
        print(f"  LOO score vector                      [{m}]        (unchanged: one per fit step)")
        if hasattr(impl, "_loo_group_sizes"):
            print(f"  per-LOO effective train array         [{m} - n_e, {d}] with n_e in "
                  f"[{sizes.min()}, {sizes.max()}]  ->  [{m - sizes.max()}..{m - sizes.min()}, {d}]")
            print(f"  blocks inverted, one per episode      {n_e} x [n_e, n_e], "
                  f"sizes {sizes.tolist()}")
            print(f"  fraction of the fit set held out      {sizes.min() / m:.1%} .. {sizes.max() / m:.1%} "
                  f"(point-LOO: {1 / m:.2%})")
        else:
            print(f"  per-LOO effective train array         [{m} - 1, {d}]")
            print(f"  blocks inverted                       none; the diagonal of B is read directly")


def report_loo_shift(task: str, impls: dict) -> None:
    """How much the calibration scores move, which is the only channel to the metrics."""
    print(f"\n=== calibration (LOO) score distribution, {task} ===")
    rows = []
    for name, impl in impls.items():
        s = np.asarray(impl.loo_scores)
        rows.append({"method": name, "min": s.min(), "q10": np.quantile(s, 0.10),
                     "median": np.median(s), "q90": np.quantile(s, 0.90),
                     "q95": np.quantile(s, 0.95), "max": s.max()})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    for name, impl in impls.items():
        if not hasattr(impl, "_point_loo_scores"):
            continue
        block, point = np.asarray(impl.loo_scores), np.asarray(impl._point_loo_scores)
        ratio = block / np.maximum(point, 1e-300)
        print(f"\n  [{name}] block / point LOO score ratio: "
              f"median {np.median(ratio):.3g}, q90 {np.quantile(ratio, 0.9):.3g}, max {ratio.max():.3g}")
        print(f"  [{name}] point-LOO scores are {np.mean(point < 1e-6) * 100:.1f}% below 1e-6 "
              f"(collapsed toward the in-sample value); block-LOO {np.mean(block < 1e-6) * 100:.1f}%")


def summarise(seed: int = 0) -> int:
    """Roll every cached (task, method) pair up into the paper table's footing.

    Two numbers per cell, because they answer different questions:

    * **headline** -- best (window, threshold) by quantile-averaged TWA, chosen per method.
      This is literally what ``paper_table.py`` prints, so it is the number that would
      change in data/results. It is also an argmax over 51 configurations, so it moves for
      reasons other than calibration quality.
    * **grid mean** -- TWA averaged over all 510 configurations. No selection, so it
      measures whether the method got better *everywhere* rather than at its own best spot.
    """
    files = sorted(pathlib.Path(CACHE_DIR).glob(f"*__seed{seed}.csv"))
    if not files:
        print(f"no cached grids in {CACHE_DIR}", file=sys.stderr)
        return 1

    grids = {}
    for f in files:
        task, name, _ = f.name.rsplit("__", 2)
        grids[(task, name)] = pd.read_csv(f)

    tasks = [t for t in run_mod.ALL_TASKS if any(k[0] == t for k in grids)]
    bases = [n for n in ("kern_cd_obs", "kern_cd_disp", "kern_cd_sig", "kern_cd_flat", "kern_cd_sum")
             if any(k[1] == n for k in grids)]

    for label, get in (("headline TWA (best window+threshold, quantile-averaged)",
                        lambda g: float(headline(g)["TWA"])),
                       ("grid-mean TWA (all 510 configs, no selection)",
                        lambda g: float(g["TWA"].mean()))):
        pt_rows, tj_rows = [], []
        for base_name in bases:
            pt_row, tj_row = {"Method": base_name}, {"Method": base_name}
            for task in tasks:
                pt, tj = grids.get((task, base_name)), grids.get((task, base_name + SUFFIX))
                if pt is None or tj is None:
                    continue
                pt_row[task], tj_row[task] = get(pt), get(tj)
            pt_rows.append(pt_row)
            tj_rows.append(tj_row)
        pt_df = pd.DataFrame(pt_rows).set_index("Method")
        tj_df = pd.DataFrame(tj_rows).set_index("Method")
        delta = tj_df - pt_df
        # Mean over the tasks a method actually has, so a partial sweep still prints.
        delta["MEAN_d"] = delta[tasks].mean(axis=1)
        delta["mean_pt"] = pt_df[tasks].mean(axis=1)
        delta["mean_tj"] = tj_df[tasks].mean(axis=1)

        print(f"\n=== {label} ===")
        print("  cells are trajectory-LOO minus point-LOO; mean_pt / mean_tj are the "
              "task-averaged levels\n")
        print(delta.to_string(float_format=lambda v: f"{v:+.4f}", na_rep="  --  "))
        arr = delta[tasks].to_numpy(dtype=float)
        n = int(np.isfinite(arr).sum())
        print(f"\n  trajectory-LOO better in {int(np.nansum(arr > 0))}/{n} (task, method) cells; "
              f"mean delta {np.nanmean(arr):+.4f}")
    return 0


# ---------------------------------------------------------------------------------- the run
def evaluate(task: str, names: list[str], seed: int) -> tuple[dict, dict]:
    """Run the FIPER evaluation for ``names`` on ``task``, returning (results, fitted impls).

    Both members of a pair go through one call so the processed dataset is built once and
    the two share it byte for byte.
    """
    results = run_mod._evaluate_task(names, task, seed, {})
    return results, dict(_LAST_IMPLS)


#: ``_evaluate_task`` returns metrics, not the method objects, and the objects hold the
#: fitted state this experiment wants to look at (Z, L, loo_scores). The adapter builds them
#: inside ``EvaluationManager``, so they are captured on the way past rather than plumbed
#: through my_methods/ -- which stays untouched.
_LAST_IMPLS: dict = {}


def _install_impl_capture() -> None:
    import my_methods.base as base_mod

    original = base_mod.make_eval_class

    def patched(method_cls, params):
        cls = original(method_cls, params)
        original_init = cls.__init__

        def __init__(self, *a, **kw):
            original_init(self, *a, **kw)
            _LAST_IMPLS[method_cls.name] = self.impl

        cls.__init__ = __init__
        return cls

    base_mod.make_eval_class = patched
    run_mod.make_eval_class = patched


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="loo_trajectories", description=__doc__)
    parser.add_argument("--task", default="push_chair", choices=run_mod.ALL_TASKS)
    parser.add_argument("--methods", nargs="+", default=["kern_cd_obs"],
                        help="base method name(s); each is run alongside its _trajloo twin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="ignore the cache")
    parser.add_argument("--check-only", action="store_true", help="verify the LOO algebra and exit")
    parser.add_argument("--summary", action="store_true", help="roll up every cached grid and exit")
    args = parser.parse_args(argv)

    if args.summary:
        return summarise(args.seed)

    check_algebra()
    if args.check_only:
        return 0

    discover()
    unknown = [m for m in args.methods if m not in REGISTRY]
    if unknown:
        print(f"unknown method(s) {unknown}; registered: {sorted(REGISTRY)}", file=sys.stderr)
        return 1
    for base_name in args.methods:
        make_variant(base_name)

    os.makedirs(CACHE_DIR, exist_ok=True)
    names = [n for b in args.methods for n in (b, b + SUFFIX)]
    cache = {n: os.path.join(CACHE_DIR, f"{args.task}__{n}__seed{args.seed}.csv") for n in names}

    grids: dict[str, pd.DataFrame] = {}
    impls: dict = {}
    if args.force or not all(os.path.exists(p) for p in cache.values()):
        _install_impl_capture()
        print(f"\nevaluating {names} on {args.task} (seed {args.seed}) ...")
        results, impls = evaluate(args.task, names, args.seed)
        for n in names:
            grids[n] = grid(n, args.task, results)
            grids[n].to_csv(cache[n], index=False)
        print(f"cached {len(names)} grids -> {os.path.relpath(CACHE_DIR, ROOT_DIR)}/")
    else:
        for n in names:
            grids[n] = pd.read_csv(cache[n])
        print(f"\nloaded cached grids for {names} on {args.task} "
              f"(--force to re-run; shape/LOO diagnostics need a re-run)")

    if impls:
        report_shapes(args.task, impls)
        report_loo_shift(args.task, impls)

    print(f"\n=== headline metrics, {args.task} (seed {args.seed}) ===")
    print("best (window, threshold style) by quantile-averaged TWA, as in paper_table.py\n")
    table = pd.DataFrame({n: headline(grids[n]) for n in names}).T
    table.index.name = "Method"
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n--- trajectory-LOO minus point-LOO ---")
    for base_name in args.methods:
        a, b = headline(grids[base_name]), headline(grids[base_name + SUFFIX])
        deltas = {k: float(b[k]) - float(a[k]) for k in ("TWA", "Accuracy", "Det. Time", "TWA_gridmax", "TWA_gridmean")}
        print(f"  {base_name:<16} " + "  ".join(f"{k} {v:+.4f}" for k, v in deltas.items()))

    print("\n--- paired by (threshold, quantile, window): TWA delta over the whole grid ---")
    keys = ["Threshold", "Quantile", "Window"]
    for base_name in args.methods:
        merged = grids[base_name].merge(grids[base_name + SUFFIX], on=keys, suffixes=("_pt", "_tj"))
        d = merged["TWA_tj"] - merged["TWA_pt"]
        print(f"  {base_name:<16} n={len(d)}  mean {d.mean():+.4f}  median {d.median():+.4f}  "
              f"win {np.mean(d > 0):.0%} / tie {np.mean(d == 0):.0%} / loss {np.mean(d < 0):.0%}  "
              f"[{d.min():+.4f}, {d.max():+.4f}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
