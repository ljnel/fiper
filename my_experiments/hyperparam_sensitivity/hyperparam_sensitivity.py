"""Hyperparameter sensitivity of kern_cd -- ``specs/experiments/hyperparam_sensitivity.md``.

Two questions, one experiment:

1. How stable are the headline metrics under the hyperparameters ``specs/methods.md``
   fixes by fiat?
2. Is there a setting that makes the *action channel* competitive? On push_t the action
   methods all score below the observation-only baseline, so "the action channel does not
   help" is currently a statement about one point in a five-dimensional box.

The five knobs, each swept in powers of two about the spec's value:

| knob | spec value | swept over | what it controls |
|---|---|---|---|
| ``sigma_o`` | median heuristic | 1/8x .. 8x | width of the observation kernel |
| ``sigma_phi`` | median heuristic | 1/8x .. 8x | width *inside* the RFF map phi, on single actions |
| ``sigma_A`` | median heuristic | 1/8x .. 8x | width of the kernel on the mean embeddings mu_A |
| ``lam`` | 1e-5 | 2^-12x .. 2^12x | ridge; the matrix factored is ``K + lam*m*I`` |
| ``n_components`` | 128 | 8 .. 512 | RFF dimension, i.e. how well phi approximates its kernel |

The three bandwidths are swept as *multiples of the median heuristic* rather than as
absolute lengths. That keeps the sweep centred on the spec no matter what the data's scale
is, and it makes 1.0x the spec's own configuration by construction, so every table below
reads as a displacement from the method as specified.

Scope, and what it costs
------------------------
``kern_cd_sum`` on push_t, one seed, one knob at a time (31 arms), plus:

* ``kern_cd_obs`` swept over the two knobs it has, ``sigma_o`` and ``lam`` (13 arms). This
  is the control. ``sigma_o`` is shared between the two methods, so without it "sum at 4x
  beats obs at 1x" cannot be told apart from "everything at 4x beats everything at 1x" --
  it would credit the action channel for a gain that belongs to the observation kernel.
* The spec's own configuration re-run at seeds 1 and 2 (4 arms). The RFF draw is
  seed-dependent, and a sensitivity study with no noise scale cannot say whether a swept
  difference is a difference at all.
* A 7x7 grid over the two knobs the sweep finds most influential (36 further arms).
* Three arms continuing ``sigma_A`` to 256x, for the degeneracy check described below.

87 end-to-end evaluations, 226 s on an RTX 5090. Every arm is a full FIPER evaluation --
features, estimator, LOO calibration, the 17x10x3 threshold sweep, metrics -- so the
numbers are directly comparable with ``data/results/complete_results.csv``. Arms are
evaluated in groups of twelve through one ``run._evaluate_task`` call, which builds the
processed dataset once per group.

What it found
-------------
Seed 0 on push_t (m = 350 fit steps; median heuristic sigma_o 0.66, sigma_phi 179,
sigma_A 0.83). Headline TWA at the paper table's footing.

**1. The headline is not stable, and the instability is of the same size as the gaps
between the methods.** The noise floor first: re-running the spec's own configuration at
three seeds moves TWA by 0.0021 (``kern_cd_obs`` draws no random features at all and is
bit-reproducible, so its range is exactly 0). Against that, each knob on its own moves TWA
over its swept range by

    sigma_A 0.036,  lam 0.030,  sigma_o 0.023,  sigma_phi 0.011,  n_components 0.008,

and on the obs baseline, sigma_o 0.042 and lam 0.025 -- 5x to 20x the seed noise. For
scale, the five kern_cd methods span 0.053 TWA on this task in
``complete_results.csv``. So a ranking of the kernels that rests on a gap of a few
hundredths is reporting the hyperparameters as much as the kernel: the *baseline* alone
gains +0.019 (0.5877 -> 0.6068) by moving sigma_o to 4x the median heuristic.

Two knobs are exceptions worth having. ``n_components`` does essentially nothing (0.008
over 8..512, with 8 components scoring *above* the spec's 128) -- the mean embedding needs
almost no capacity, and the RFF dimension is not where any accuracy is hiding. And
``sigma_phi``'s best value over its whole 64x range is 1.0x, i.e. the median heuristic is
already at the optimum for the one bandwidth this experiment can move without changing
anything else.

**2. No, the action channel does not become competitive -- and the direction that helps it
is the direction that deletes it.** ``kern_cd_sum`` scores 0.5483 against the baseline's
0.5877 at the spec's settings. Of the 31 one-at-a-time arms, 0 beat the baseline; the best
(0.5754) is sigma_A at 8x. Of the 49 cells of the (sigma_A, lam) grid, exactly 1 beats it:
0.5897 at sigma_A = 8x, lam = 2.4e-9. That is +0.0020 over the baseline -- one seed's worth
of noise -- at a corner of the grid, while the baseline's *own* sweep reaches 0.6068.

The mechanism kills the remaining interpretation. The kernel is a product,
``k_o * exp(-||mu_A - mu_A'||^2 / 2 sigma_A^2)``, so as sigma_A grows the action factor
tends to 1 uniformly and the kernel tends to the observation-only baseline's. Reporting the
mean off-diagonal action factor beside TWA shows exactly that happening:

    sigma_A / median   0.125    1.0     8.0     16      64      256      (obs baseline)
    action factor      0.019   0.624   0.992   0.998   1.000   1.000
    TWA                0.5605  0.5483  0.5754  0.5792  0.5849  0.5873     0.5877

TWA rises monotonically towards the baseline's value as the action channel is switched off,
and at the best swept setting the channel is already 99% inert. "Widening sigma_A helps" is
therefore not a tuning result; it is the method reverting to ``kern_cd_obs``.

**What this does not settle.** One task, one seed for the sweeps, and one of the four
action kernels. push_t is the task where the action channel does worst, and 350 fit steps
is the smallest fit set in the benchmark. The degeneracy argument is structural and applies
to all four action methods unchanged (they differ only in what goes into ``mu_A``), but the
sensitivity numbers are single-task. Interactions beyond the one 2-D grid are unexplored:
the sweep is one-at-a-time, so a knob that only matters jointly with another would be
invisible here.

Running it
----------
    pixi run python my_experiments/hyperparam_sensitivity/hyperparam_sensitivity.py            # the whole thing
    pixi run python my_experiments/hyperparam_sensitivity/hyperparam_sensitivity.py --check-only
    pixi run python my_experiments/hyperparam_sensitivity/hyperparam_sensitivity.py --task pretzel
    pixi run python my_experiments/hyperparam_sensitivity/hyperparam_sensitivity.py --heatmap sigma_A lam
    pixi run python my_experiments/hyperparam_sensitivity/hyperparam_sensitivity.py --report    # cached arms only

Per-arm metric grids are cached as CSVs under ``my_experiments/hyperparam_sensitivity/cache/``
and reused; ``--force`` re-runs. Nothing in ``my_methods/`` or under ``data/results/`` is
modified, but FIPER's own eval class writes a small pickle per arm into
``data/<task>/results/<arm>/``; every arm this file creates is named ``hps_*``, so the
litter is removable with ``rm -rf data/*/results/hps_*``.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import textwrap
import time
from dataclasses import dataclass, fields, replace

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

#: The method under test, and the baseline it has to beat.
SUM, OBS = "kern_cd_sum", "kern_cd_obs"

#: Arms evaluated per ``_evaluate_task`` call. The processed dataset is rebuilt once per
#: call (~1 s on push_t), and a group is the unit of crash recovery, so this trades a
#: little rebuild time for not losing an eight-minute run to one bad arm.
GROUP = 12


# ------------------------------------------------------------------------------- the axes
@dataclass(frozen=True)
class Axis:
    """One hyperparameter, its grid, and how it reaches the method."""

    key: str  # field of Arm
    values: tuple  # swept values, including the spec's own
    label: str  # for plots
    unit: str  # for tables: what the value means
    obs: bool = False  # does the observation-only baseline have this knob?
    fmt: str = "{:g}"  # tick labels

    def tick(self, value) -> str:
        return self.fmt.format(value)

    @property
    def default(self):
        return getattr(Arm(base=SUM), self.key)


#: Powers of two, +-3 octaves: a 64x range on each bandwidth. Wide enough that the kernel
#: is nearly constant at one end (every point looks like every other) and nearly diagonal
#: at the other (no point supports any other), so the sweep brackets both degenerate
#: regimes rather than probing a neighbourhood of the median heuristic.
_OCTAVES = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
_MULTS = tuple(2.0**k for k in _OCTAVES)

AXES: dict[str, Axis] = {
    # Still powers of two, but in strides of four octaves: lam is a ridge, not a length,
    # and its interesting range is orders of magnitude wide (at the small end the fit is a
    # near-interpolant, at the large end K + lam*m*I is all ridge). A +-3 octave window
    # around 1e-5 would not leave the near-interpolant regime at all.
    "lam": Axis("lam", tuple(1e-5 * 2.0**k for k in (-12, -8, -4, 0, 4, 8, 12)),
                r"$\lambda$", "absolute", obs=True, fmt="{:.0e}"),
    "sigma_o": Axis("mult_o", _MULTS, r"$\sigma_o$", "x median heuristic", obs=True),
    "sigma_phi": Axis("mult_phi", _MULTS, r"$\sigma_\varphi$", "x median heuristic"),
    "sigma_A": Axis("mult_A", _MULTS, r"$\sigma_A$", "x median heuristic"),
    "n_components": Axis("n_components", (8, 16, 32, 64, 128, 256, 512),
                         "RFF components", "absolute"),
}


@dataclass(frozen=True)
class Arm:
    """One end-to-end evaluation: a base method plus a point in the hyperparameter box."""

    base: str
    mult_o: float = 1.0
    mult_phi: float = 1.0
    mult_A: float = 1.0
    lam: float = 1.0e-5
    n_components: int = 128

    @property
    def name(self) -> str:
        """Registry key and cache key: the base, plus only what differs from the spec.

        The spec's own configuration is therefore ``hps_sum`` / ``hps_obs``, and every
        other arm's name says exactly how it deviates.
        """
        base = Arm(base=self.base)
        tag = "".join(
            f"__{f.name}{getattr(self, f.name):g}"
            for f in fields(self)
            if f.name != "base" and getattr(self, f.name) != getattr(base, f.name)
        )
        return "hps_" + self.base.removeprefix("kern_cd_") + tag


def spec_arm(base: str) -> Arm:
    """The method exactly as ``specs/methods.md`` specifies it."""
    return Arm(base=base)


# ------------------------------------------------------------------------------ the variant
#: arm name -> the bandwidths that arm actually used. Populated during fit; kept as scalars
#: rather than by holding the fitted methods, which would pin an m x m Cholesky factor per
#: arm (72 MB each at push_t's m).
_SIGMAS: dict[str, dict] = {}


class _HyperMixin:
    """Scales each of the three bandwidths by a fixed multiple of its median heuristic.

    The two hooks are the two moments at which a bandwidth has just been fixed by the
    median heuristic and has not yet been used:

    * ``_fit_phi`` sets ``sigma_phi`` and immediately draws the RFF weights from it.
    * ``_pack`` runs once ``sigma_o`` and ``sigma_A`` are both known, and its output is
      what ``KernCD`` is fitted on. ``_features`` calls it again on every scoring pass, so
      the scaling is guarded to happen once -- on the fit set, where the medians live.

    ``lam`` and ``n_components`` need no hook: they are already ``params`` entries, and
    ``make_arm`` writes them into the variant's class-level ``params``.
    """

    hp_mult_o = 1.0
    hp_mult_phi = 1.0
    hp_mult_A = 1.0

    def _fit_phi(self, parts) -> None:
        """Draw phi at the median heuristic, then rescale it to ``hp_mult_phi`` times that.

        Rescaling the drawn weights is not an approximation to re-drawing at the new
        bandwidth -- it *is* the same draw. ``RBFSampler`` sets
        ``W = sqrt(2*gamma) * N(0,1) = N(0,1)/sigma_phi`` and draws the offsets ``b`` from a
        distribution that does not involve gamma at all, so the map at ``c*sigma_phi`` is
        ``(W/c, b)`` with the identical underlying normal and uniform draws. Multipliers are
        powers of two, so in float32 the division is exact as well.
        :func:`check_phi_scaling` asserts this against sklearn rather than assuming it.
        """
        super()._fit_phi(parts)
        c = float(self.hp_mult_phi)
        if c != 1.0:
            self._sigma_phi_median = float(self._sigma_phi)
            self._sigma_phi = float(self._sigma_phi) * c
            self._phi_W = self._phi_W / c

    def _pack(self, obs, mu):
        if not getattr(self, "_hp_done", False):
            self._hp_done = True
            self._sigma_o_median = float(self._sigma_o)
            self._sigma_o = float(self._sigma_o) * float(self.hp_mult_o)
            if mu is not None:
                self._sigma_A_median = float(self._sigma_A)
                self._sigma_A = float(self._sigma_A) * float(self.hp_mult_A)
            _SIGMAS[self.name] = {
                "arm": self.name,
                "m": int(len(obs)),
                "sigma_o_median": self._sigma_o_median,
                "sigma_o": self._sigma_o,
                "sigma_phi_median": float(getattr(self, "_sigma_phi_median", getattr(self, "_sigma_phi", np.nan))),
                "sigma_phi": float(getattr(self, "_sigma_phi", np.nan)),
                "sigma_A_median": float(getattr(self, "_sigma_A_median", getattr(self, "_sigma_A", np.nan))),
                "sigma_A": float(getattr(self, "_sigma_A", np.nan)),
                "k_obs": _mean_factor(obs, self._sigma_o),
                "k_A": _mean_factor(mu, getattr(self, "_sigma_A", np.nan)) if mu is not None else np.nan,
            }
        return super()._pack(obs, mu)


def _mean_factor(X: np.ndarray, sigma: float, cap: int = 800) -> float:
    """Mean off-diagonal value of ``exp(-||x-x'||^2 / 2 sigma^2)`` on the fit set.

    How much work a kernel factor is doing, in one number. The product kernel multiplies
    this factor into every entry, so a factor whose mean is ~1 with no spread is a factor
    that has been switched off: ``k((o,A),(o',A')) = k_o * 1`` is the observation-only
    baseline. That is the difference between "widening sigma_A tunes the action channel"
    and "widening sigma_A deletes it", which TWA alone cannot tell apart.
    """
    from sklearn.metrics.pairwise import euclidean_distances

    if X is None or not np.isfinite(sigma):
        return float("nan")
    A = np.ascontiguousarray(X[:cap], dtype=np.float64)
    K = np.exp(-euclidean_distances(A, A, squared=True) / (2.0 * sigma**2))
    n = len(A)
    return float((K.sum() - np.trace(K)) / (n * (n - 1)))


def make_arm(arm: Arm):
    """Register (once) the variant class implementing one arm."""
    if arm.name in REGISTRY:
        return REGISTRY[arm.name]
    base = REGISTRY[arm.base]
    cls = type(
        "Arm" + arm.name.replace("hps_", "").replace("__", "_").replace(".", "p").replace("-", "m"),
        (_HyperMixin, base),
        {
            "name": arm.name,
            # A fresh dict, not base.params mutated: params is a class attribute the whole
            # family shares.
            "params": dict(base.params, lam=float(arm.lam), n_components=int(arm.n_components)),
            "hp_mult_o": float(arm.mult_o),
            "hp_mult_phi": float(arm.mult_phi),
            "hp_mult_A": float(arm.mult_A),
        },
    )
    cls.__doc__ = f"{arm.base} at {arm}."
    return cls


def check_phi_scaling(seed: int = 0) -> None:
    """Verify the one piece of arithmetic this file adds to the methods.

    Everything else here is configuration; ``_fit_phi``'s rescaling is a substantive claim
    about ``RBFSampler``, so it is checked against ``RBFSampler`` itself: the map obtained
    by dividing the weights of a draw at ``sigma`` must be bit-for-bit the map that a draw
    at ``c*sigma`` produces, and the kernel it approximates must be the RBF at ``c*sigma``.
    """
    from sklearn.kernel_approximation import RBFSampler

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, 3))
    sigma, c, d = 0.37, 4.0, 4096

    def draw(s):
        rbf = RBFSampler(gamma=1.0 / (2.0 * s**2), n_components=d, random_state=seed)
        rbf.fit(np.zeros((1, X.shape[1])))
        return rbf.random_weights_, rbf.random_offset_

    W, b = draw(sigma)
    W_c, b_c = draw(sigma * c)
    assert np.array_equal(b, b_c), "the RFF offsets depend on gamma"
    assert np.allclose(W / c, W_c, rtol=0, atol=0), "W(c*sigma) != W(sigma)/c"

    phi = np.sqrt(2.0 / d) * np.cos(X @ W_c + b_c)
    K_hat, K = phi @ phi.T, np.exp(-((X[:, None] - X[None]) ** 2).sum(-1) / (2 * (sigma * c) ** 2))
    err = np.abs(K_hat - K).max()
    # An RFF estimate of one kernel entry has standard error ~1/sqrt(d); this is a maximum
    # over 1.6e5 entries, so the tolerance is set at five of those rather than at one.
    tol = 5.0 / np.sqrt(d)
    assert err < tol, f"the rescaled map does not approximate the RBF at c*sigma (max err {err:.3f})"
    print(f"phi rescaling OK: W(c*sigma) == W(sigma)/c exactly, offsets unchanged, and the "
          f"resulting map approximates exp(-||x-y||^2 / 2({c}*{sigma})^2) to {err:.3f} "
          f"(tol {tol:.3f}) at d={d}")


# ------------------------------------------------------------------------------ the metrics
def grid(name: str, task: str, results: dict) -> pd.DataFrame:
    """The full (threshold style x quantile x window) metric grid for one arm."""
    return pd.DataFrame(run_mod._rows(name, task, results))


def headline(df: pd.DataFrame) -> pd.Series:
    """The three headline metrics at the footing ``paper_table.py`` reports.

    Quantile-averaged, then the best (window, threshold style) by TWA. ``TWA_gridmean`` --
    the flat mean over all 510 configurations -- rides along because the headline is an
    argmax over 51 noisy numbers and can move on its own; a knob that genuinely helps moves
    both.
    """
    avg = df.groupby(["Threshold", "Window"], as_index=False)[["TWA", "Accuracy", "Det. Time"]].mean()
    best = avg.loc[avg["TWA"].idxmax()]
    return pd.Series({
        "TWA": float(best["TWA"]),
        "Accuracy": float(best["Accuracy"]),
        "Det. Time": float(best["Det. Time"]),
        "TWA_gridmean": float(df["TWA"].mean()),
        "Window": best["Window"],
        "Threshold": best["Threshold"],
    })


# ---------------------------------------------------------------------------------- the run
def _cache_path(task: str, arm: Arm, seed: int) -> str:
    return os.path.join(CACHE_DIR, f"{task}__{arm.name}__seed{seed}.csv")


def _sigma_path(task: str, seed: int) -> str:
    return os.path.join(CACHE_DIR, f"{task}__sigmas__seed{seed}.csv")


def evaluate(task: str, seed: int, arms: list[Arm], force: bool = False,
             cache_only: bool = False) -> dict[str, pd.DataFrame]:
    """Evaluate every arm not already cached, and return every arm's metric grid.

    A group that raises is retried arm by arm, so one configuration that the estimator
    cannot handle (a ridge small enough to break the Cholesky, say) costs its own arm and
    not the eleven it was batched with. A failed arm is simply absent from the result, and
    the reports render it as ``--``.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    todo: list[Arm] = []
    for arm in arms:
        path = _cache_path(task, arm, seed)
        if os.path.exists(path) and not force:
            out[arm.name] = pd.read_csv(path)
        elif arm.name not in {a.name for a in todo}:
            todo.append(arm)

    if cache_only or not todo:
        return out

    print(f"\nevaluating {len(todo)} arm(s) on {task}, seed {seed} "
          f"({len(out)} already cached) ...")
    started = time.time()
    for i in range(0, len(todo), GROUP):
        group = todo[i : i + GROUP]
        done = _evaluate_group(task, seed, group, out)
        if not done:  # the group failed as a whole; find out which arm did
            for arm in group:
                _evaluate_group(task, seed, [arm], out)
        print(f"  {min(i + GROUP, len(todo))}/{len(todo)} arms, {time.time() - started:.0f}s elapsed")

    if _SIGMAS:
        path = _sigma_path(task, seed)
        old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["arm"])
        new = pd.DataFrame(list(_SIGMAS.values()))
        merged = pd.concat([old[~old["arm"].isin(new["arm"])], new], ignore_index=True)
        merged.to_csv(path, index=False)
    return out


def _evaluate_group(task: str, seed: int, group: list[Arm], out: dict) -> bool:
    for arm in group:
        make_arm(arm)
    names = [a.name for a in group]
    try:
        results = run_mod._evaluate_task(names, task, seed, {})
    except Exception as exc:  # noqa: BLE001 -- an arm that cannot be evaluated is a result
        print(f"  !! {names if len(names) > 1 else names[0]} failed: "
              f"{type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
        return False
    for arm in group:
        df = grid(arm.name, task, results)
        df.to_csv(_cache_path(task, arm, seed), index=False)
        out[arm.name] = df
    return True


def oat_arms(base: str) -> list[Arm]:
    """One arm per (axis, value): every knob swept with the others at the spec's value."""
    arms = [spec_arm(base)]
    for axis in AXES.values():
        if base == OBS and not axis.obs:
            continue
        for value in axis.values:
            arm = replace(spec_arm(base), **{axis.key: value})
            if arm.name not in {a.name for a in arms}:
                arms.append(arm)
    return arms


# ----------------------------------------------------------------------------- the reporting
def collect(grids: dict[str, pd.DataFrame], arms: list[Arm]) -> pd.DataFrame:
    """Headline metrics for a list of arms, indexed by arm name."""
    rows = []
    for arm in arms:
        df = grids.get(arm.name)
        row = {"arm": arm.name, "base": arm.base}
        row.update({f.name: getattr(arm, f.name) for f in fields(arm) if f.name != "base"})
        if df is None:
            row.update(dict.fromkeys(["TWA", "Accuracy", "Det. Time", "TWA_gridmean"], np.nan))
        else:
            row.update(headline(df).to_dict())
        rows.append(row)
    return pd.DataFrame(rows).set_index("arm")


def report_oat(table: pd.DataFrame, base: str, noise: float | None) -> pd.DataFrame:
    """Per-axis sweep tables plus the range each axis spans. Returns the range summary."""
    ref = table.loc[spec_arm(base).name]
    print(f"\n=== {base}: one knob at a time (the other four at the spec's value) ===")
    print(f"the spec's own configuration: TWA {ref['TWA']:.4f}   Accuracy {ref['Accuracy']:.4f}   "
          f"Det. Time {ref['Det. Time']:.4f}   grid-mean TWA {ref['TWA_gridmean']:.4f}\n")

    summary = []
    for axis_name, axis in AXES.items():
        if base == OBS and not axis.obs:
            continue
        rows = []
        for value in axis.values:
            arm = replace(spec_arm(base), **{axis.key: value})
            if arm.name not in table.index:
                continue
            r = table.loc[arm.name]
            rows.append({
                axis_name: value,
                "x spec": value / axis.default,
                "TWA": r["TWA"],
                "dTWA": r["TWA"] - ref["TWA"],
                "Accuracy": r["Accuracy"],
                "Det. Time": r["Det. Time"],
                "TWA_gridmean": r["TWA_gridmean"],
            })
        df = pd.DataFrame(rows)
        print(f"-- {axis_name}  ({axis.unit})")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}", na_rep="  --  "))
        print()
        twa = df["TWA"].to_numpy(dtype=float)
        best = df.loc[df["TWA"].idxmax()] if np.isfinite(twa).any() else None
        summary.append({
            "axis": axis_name,
            "n": int(np.isfinite(twa).sum()),
            "TWA range": np.nanmax(twa) - np.nanmin(twa),
            "TWA at spec": ref["TWA"],
            "best TWA": np.nanmax(twa),
            "best at": f"{best[axis_name]:g}" if best is not None else "--",
            "gain": np.nanmax(twa) - ref["TWA"],
            "gridmean range": np.nanmax(df["TWA_gridmean"]) - np.nanmin(df["TWA_gridmean"]),
        })

    out = pd.DataFrame(summary).sort_values("TWA range", ascending=False)
    print(f"=== {base}: how much each knob moves TWA over its whole swept range ===")
    if noise is not None:
        print(f"    (seed-to-seed range of the spec configuration, same task: {noise:.4f} TWA)")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return out


def report_noise(task: str, seeds: list[int], force: bool, cache_only: bool = False) -> dict[str, float]:
    """The spec's own configuration at several seeds: the scale a sweep effect is judged on.

    The only stochastic ingredient is the RFF draw, which is seeded from the run seed, so
    this is the *method's* variance at a fixed configuration, on fixed data. It is what
    separates "this knob matters" from "this knob is noise".
    """
    print("\n=== noise floor: the spec's configuration re-run at several seeds ===")
    rows = []
    for seed in seeds:
        grids = evaluate(task, seed, [spec_arm(SUM), spec_arm(OBS)], force=force, cache_only=cache_only)
        for base in (SUM, OBS):
            df = grids.get(spec_arm(base).name)
            if df is None:
                continue
            h = headline(df)
            rows.append({"base": base, "seed": seed, "TWA": h["TWA"], "Accuracy": h["Accuracy"],
                         "Det. Time": h["Det. Time"], "TWA_gridmean": h["TWA_gridmean"]})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    spread = {}
    for base, g in df.groupby("base"):
        spread[base] = float(g["TWA"].max() - g["TWA"].min())
        print(f"  {base:<14} TWA range over {len(g)} seeds: {spread[base]:.4f}   "
              f"(std {g['TWA'].std(ddof=1):.4f})")
    return spread


def heatmap_arms(base: str, ax1: str, ax2: str) -> tuple[list[Arm], list, list]:
    """The 2-D grid, over each axis's full swept range.

    A sub-box was the first attempt and it was the wrong call: the best cell landed on its
    edge, which says nothing about whether the optimum is inside the box or outside it. The
    full range costs 49 arms instead of 25 -- about a minute -- and its own edges are the
    edges of the whole study, where the degeneracy check below takes over.
    """
    v1, v2 = [list(AXES[a].values) for a in (ax1, ax2)]
    arms = [replace(spec_arm(base), **{AXES[ax1].key: a, AXES[ax2].key: b}) for a in v1 for b in v2]
    return arms, v1, v2


#: sigma_A far past the swept range: 16x to 256x the median heuristic.
_DEGENERATE_MULTS = (16.0, 64.0, 256.0)


def report_degeneracy(task: str, seed: int, table: pd.DataFrame, obs_twa: float,
                      force: bool, cache_only: bool) -> dict:
    """Where a wide sigma_A leads, and what it means that it leads there.

    The product kernel is ``k_o * exp(-||mu-mu'||^2 / 2 sigma_A^2)``, so as sigma_A grows
    the action factor tends to 1 uniformly and the kernel tends to the observation-only
    baseline's -- the packed ``mu/sigma_A`` block shrinks to zero and stops contributing to
    any distance. "Widening sigma_A improves TWA" is therefore not a tuning result until it
    is shown to stop somewhere short of that limit. This continues the sweep to 256x and
    prints, beside each TWA, the mean off-diagonal value of the action factor itself: the
    number that says how much of the action channel is left.
    """
    arms = [replace(spec_arm(SUM), mult_A=m) for m in _DEGENERATE_MULTS]
    grids = evaluate(task, seed, arms, force=force, cache_only=cache_only)
    extra = collect(grids, arms)

    print("\n=== what a wide sigma_A actually does ===")
    print("  as sigma_A -> inf the action factor -> 1 and the kernel -> the obs-only baseline's;")
    print("  k_A is the mean off-diagonal action factor on the fit set (1.000 = channel inert)\n")
    sig_path = _sigma_path(task, seed)
    sig = pd.read_csv(sig_path).set_index("arm") if os.path.exists(sig_path) else pd.DataFrame()
    rows = []
    for mult in (0.125, 1.0, 8.0, *_DEGENERATE_MULTS):
        arm = replace(spec_arm(SUM), mult_A=mult)
        src = table if arm.name in table.index else extra
        if arm.name not in src.index:
            continue
        r = src.loc[arm.name]
        rows.append({
            "sigma_A / median": mult,
            "k_A": float(sig.loc[arm.name, "k_A"]) if arm.name in sig.index else np.nan,
            "TWA": r["TWA"],
            "Accuracy": r["Accuracy"],
            "Det. Time": r["Det. Time"],
            "TWA_gridmean": r["TWA_gridmean"],
            "TWA - obs": r["TWA"] - obs_twa,
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}", na_rep="  --  "))
    print(f"  {OBS} at the spec's configuration: TWA {obs_twa:.4f}   "
          f"(the limit the row above it is converging to)")
    top = out[out["sigma_A / median"] == max(AXES["sigma_A"].values)]
    return {"k_A_top": float(top["k_A"].iloc[0]) if len(top) else np.nan,
            "mult_top": max(AXES["sigma_A"].values),
            "limit_TWA": float(out["TWA"].iloc[-1])}


def report_heatmap(table: pd.DataFrame, base: str, ax1: str, ax2: str,
                   v1: list, v2: list, obs_twa: float | None) -> pd.DataFrame:
    """The 2-D grid as a pivot table of TWA, printed before it is plotted."""
    rows = []
    for a in v1:
        for b in v2:
            arm = replace(spec_arm(base), **{AXES[ax1].key: a, AXES[ax2].key: b})
            r = table.loc[arm.name] if arm.name in table.index else None
            rows.append({ax1: a, ax2: b,
                         "TWA": np.nan if r is None else r["TWA"],
                         "Accuracy": np.nan if r is None else r["Accuracy"],
                         "Det. Time": np.nan if r is None else r["Det. Time"],
                         "TWA_gridmean": np.nan if r is None else r["TWA_gridmean"]})
    df = pd.DataFrame(rows)
    print(f"\n=== {base}: TWA over ({ax1}, {ax2}), the two most influential knobs ===")
    print(df.pivot(index=ax1, columns=ax2, values="TWA").to_string(float_format=lambda v: f"{v:.4f}",
                                                                   na_rep="  --  "))
    best = df.loc[df["TWA"].idxmax()]
    print(f"\n  best cell: {ax1}={AXES[ax1].tick(best[ax1])}, {ax2}={AXES[ax2].tick(best[ax2])}   "
          f"TWA {best['TWA']:.4f}   Accuracy {best['Accuracy']:.4f}   Det. Time {best['Det. Time']:.4f}")
    if obs_twa is not None:
        n_above = int((df["TWA"] > obs_twa).sum())
        print(f"  cells beating the obs-only baseline ({obs_twa:.4f}): {n_above}/{len(df)}")
    return df


# -------------------------------------------------------------------------------- the plot
# Repo house style (my_experiments/profiling/profiling.py, logdet_kernel.py): near-white surface,
# near-black ink, muted labels. A grid of one magnitude gets a sequential ramp in a single
# hue, light -> dark; the categorical palette has no job here.
_INK, _MUTED, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
_ACCENT = "#eb6834"


def plot_heatmap(df: pd.DataFrame, base: str, task: str, ax1: str, ax2: str,
                 spec_twa: float, obs_twa: float | None, noise: float | None, path: str,
                 note: str = "") -> None:
    """TWA over the two knobs, with the spec's configuration and the baseline marked.

    Three things have to be legible at once, and none of them is the colour ramp on its
    own: the value in each cell (printed, since 49 cells is few enough that a reader wants
    the number), where the spec sits (a ring), and whether any cell clears the
    observation-only baseline (cells that do are outlined; the subtitle says so either way,
    because "no outlines" is otherwise indistinguishable from "no such annotation"). The
    ramp is a single hue -- this is one magnitude, not a polarity -- and the colour range is
    stretched to the data, with the range named on the colour bar so a small spread cannot
    be mistaken for a large one.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    piv = df.pivot(index=ax2, columns=ax1, values="TWA")
    xv, yv = piv.columns.to_numpy(dtype=float), piv.index.to_numpy(dtype=float)
    Z = piv.to_numpy(dtype=float)
    # Every axis here is geometric (powers of two, or powers of two times 1e-5), so the
    # grid is drawn in log2 coordinates. A log-scaled linear axis would give cells of
    # unequal width, which reads as unequal importance.
    X, Y = np.log2(xv), np.log2(yv)

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    mesh = ax.pcolormesh(X, Y, Z, cmap="Blues", shading="nearest",
                         vmin=np.nanmin(Z), vmax=np.nanmax(Z),
                         edgecolors=_SURFACE, linewidth=2)
    bar = fig.colorbar(mesh, ax=ax, pad=0.02)
    bar.set_label(f"headline TWA (range {np.nanmin(Z):.3f}-{np.nanmax(Z):.3f}; the metric runs 0-1)",
                  color=_MUTED, fontsize=9)
    bar.ax.tick_params(colors=_MUTED, labelsize=8)
    bar.outline.set_visible(False)

    span = np.nanmax(Z) - np.nanmin(Z) or 1.0
    dx = float(np.diff(X).mean()) if len(X) > 1 else 1.0
    dy = float(np.diff(Y).mean()) if len(Y) > 1 else 1.0
    for j, y in enumerate(Y):
        for i, x in enumerate(X):
            if not np.isfinite(Z[j, i]):
                continue
            # Ink on the pale end, surface colour on the dark end: one text colour cannot
            # clear both ends of a sequential ramp. Which end a cell is on is decided by
            # its rendered luminance, not by its position in the value range -- a ramp is
            # not linear in lightness, and eyeballing the crossover puts white text on
            # mid-blues at about 2:1.
            rgb = mesh.cmap((Z[j, i] - np.nanmin(Z)) / span)[:3]
            lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
            luminance = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
            beats = obs_twa is not None and Z[j, i] > obs_twa
            ax.text(x, y, f"{Z[j, i]:.3f}", ha="center", va="center", fontsize=9,
                    color=_SURFACE if luminance < 0.179 else _INK, zorder=4,
                    fontweight="bold" if beats else "normal")
            if beats:
                ax.add_patch(plt.Rectangle((x - dx / 2, y - dy / 2), dx, dy, fill=False,
                                           edgecolor=_ACCENT, linewidth=2.5, zorder=3))

    # The spec's cell gets a ring and nothing else. A text box beside it is what this
    # started as, and on a 7x7 grid there is no offset at which it does not cover a
    # neighbour's value; the subtitle names the ring instead, where it costs no space.
    d1, d2 = np.log2(AXES[ax1].default), np.log2(AXES[ax2].default)
    ax.add_patch(plt.Rectangle((d1 - dx / 2, d2 - dy / 2), dx, dy, fill=False,
                               edgecolor=_INK, linewidth=2.0, zorder=3))

    ax.set_xticks(X)
    ax.set_yticks(Y)
    ax.set_xticklabels([AXES[ax1].tick(v) for v in xv])
    ax.set_yticklabels([AXES[ax2].tick(v) for v in yv])
    ax.set_xlabel(f"{AXES[ax1].label}   ({AXES[ax1].unit})", color=_MUTED, fontsize=10)
    ax.set_ylabel(f"{AXES[ax2].label}   ({AXES[ax2].unit})", color=_MUTED, fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=9)

    lines = [f"{base} on {task};  black outline = the configuration in specs/methods.md "
             f"(TWA {spec_twa:.3f})" + (f";  seed noise {noise:.3f} TWA" if noise else "")]
    if obs_twa is not None:
        n_beat = int(np.nansum(Z > obs_twa))
        lines.append(f"{f'orange outline: {n_beat} cell' + ('s' if n_beat > 1 else '') + ' beat'
                       if n_beat else 'no cell beats'} the obs-only baseline (TWA {obs_twa:.3f})")
    lines += textwrap.wrap(note, width=108) if note else []

    fig.suptitle(f"Headline TWA over {ax1} and {ax2}", color=_INK, fontsize=13, x=0.02, ha="left")
    # One text block hung from a fixed top edge, just below the title: each extra caption
    # line then pushes the axes down by its own height rather than by a guessed constant,
    # so neither the title nor the plot is overlapped at any number of lines.
    fig.text(0.02, 0.935, "\n".join(lines), color=_MUTED, fontsize=9.5, ha="left", va="top",
             linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1.0, 0.925 - 0.033 * len(lines)))
    fig.savefig(path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    print(f"\n  heatmap -> {os.path.relpath(path, ROOT_DIR)}")


# --------------------------------------------------------------------------------- the main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hyperparam_sensitivity", description=__doc__)
    parser.add_argument("--task", default="push_t", choices=run_mod.ALL_TASKS)
    parser.add_argument("--seed", type=int, default=0, help="seed for the sweeps")
    parser.add_argument("--noise-seeds", type=int, nargs="*", default=[0, 1, 2],
                        help="seeds for the noise floor (the spec configuration only)")
    parser.add_argument("--heatmap", nargs=2, metavar=("AXIS", "AXIS"), default=None,
                        help="axes for the 2-D grid; default: the two with the widest TWA range")
    parser.add_argument("--no-heatmap", action="store_true")
    parser.add_argument("--report", action="store_true", help="report cached arms only; evaluate nothing new")
    parser.add_argument("--force", action="store_true", help="ignore the cache")
    parser.add_argument("--check-only", action="store_true", help="verify the phi rescaling and exit")
    args = parser.parse_args(argv)

    check_phi_scaling()
    if args.check_only:
        return 0

    discover()
    os.makedirs(CACHE_DIR, exist_ok=True)
    started = time.time()

    arms = oat_arms(SUM) + oat_arms(OBS)
    grids = evaluate(args.task, args.seed, arms, force=args.force, cache_only=args.report)
    table = collect(grids, arms)

    noise = report_noise(args.task, args.noise_seeds, args.force, cache_only=args.report)
    obs_twa = float(table.loc[spec_arm(OBS).name, "TWA"])

    summary = report_oat(table, SUM, noise.get(SUM))
    report_oat(table, OBS, noise.get(OBS))

    print("\n=== is the action channel ever competitive? ===")
    spec_twa = float(table.loc[spec_arm(SUM).name, "TWA"])
    sum_rows = table[table["base"] == SUM].dropna(subset=["TWA"])
    obs_rows = table[table["base"] == OBS].dropna(subset=["TWA"])
    print(f"  {SUM} at the spec's configuration      TWA {spec_twa:.4f}")
    print(f"  {OBS} at the spec's configuration      TWA {obs_twa:.4f}   "
          f"(the action channel costs {spec_twa - obs_twa:+.4f})")
    print(f"  best of the {len(sum_rows)} swept {SUM} arms         TWA {sum_rows['TWA'].max():.4f}   "
          f"({sum_rows['TWA'].idxmax()})")
    print(f"  best of the {len(obs_rows)} swept {OBS} arms         TWA {obs_rows['TWA'].max():.4f}   "
          f"({obs_rows['TWA'].idxmax()})")
    print(f"  {SUM} arms beating {OBS} at its spec value: "
          f"{int((sum_rows['TWA'] > obs_twa).sum())}/{len(sum_rows)}")
    print(f"  {SUM} arms beating the best swept {OBS} arm: "
          f"{int((sum_rows['TWA'] > obs_rows['TWA'].max()).sum())}/{len(sum_rows)}")

    degen = report_degeneracy(args.task, args.seed, table, obs_twa, args.force, args.report)

    if not args.no_heatmap:
        ax1, ax2 = args.heatmap or list(summary["axis"].head(2))
        h_arms, v1, v2 = heatmap_arms(SUM, ax1, ax2)
        grids.update(evaluate(args.task, args.seed, h_arms, force=args.force, cache_only=args.report))
        h_table = collect(grids, h_arms)
        h_df = report_heatmap(h_table, SUM, ax1, ax2, v1, v2, obs_twa)
        h_df.to_csv(os.path.join(CACHE_DIR, f"{args.task}__heatmap__{ax1}__{ax2}__seed{args.seed}.csv"),
                    index=False)
        # The sigma_A axis needs its top row read with the degeneracy in mind, so the
        # caveat travels with the picture instead of living only in the log.
        note = ""
        if "sigma_A" in (ax1, ax2) and np.isfinite(degen.get("k_A_top", np.nan)):
            note = (f"at $\\sigma_A$ = {degen['mult_top']:g}x the action factor already averages "
                    f"{degen['k_A_top']:.3f}, so those cells are the observation kernel with the "
                    f"action channel switched off rather than a tuned action channel")
        plot_heatmap(h_df, SUM, args.task, ax1, ax2, spec_twa, obs_twa, noise.get(SUM),
                     str(HERE / f"{args.task}__heatmap__{ax1}__{ax2}.png"), note=note)

    table.to_csv(os.path.join(CACHE_DIR, f"{args.task}__oat__seed{args.seed}.csv"))
    sig_path = _sigma_path(args.task, args.seed)
    if os.path.exists(sig_path):
        sig = pd.read_csv(sig_path)
        spec = sig[sig["arm"] == spec_arm(SUM).name]
        if len(spec):
            r = spec.iloc[0]
            print(f"\nmedian heuristic on {args.task} (fit set, m={int(r['m'])}): "
                  f"sigma_o {r['sigma_o_median']:.4g}, sigma_phi {r['sigma_phi_median']:.4g}, "
                  f"sigma_A {r['sigma_A_median']:.4g}")
    print(f"\ntotal wall clock: {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
