"""Regularized log-determinant kernel objective -- ``specs/experiments/logdet_kernel_objective.md``.

The criterion
-------------
``specs/methods.md`` fixes every bandwidth by the median heuristic. The alternative under
test tunes them instead by

    max_{theta, tau}   (1/n) log det(K_theta + lam I) + log(1 - tau)
    s.t.               q_i(theta) <= tau  for all i,      0 <= tau < 1,

with ``q_i`` the leave-one-out score of kern_cd on fit point ``i``.

**The constraint is not really a constraint.** ``log(1 - tau)`` is strictly decreasing in
``tau`` and the constraints only bound it from below, so the inner problem is solved in
closed form by ``tau* = max_i q_i(theta)`` and the whole thing collapses to an
unconstrained objective in ``theta`` alone:

    F(theta) = (1/n) log det(K_theta + lam I) + log(1 - max_i q_i(theta)).

Feasibility is free rather than restrictive: for an RBF kernel ``q_i = k_ii - k^T(K_{-i} +
lam I)^{-1} k <= k_ii = 1``, so ``tau < 1`` always holds (:func:`check_criterion` asserts
both facts). That is what makes this cheap: one Cholesky of ``K + lam I`` gives the log-det
*and*, via ``q_i = 1/B_ii - lam``, every LOO score -- the same identity
``KernCDMethod._loo_scores`` already uses. No inner optimisation, no gradients.

What the two terms want
-----------------------
They pull in opposite directions along a bandwidth, which is why the criterion has an
interior optimum at all:

* ``sigma -> 0``: ``K -> I``, log-det term ``-> log(1 + lam)`` (its maximum), but every
  point is isolated so ``q_i -> 1`` and the penalty ``-> -inf``.
* ``sigma -> inf``: ``K -> 11^T``, every point supports every other so ``q_i -> 0`` and the
  penalty ``-> 0``, but the gram matrix is rank-one and the log-det term ``-> -inf``.

So term one is diversity in feature space and term two is "no fit point is an island". The
median heuristic knows about neither.

What is tuned
-------------
The bandwidths the criterion can move without recomputing features: ``sigma_o`` always, and
``sigma_A`` for the action-channel methods. ``sigma_phi`` -- the bandwidth *inside* the RFF
feature map -- is left at its median value, because ``mu_A`` is a function of it, so moving
it means re-drawing phi and re-embedding every part for every candidate. If the two cheap
bandwidths show a real effect, that is the next thing to buy.

How it is wired in
------------------
``KernCDMethod._pack`` is the one place where both bandwidths have been fixed but nothing
has been consumed yet: ``fit`` calls it immediately after ``sigma_A``, and its output is
what ``KernCD`` is fitted on. So the variant overrides ``_pack``, tunes on first call
(the fit set) and delegates; later calls, from ``_features`` during scoring, see the tuned
values already installed. Nothing in ``my_methods/`` is touched, and the scored estimator is
byte-for-byte the ordinary one with two numbers changed.

What it found
-------------
Seed 0 unless noted; the summary below is what ``--summary --all-seeds`` prints.

**The criterion does not improve headline results.** Averaged over the cached seeds, across
17 (task, method) cells, the log-det-tuned twin beats the median heuristic by **+0.0016 TWA**
on the headline (best window and threshold style, quantile-averaged -- what
``paper_table.py`` reports) and loses **-0.0111 TWA** on the grid mean over all 510
configurations. Those two disagreeing is the whole result: the headline improves in 11 of 17
cells while the grid mean improves in 4, which is the signature of a better argmax rather
than a better method. The paired per-configuration comparison agrees -- on sorting the
tuned arm wins 37-45% of the 510 cells and loses 54-62% of them, while its headline goes
*up* by ~+0.02.

**The criterion is well behaved; it is just optimising the wrong thing.** It always finds an
interior optimum (the two terms are monotone in opposite directions), it always improves its
own objective (+0.4 to +1.7 nats), and its choice is stable across seeds to within 1%. It is
also consistent about direction: on every task it shrinks ``sigma_o`` to 0.31-0.84x the
median heuristic, and on the action methods it moves the two bandwidths *in opposite
directions* -- ``sigma_o`` down to ~0.4x, ``sigma_A`` up to ~1.8x -- so the spec's
expectation that different channels want different bandwidths is confirmed. That is a real
structural finding; it just does not pay.

**Why it does not pay: TWA barely depends on the bandwidth, and where it does, the criterion
usually pushes the wrong way.** ``--sweep`` scores 13 fixed multipliers end-to-end through
FIPER. Over the full 64x range of ``sigma_o``, the best achievable headline TWA exceeds the
median heuristic's by only +0.0002 (pretzel), +0.011 (sorting), +0.021 (push_t), +0.033
(push_chair) and +0.035 (stacking) -- and the TWA-optimal multiplier is 0.71x, 0.125x, 1.41x,
8x and 0.177x respectively. There is no consistent direction to move in, the total prize is
a few TWA points, and the criterion's uniform "shrink" is the right sign on two tasks of
five.

Two further cautions, both from the diagnostics this file prints:

* The criterion's ``lam`` is a free knob that ``specs/methods.md`` does not pin (it fixes
  ``lam`` for the *estimator*, where the ridge is ``lam*m``). Sweeping it over four orders of
  magnitude moves the chosen multiplier from 0.82x to 1.88x on push_chair -- i.e. across the
  median heuristic and out the other side. Nothing in the criterion says which value to use.
* ``tau = max_i q_i`` is a maximum over the fit set, so the penalty term is a function of the
  single most isolated calibration step. Replacing it with the 95th percentile of ``q`` moves
  the chosen multiplier by ~35%.

And a measurement caution: on the two smallest tasks the seed-to-seed spread of the delta
(0.098 TWA for ``kern_cd_sig`` on push_chair, where seed 0 gives -0.070 and seeds 1-2 give
+0.025/+0.028) is far larger than the effect. Single-seed cells on small tasks say nothing.

**Verdict: not worth adopting.** Tuning bandwidths on this objective costs ~50 Cholesky
factorisations and returns a headline change indistinguishable from zero, on a knob whose
whole end-to-end headroom is a few TWA points. The unexplored direction, if any, is
``sigma_phi``, which this study does not touch.

Running it
----------
    pixi run python my_experiments/logdet_kernel/logdet_kernel.py                     # push_chair, obs only
    pixi run python my_experiments/logdet_kernel/logdet_kernel.py --task pretzel
    pixi run python my_experiments/logdet_kernel/logdet_kernel.py --methods kern_cd_obs kern_cd_disp
    pixi run python my_experiments/logdet_kernel/logdet_kernel.py --check-only        # just the algebra
    pixi run python my_experiments/logdet_kernel/logdet_kernel.py --sweep             # TWA over the whole axis
    pixi run python my_experiments/logdet_kernel/logdet_kernel.py --summary --all-seeds

Per-(task, method) metric grids are cached as CSVs under
``my_experiments/logdet_kernel/cache/`` and reused; ``--force`` re-runs. The landscape scan,
the sweep and their plots land in the same directory. Nothing under ``data/`` is written.
"""

from __future__ import annotations

import argparse
import itertools
import os
import pathlib
import sys
from dataclasses import dataclass, field

HERE = pathlib.Path(__file__).resolve().parent
ROOT_DIR = str(HERE.parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.linalg import cho_factor, cho_solve, solve_triangular  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from sklearn.metrics.pairwise import euclidean_distances  # noqa: E402

from my_methods import run as run_mod  # noqa: E402
from my_methods.base import REGISTRY, discover  # noqa: E402
from my_methods.kern_cd_core import KernCDMethod  # noqa: E402

CACHE_DIR = str(HERE / "cache")

#: Suffix appended to a base method's name to get its log-det-tuned twin.
SUFFIX = "_logdet"

#: Bandwidth search range, as log2 multiples of the median heuristic: [1/8, 8].
GRID_LO, GRID_HI = -3.0, 3.0
#: Grid points per bandwidth. One bandwidth gets a fine grid for free; two get the square.
GRID_N_1D, GRID_N_2D = 25, 13


# --------------------------------------------------------------------------- the criterion
def sq_dists(X: np.ndarray) -> np.ndarray:
    """Pairwise squared euclidean distances, by the same route ``RBF`` takes.

    ``cd.algs.kernels.RBF`` calls sklearn's ``rbf_kernel``, which is
    ``exp(-gamma * euclidean_distances(X, squared=True))``. Using the same function here is
    what makes the criterion's gram matrix identical to the one ``KernCD`` will factor,
    rather than merely equal to it up to a different summation order -- which is what lets
    :func:`loo_check` assert agreement at 1e-10 instead of hoping for 1e-6.
    """
    return euclidean_distances(X, X, squared=True)


@dataclass
class Terms:
    """One evaluation of the criterion, kept in pieces because the pieces are the story."""

    obj: float
    logdet: float  # (1/n) log det(K + lam I)
    penalty: float  # log(1 - tau)
    tau: float  # max_i q_i, the tight slack
    sigmas: np.ndarray


class Criterion:
    """``F(theta)`` at fixed features, as a function of log-bandwidth multipliers.

    The features (observation embeddings, and the mean embeddings ``mu_A`` for an action
    method) do not depend on ``sigma_o`` or ``sigma_A`` at all -- those two only *rescale*
    the blocks of ``z = [o/sigma_o | mu_A/sigma_A]``. And a product of RBFs on the blocks is
    one RBF on ``z``:

        ||z - z'||^2 = ||o - o'||^2 / sigma_o^2 + ||mu - mu'||^2 / sigma_A^2.

    So the per-block squared distances are computed once, up front, and a candidate costs
    only ``exp`` of their weighted sum plus one Cholesky -- no feature recomputation, no
    kernel construction from raw data. That is the difference between a scan that takes
    seconds and one that takes an hour.
    """

    def __init__(self, blocks: list[np.ndarray], sigmas: list[float], ridge: float, tau_q: float = 1.0):
        self.sq = [sq_dists(np.ascontiguousarray(X)) for X in blocks]
        self.sigma0 = np.asarray(sigmas, dtype=float)
        self.ridge = float(ridge)
        #: Quantile of ``q`` standing in for ``tau``. 1.0 is the criterion as specified --
        #: ``tau`` upper-bounds *every* ``q_i``. Anything less is a deliberate weakening,
        #: used only by :func:`report_tau_robustness` to measure how much of the criterion's
        #: answer rests on the single most isolated fit point.
        self.tau_q = float(tau_q)
        self.m = int(self.sq[0].shape[0])
        self.n_evals = 0

    @property
    def n_blocks(self) -> int:
        return len(self.sq)

    def gram(self, u: np.ndarray) -> np.ndarray:
        sigmas = self.sigma0 * np.exp(np.asarray(u, dtype=float).ravel())
        acc = np.zeros_like(self.sq[0])
        for D2, s in zip(self.sq, sigmas):
            acc += D2 / (s * s)
        return np.exp(-0.5 * acc)

    def terms(self, u: np.ndarray) -> Terms:
        """Evaluate at ``sigma = sigma_median * exp(u)``.

        An infeasible or numerically degenerate candidate returns ``-inf`` rather than
        raising: at very small bandwidths ``tau`` reaches 1 in double precision (every point
        genuinely is an island), which is the criterion correctly assigning it no value.
        """
        self.n_evals += 1
        sigmas = self.sigma0 * np.exp(np.asarray(u, dtype=float).ravel())
        K = self.gram(u)
        M = K + self.ridge * np.eye(self.m)
        try:
            L = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            return Terms(-np.inf, -np.inf, -np.inf, 1.0, sigmas)
        logdet = float(2.0 * np.sum(np.log(np.diag(L))) / self.m)
        q = self.loo(L)
        # tau is the tightest feasible slack (see the module docstring); clipped into
        # [0, 1) only against roundoff, never to hide an infeasible candidate -- q > 1 is
        # impossible for this kernel and q < 0 only ever appears at the 1e-16 level.
        tau = float(min(max(q.max() if self.tau_q >= 1.0 else np.quantile(q, self.tau_q), 0.0), 1.0))
        if tau >= 1.0:
            return Terms(-np.inf, logdet, -np.inf, tau, sigmas)
        penalty = float(np.log1p(-tau))
        return Terms(logdet + penalty, logdet, penalty, tau, sigmas)

    def loo(self, L: np.ndarray) -> np.ndarray:
        """``q_i = 1/B_ii - lam``, ``B = (K + lam I)^{-1}`` -- kern_cd's own LOO identity.

        Identical to ``KernCDMethod._loo_scores`` (which writes the ridge as ``lam * m``);
        :func:`loo_check` asserts that on the fitted models rather than taking it on trust.
        """
        L_inv = solve_triangular(L, np.eye(L.shape[0]), lower=True)
        return 1.0 / np.einsum("ki,ki->i", L_inv, L_inv) - self.ridge


def objective_at_fit(impl) -> Terms:
    """The criterion at a *fitted* method's own bandwidths, for free.

    Both ingredients are already lying around after ``fit``: the Cholesky factor of
    ``K + lam*m*I`` on the model, and the LOO scores on the method. So the baseline's
    objective value costs no kernel evaluation at all -- which is the number the tuned
    variant has to beat.
    """
    m = len(impl.model.data)
    logdet = float(2.0 * np.sum(np.log(np.diag(impl.model.L))) / m)
    tau = float(min(max(np.asarray(impl.loo_scores).max(), 0.0), 1.0))
    penalty = float(np.log1p(-tau)) if tau < 1.0 else -np.inf
    sigmas = np.array([impl._sigma_o] + ([impl._sigma_A] if impl.uses_actions else []))
    return Terms(logdet + penalty, logdet, penalty, tau, sigmas)


def loo_check(impl) -> float:
    """Max relative gap between :class:`Criterion`'s ``q`` and the method's ``loo_scores``.

    The criterion is only measuring the right thing if its ``q_i`` *is* the estimator's LOO
    score. Recomputing it from the packed matrix the model was fitted on -- rather than from
    the criterion's own blocks -- checks the whole chain, scaling included.
    """
    Z = np.asarray(impl.model.data)
    m = len(Z)
    ridge = float(impl.model.lam_) * m
    L = np.linalg.cholesky(np.exp(-0.5 * sq_dists(Z)) + ridge * np.eye(m))
    L_inv = solve_triangular(L, np.eye(m), lower=True)
    q = 1.0 / np.einsum("ki,ki->i", L_inv, L_inv) - ridge
    ref = np.asarray(impl.loo_scores)
    return float(np.max(np.abs(q - ref) / np.maximum(np.abs(ref), 1e-12)))


def check_criterion(seed: int = 0) -> None:
    """Verify on synthetic data everything the module docstring asserts about the criterion.

    Four claims, in the order they are relied on:

    1. ``q_i`` is the honest leave-one-out variance -- it matches refitting from scratch
       with point ``i`` deleted.
    2. ``q_i <= 1``, so ``tau < 1`` is never binding and the problem is always feasible.
    3. ``tau* = max_i q_i``: no smaller value is feasible, no larger one scores better.
    4. The two terms are monotone in opposite directions along a bandwidth, so ``F`` has an
       interior optimum rather than running off to an edge.
    """
    rng = np.random.default_rng(seed)
    m, d = 60, 5
    X = rng.normal(size=(m, d))
    ridge = 1e-5 * m
    crit = Criterion([X], [1.0], ridge)

    # 1: closed-form LOO == brute-force refit.
    K = crit.gram(np.zeros(1))
    L = np.linalg.cholesky(K + ridge * np.eye(m))
    q = crit.loo(L)
    brute = np.empty(m)
    for i in range(m):
        idx = np.arange(m) != i
        M = K[np.ix_(idx, idx)] + ridge * np.eye(m - 1)
        k_vec = K[idx, i]
        brute[i] = K[i, i] - k_vec @ cho_solve(cho_factor(M, lower=True), k_vec)
    assert np.allclose(brute, q, atol=1e-9, rtol=1e-7), np.abs(brute - q).max()

    # 2: feasibility is free, at every bandwidth on the search range.
    scan = [crit.terms(np.array([u])) for u in np.linspace(GRID_LO, GRID_HI, 41)]
    assert all(0.0 <= t.tau <= 1.0 for t in scan), "q_i left [0, 1]"

    # 3: tau* = max_i q_i solves the inner problem -- nothing smaller is feasible (some
    # constraint q_i <= tau is violated), nothing larger scores as well.
    t = crit.terms(np.zeros(1))
    assert np.isclose(t.tau, q.max())
    assert all((q > tau).any() for tau in (t.tau * 0.999, t.tau * 0.9)), "a smaller tau was feasible?"
    assert all(np.log1p(-tau) < t.penalty for tau in t.tau + (1.0 - t.tau) * np.array([0.1, 0.5])), (
        "a larger tau scored better?"
    )

    # 4: opposite monotonicity, hence an interior optimum.
    # Compared pairwise rather than through np.diff: the penalty is -inf at the smallest
    # bandwidths (tau reaches 1 in double precision), and inf - inf is nan.
    logdets = np.array([s.logdet for s in scan])
    penalties = np.array([s.penalty for s in scan])
    assert np.all(logdets[1:] <= logdets[:-1] + 1e-12), "log-det term is not decreasing in sigma"
    assert np.all(penalties[1:] >= penalties[:-1] - 1e-12), "penalty term is not increasing in sigma"
    objs = np.array([s.obj for s in scan])
    best = int(np.argmax(objs))
    assert 0 < best < len(objs) - 1, f"optimum at the edge of the search range (index {best})"

    print(
        f"criterion OK (m={m}): closed-form q == brute-force LOO (max diff "
        f"{np.abs(brute - q).max():.2e}); q stays in [0, 1] over the whole search range "
        f"(reaching 1 only in the sigma -> 0 limit, where the objective is -inf anyway, so "
        f"tau < 1 never binds at any candidate worth considering); tau* = max_i q_i; "
        f"log-det falls and the penalty rises with sigma, optimum interior at "
        f"{np.exp(np.linspace(GRID_LO, GRID_HI, 41)[best]):.2f}x"
    )


# --------------------------------------------------------------------------- the optimiser
@dataclass
class TuneResult:
    """What the tuner found, plus everything the report needs to explain it."""

    mults: np.ndarray  # sigma_tuned / sigma_median, per block
    best: Terms
    ref: Terms  # the criterion at the median heuristic, i.e. what is being beaten
    scan: pd.DataFrame
    n_evals: int
    at_edge: bool
    labels: list[str] = field(default_factory=list)
    #: label -> 1-D scan with the *other* bandwidths held at their tuned values. Taken from
    #: the optimum rather than read off the nearest grid line, so the optimum lies on the
    #: plotted curve by construction instead of hovering above it.
    profiles: dict[str, pd.DataFrame] = field(default_factory=dict)


def tune(crit: Criterion, labels: list[str], grid_n: int | None = None, polish: bool = True) -> TuneResult:
    """Maximise the criterion over log-bandwidth multipliers: grid, then Nelder-Mead.

    The grid is not an implementation detail that a smarter optimiser would remove -- it is
    half the deliverable. A local method started at the median heuristic would report a
    multiplier and nothing else; the grid says whether the objective is unimodal, how sharp
    the optimum is, and how far the median heuristic sits from it, which is what decides
    whether this criterion is worth anything. The polish then buys the last fraction of a
    grid step so the reported multiplier is not quantised.

    Cost is one Cholesky plus one triangular inverse per candidate, both ``O(m^3)``, on a
    gram matrix that is never rebuilt from the raw features.
    """
    n = crit.n_blocks
    if grid_n is None:
        grid_n = GRID_N_1D if n == 1 else GRID_N_2D
    axis = np.linspace(GRID_LO, GRID_HI, grid_n)

    rows = []
    for u in itertools.product(*[axis] * n):
        t = crit.terms(np.asarray(u))
        row = {f"mult_{lab}": float(np.exp(ui)) for lab, ui in zip(labels, u)}
        row.update(obj=t.obj, logdet=t.logdet, penalty=t.penalty, tau=t.tau)
        row.update({f"sigma_{lab}": float(s) for lab, s in zip(labels, t.sigmas)})
        rows.append(row)
    scan = pd.DataFrame(rows)

    u_best = np.array(
        [np.log(scan.loc[scan["obj"].idxmax(), f"mult_{lab}"]) for lab in labels], dtype=float
    )
    if polish:
        res = minimize(
            lambda u: -crit.terms(u).obj,
            u_best,
            method="Nelder-Mead",
            options=dict(xatol=1e-3, fatol=1e-7, maxiter=200 * n),
        )
        if np.isfinite(res.fun) and -res.fun >= scan["obj"].max():
            u_best = np.asarray(res.x, dtype=float)

    best = crit.terms(u_best)

    profiles = {}
    for k, lab in enumerate(labels):
        rows = []
        for ui in np.append(axis, u_best[k]):
            u = u_best.copy()
            u[k] = ui
            t = crit.terms(u)
            rows.append(dict(mult=float(np.exp(ui)), obj=t.obj, logdet=t.logdet, penalty=t.penalty, tau=t.tau))
        profiles[lab] = pd.DataFrame(rows).sort_values("mult").reset_index(drop=True)

    return TuneResult(
        mults=np.exp(u_best),
        best=best,
        ref=crit.terms(np.zeros(n)),
        scan=scan,
        n_evals=crit.n_evals,
        at_edge=bool(np.any(u_best <= GRID_LO + 1e-9) or np.any(u_best >= GRID_HI - 1e-9)),
        labels=list(labels),
        profiles=profiles,
    )


# ----------------------------------------------------------------------------- the variant
class _LogDetMixin:
    """Replaces the median heuristic for ``sigma_o`` (and ``sigma_A``) with the criterion.

    ``_pack`` is the hinge: ``KernCDMethod.fit`` calls it once, after both bandwidths are
    fixed and before anything consumes them, and ``_features`` calls it again on every
    scoring pass. So the first call -- always the fit set -- tunes and installs, and every
    later call sees the installed values. The median heuristic still runs first and its
    output is kept as ``_sigma_o_median`` / ``_sigma_A_median``: it is the reference point
    the multipliers are measured against, and the initialisation the polish starts from.
    """

    #: ``lam * m * logdet_ridge_scale``. The default matches the ridge ``KernCD`` actually
    #: puts on the diagonal, so the criterion's ``K + lam I`` is the matrix that gets
    #: factored and its ``q_i`` are the estimator's own LOO scores rather than a proxy.
    logdet_ridge_scale = 1.0
    logdet_grid_n: int | None = None
    logdet_polish = True

    def _pack(self, obs: np.ndarray, mu: np.ndarray | None) -> np.ndarray:
        if not getattr(self, "_logdet_done", False):
            self._logdet_tune(obs, mu)
            self._logdet_done = True
        return super()._pack(obs, mu)

    def _logdet_tune(self, obs: np.ndarray, mu: np.ndarray | None) -> None:
        self._sigma_o_median = float(self._sigma_o)
        blocks, sigmas, labels = [obs], [self._sigma_o], ["o"]
        if mu is not None:
            self._sigma_A_median = float(self._sigma_A)
            blocks.append(mu)
            sigmas.append(self._sigma_A)
            labels.append("A")

        ridge = float(self.p.lam) * len(obs) * self.logdet_ridge_scale
        crit = Criterion(blocks, sigmas, ridge)
        res = tune(crit, labels, grid_n=self.logdet_grid_n, polish=self.logdet_polish)
        self._logdet_result = res
        self._logdet_ridge = ridge

        self._sigma_o = float(res.best.sigmas[0])
        if mu is not None:
            self._sigma_A = float(res.best.sigmas[1])


def make_variant(base_name: str) -> type[KernCDMethod]:
    """Register (once) the log-det-tuned twin of a registered kern_cd method."""
    name = base_name + SUFFIX
    if name in REGISTRY:
        return REGISTRY[name]
    base = REGISTRY[base_name]
    cls = type(base.__name__ + "LogDet", (_LogDetMixin, base), {"name": name})
    cls.__doc__ = f"{base_name} with bandwidths tuned by the log-det criterion."
    return cls


# ------------------------------------------------------------------------- the sweep
#: Bandwidth multipliers scored end-to-end by the FIPER pipeline, sqrt(2) apart over the
#: same [1/8, 8] the criterion searches.
SWEEP_MULTS = (0.125, 0.177, 0.25, 0.354, 0.5, 0.707, 1.0, 1.414, 2.0, 2.828, 4.0, 5.657, 8.0)


class _FixedMultMixin:
    """Scales ``sigma_o`` by a fixed multiple of the median heuristic. The control arm.

    "The criterion did not improve TWA" has two very different explanations -- the criterion
    picks a bad bandwidth, or TWA does not care about the bandwidth -- and the tuned-vs-median
    comparison cannot separate them, because it only ever evaluates two points on the axis.
    Scoring the whole axis end-to-end does: it shows the shape of TWA as a function of the
    bandwidth, where the median heuristic sits on it, where the criterion moved to, and what
    the best attainable value would have been.
    """

    logdet_mult = 1.0

    def _pack(self, obs: np.ndarray, mu: np.ndarray | None) -> np.ndarray:
        if not getattr(self, "_mult_done", False):
            self._sigma_o_median = float(self._sigma_o)
            self._sigma_o = float(self._sigma_o) * float(self.logdet_mult)
            self._mult_done = True
        return super()._pack(obs, mu)


def sweep_name(base_name: str, mult: float) -> str:
    return f"{base_name}_x{mult:g}"


def make_sweep_variant(base_name: str, mult: float) -> type[KernCDMethod]:
    name = sweep_name(base_name, mult)
    if name in REGISTRY:
        return REGISTRY[name]
    base = REGISTRY[base_name]
    cls = type(f"{base.__name__}X{str(mult).replace('.', '_')}", (_FixedMultMixin, base),
               {"name": name, "logdet_mult": float(mult)})
    cls.__doc__ = f"{base_name} with sigma_o fixed at {mult}x the median heuristic."
    return cls


# --------------------------------------------------------------------------- the reporting
def grid(name: str, task: str, results: dict) -> pd.DataFrame:
    """The full (threshold style x quantile x window) metric grid for one method."""
    return pd.DataFrame(run_mod._rows(name, task, results))


def headline(df: pd.DataFrame) -> pd.Series:
    """The number the paper table reports: best (window, threshold) by quantile-averaged TWA.

    ``my_methods/paper_table.py`` averages over quantiles and then picks the best window and
    threshold style by TWA, so that is what a change has to move to matter. ``TWA_gridmean``
    -- the flat mean over all 510 configurations -- is reported beside it because the
    headline is an argmax over 51 noisy numbers and can move on its own; the grid mean
    cannot.
    """
    avg = df.groupby(["Threshold", "Window"], as_index=False)[["TWA", "Accuracy", "Det. Time"]].mean()
    best = avg.loc[avg["TWA"].idxmax()]
    return pd.Series(
        {
            "TWA": best["TWA"],
            "Accuracy": best["Accuracy"],
            "Det. Time": best["Det. Time"],
            "Threshold": best["Threshold"],
            "Window": best["Window"],
            "TWA_gridmax": df["TWA"].max(),
            "TWA_gridmean": df["TWA"].mean(),
        }
    )


def report_tuning(task: str, impls: dict) -> None:
    """What the criterion chose, what it was worth, and whether it measured the right thing."""
    print(f"\n=== log-det criterion, {task} ===")
    for name, impl in impls.items():
        res = getattr(impl, "_logdet_result", None)
        fitted = objective_at_fit(impl)
        m = len(impl.model.data)
        gap = loo_check(impl)
        print(f"\n-- {name}   (m = {m} fit steps, ridge lam*m = {impl.model.lam_ * m:.3g})")
        print(f"  criterion's q_i vs the method's loo_scores: max rel diff {gap:.2e}")
        assert gap < 1e-6, f"{name}: the criterion is not evaluating kern_cd's own LOO scores"
        if res is None:
            print("  bandwidths (median heuristic)   " + ", ".join(f"{s:.4g}" for s in fitted.sigmas))
            print(f"  objective at those bandwidths   {fitted.obj:+.4f}  "
                  f"= logdet/n {fitted.logdet:+.4f}  +  log(1-tau) {fitted.penalty:+.4f}   (tau {fitted.tau:.4g})")
            continue
        for lab, mult, sig in zip(res.labels, res.mults, res.best.sigmas):
            median = getattr(impl, f"_sigma_{lab}_median")
            print(f"  sigma_{lab}: median {median:.4g}  ->  tuned {sig:.4g}   ({mult:.3f}x)")
        print(f"  objective   median {res.ref.obj:+.4f}  ->  tuned {res.best.obj:+.4f}   "
              f"({res.best.obj - res.ref.obj:+.4f})")
        print(f"    logdet/n  {res.ref.logdet:+.4f}  ->  {res.best.logdet:+.4f}")
        print(f"    log(1-tau){res.ref.penalty:+.4f}  ->  {res.best.penalty:+.4f}   "
              f"(tau {res.ref.tau:.4g} -> {res.best.tau:.4g})")
        print(f"  {res.n_evals} criterion evaluations" + ("   [WARNING: optimum at the search edge]" if res.at_edge else ""))
        assert abs(fitted.obj - res.best.obj) < 1e-6, "the fitted model is not at the tuned bandwidths"


def report_loo_shift(task: str, impls: dict) -> None:
    """How the calibration scores move -- one of the two channels to the metrics.

    The other is the test scores, which the base variant's LOO comparison does not have:
    unlike a change of LOO rule, a change of bandwidth moves the score function itself, so
    both the thresholds and the numbers they are compared against shift.
    """
    print(f"\n=== calibration (LOO) score distribution, {task} ===")
    rows = []
    for name, impl in impls.items():
        s = np.asarray(impl.loo_scores)
        rows.append({"method": name, "min": s.min(), "q10": np.quantile(s, 0.10), "median": np.median(s),
                     "q90": np.quantile(s, 0.90), "q95": np.quantile(s, 0.95), "max": s.max()})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4g}"))


def report_ridge_sensitivity(impls: dict, scales=(0.01, 0.1, 1.0, 10.0, 100.0)) -> None:
    """Does the answer depend on the one number the criterion does not pin down?

    ``specs/methods.md`` fixes ``lam = 1e-5`` for the *estimator*, where the matrix factored
    is ``K + lam*m*I``; the criterion's ``lam`` is written as an absolute ridge and is a free
    knob. This re-tunes at several multiples of the estimator's ridge, on the already-built
    distance matrices, and prints the multiplier each one picks. A criterion whose optimum
    swings with a knob nobody has a principled value for is not a criterion.
    """
    tuned = {n: i for n, i in impls.items() if hasattr(i, "_logdet_result")}
    if not tuned:
        return
    print("\n=== ridge sensitivity: the criterion's lam is a free knob ===")
    rows = []
    for name, impl in tuned.items():
        res = impl._logdet_result
        blocks = [impl._logdet_obs] + ([impl._logdet_mu] if impl._logdet_mu is not None else [])
        sigmas = [getattr(impl, f"_sigma_{lab}_median") for lab in res.labels]
        for scale in scales:
            ridge = float(impl.p.lam) * len(blocks[0]) * scale
            crit = Criterion(blocks, sigmas, ridge)
            r = tune(crit, res.labels, polish=True)
            row = {"method": name, "ridge / (lam*m)": scale, "ridge": ridge, "obj": r.best.obj, "tau": r.best.tau}
            row.update({f"mult_{lab}": mult for lab, mult in zip(res.labels, r.mults)})
            rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4g}"))


def report_tau_robustness(impls: dict, quantiles=(1.0, 0.999, 0.99, 0.95)) -> None:
    """How much of the criterion's answer rests on a single fit point.

    ``tau`` upper-bounds *every* ``q_i``, so it is a maximum over the fit set and the penalty
    term is a function of one point -- the most isolated step in the calibration split. That
    is a design choice with consequences: a single outlying step (a rollout's first frame,
    an unusual grasp) sets the bandwidth for all m of them. Re-tuning with ``tau`` taken as a
    high quantile of ``q`` instead measures the exposure. A multiplier that barely moves says
    the maximum is representative; one that jumps says the criterion is fitting an outlier.

    Nothing here changes the criterion under test -- the tuned methods all use the specified
    ``tau = max_i q_i``. This is a diagnostic on it.
    """
    tuned = {n: i for n, i in impls.items() if hasattr(i, "_logdet_result")}
    if not tuned:
        return
    print("\n=== tau robustness: the penalty term is a maximum over the fit set ===")
    rows = []
    for name, impl in tuned.items():
        res = impl._logdet_result
        blocks = [impl._logdet_obs] + ([impl._logdet_mu] if impl._logdet_mu is not None else [])
        sigmas = [getattr(impl, f"_sigma_{lab}_median") for lab in res.labels]
        for q in quantiles:
            crit = Criterion(blocks, sigmas, impl._logdet_ridge, tau_q=q)
            r = tune(crit, res.labels, polish=True)
            row = {"method": name, "tau": "max" if q >= 1.0 else f"q{q:g}", "tau_value": r.best.tau,
                   "obj": r.best.obj}
            row.update({f"mult_{lab}": mult for lab, mult in zip(res.labels, r.mults)})
            rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4g}"))


# ------------------------------------------------------------------------------- the plot
#: Three series, one axis, same units (nats). Slots 1-3 of the categorical palette already
#: validated for this repo in ``my_experiments/profiling/profiling.py``; re-checked here for this
#: subset (worst adjacent CVD dE 9.2 deutan, normal-vision 27.6, light surface #fcfcfb).
_COLORS = {"objective": "#2a78d6", "logdet": "#eb6834", "penalty": "#1baf7a"}
_INK, _MUTED, _GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def _label_indices(series: list[np.ndarray], avoid: tuple[int, ...] = (), halo: int = 2) -> list[int]:
    """One direct-label anchor per curve: each where it is furthest from everything else.

    Two of these three curves converge at large bandwidth (once ``tau -> 0`` the penalty
    vanishes and the objective *is* the log-det term), so a fixed anchor -- the right end,
    the maximum, anywhere -- collides. Each candidate x is scored by the vertical gap to the
    nearest other curve, in axis-normalised units, and anchors are assigned greedily,
    best-separated curve first.

    Two constraints then apply, and they are deliberately unequal in strength. Anchors must
    be **distinct**, at least ``halo`` apart: two labels at one x overlap no matter how far
    apart their curves are, so that one is never relaxed -- an exhausted curve takes the
    x furthest from every anchor already placed. Staying clear of the axis edges and of the
    fixed annotations (the ``avoid`` positions) only shapes the preference, because a label
    a little close to a dashed reference line is still readable.
    """
    stack = np.vstack(series)
    span = np.ptp(stack[np.isfinite(stack)]) or 1.0
    n = stack.shape[1]
    lo, hi = int(0.15 * n), max(int(0.15 * n) + 1, int(0.85 * n))

    preference = []
    for i in range(len(series)):
        g = np.min(np.abs(stack - series[i]) / span + np.eye(len(series))[:, i, None] * 1e9, axis=0)
        g = np.nan_to_num(g, nan=0.0)
        g[:lo] = g[hi:] = 0.0
        for a in avoid:
            g[max(0, a - halo) : a + halo + 1] = 0.0
        preference.append(g)

    out = [0] * len(series)
    taken: list[int] = []
    for i in sorted(range(len(series)), key=lambda k: -preference[k].max()):
        free = np.array([j for j in range(n) if all(abs(j - t) > halo for t in taken)])
        if not len(free):  # more labels than the axis can hold; nothing sensible is left
            free = np.arange(n)
        score = preference[i][free]
        # Every preferred position is taken: fall back to the x furthest from the anchors
        # already placed, which keeps the labels apart even when it cannot keep them clear.
        j = int(free[np.argmax(score)]) if score.max() > 0 else int(
            free[np.argmax([min(abs(int(j) - t) for t in taken) if taken else 0 for j in free])]
        )
        out[i] = j
        taken.append(j)
    return out


def _plot_surface(ax, res: TuneResult, fig) -> None:
    """The objective over both bandwidth multipliers: one sequential hue, light -> dark.

    Whether the optimum is a point or a diagonal ridge is the whole question for the
    action-channel methods, and it is a property of the surface that no profile shows. The
    colour range is clipped to the top 10 nats -- below that the penalty term is diverging
    and every candidate is equally worthless -- and the clip is named on the colour bar.
    """
    x_lab, y_lab = res.labels
    piv = res.scan.pivot_table(index=f"mult_{y_lab}", columns=f"mult_{x_lab}", values="obj")
    X, Y, Z = piv.columns.to_numpy(), piv.index.to_numpy(), piv.to_numpy()
    top = np.nanmax(Z[np.isfinite(Z)])
    # Clipped rather than left to render as "no data": below the floor the penalty term is
    # diverging and every candidate is equally worthless, which is a value, not a gap.
    Z = np.maximum(np.nan_to_num(Z, nan=-np.inf), top - 10.0)

    mesh = ax.pcolormesh(X, Y, Z, cmap="Blues", vmin=top - 10.0, vmax=top, shading="nearest")
    ax.contour(X, Y, Z, levels=np.linspace(top - 8, top, 5), colors="#fcfcfb", linewidths=0.8, alpha=0.8)
    bar = fig.colorbar(mesh, ax=ax, pad=0.02)
    bar.set_label("objective (nats, clipped 10 below the optimum)", color=_MUTED, fontsize=9)
    bar.ax.tick_params(colors=_MUTED, labelsize=8)
    bar.outline.set_visible(False)

    # Both labels sit on top of the colour ramp, whose dark end is far too dark for ink
    # text, so they carry the surface colour behind them rather than relying on the cell.
    plate = dict(boxstyle="round,pad=0.25", facecolor="#fcfcfb", edgecolor="none", alpha=0.85)
    ax.plot([1.0], [1.0], "o", ms=8, color="#fcfcfb", mec=_INK, mew=1.5, zorder=4)
    ax.annotate("median heuristic", (1.0, 1.0), textcoords="offset points", xytext=(0, 13),
                color=_INK, fontsize=9, ha="center", zorder=5, bbox=plate)
    ax.plot([res.mults[0]], [res.mults[1]], "o", ms=8, color=_COLORS["objective"], mec="#fcfcfb", mew=2, zorder=4)
    ax.annotate(f"optimum {res.mults[0]:.2f}x, {res.mults[1]:.2f}x", (res.mults[0], res.mults[1]),
                textcoords="offset points", xytext=(0, -18), color=_INK, fontsize=9,
                ha="center", va="top", zorder=5, bbox=plate)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel(f"$\\sigma_{x_lab}$ / median heuristic", color=_MUTED, fontsize=9)
    ax.set_ylabel(f"$\\sigma_{y_lab}$ / median heuristic", color=_MUTED, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=9)


def plot_landscape(res: TuneResult, task: str, name: str, path: str) -> None:
    """The objective and its two terms against each bandwidth, through the optimum.

    One panel per bandwidth: that bandwidth varies, the other is held at its tuned value.
    With two bandwidths a third panel draws the surface itself, because the profiles cannot
    answer the question the action-channel methods raise -- whether the two bandwidths have
    genuinely different optima or the objective only sees some combination of them, in which
    case the "second bandwidth" is not a second degree of freedom at all.

    The x-range is trimmed to where the objective is within 10 nats of its maximum, and the
    trimming is stated on the axis label. Outside it the penalty term diverges and the
    optimum's neighbourhood -- the only region where the median heuristic and the criterion
    can be told apart -- is compressed to a point. The untrimmed scan is in the CSV beside
    this file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = res.labels
    n_panels = len(labels) + (1 if len(labels) == 2 else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 4.4), squeeze=False)
    for ax, lab in zip(axes[0], labels):
        prof = res.profiles[lab]
        prof = prof[prof["obj"] >= prof["obj"].max() - 10.0]
        x = prof["mult"].to_numpy()

        # Short text on the curve, the formula in the legend: a direct label has to fit in
        # the gap between two curves, and a rendered fraction does not.
        drawn = [
            ("logdet", prof["logdet"].to_numpy(), "log-det", r"$\frac{1}{n}\log\det(K+\lambda I)$"),
            ("penalty", prof["penalty"].to_numpy(), "penalty", r"$\log(1-\tau)$"),
            ("objective", prof["obj"].to_numpy(), "objective", "objective (their sum)"),
        ]
        for key, series, _, legend in drawn:
            ax.plot(x, series, color=_COLORS[key], lw=2.0, label=legend, zorder=3 if key == "objective" else 2)
        mult = float(res.mults[labels.index(lab)])
        anchors = _label_indices(
            [s for _, s, _, _ in drawn],
            avoid=(int(np.abs(x - mult).argmin()), int(np.abs(x - 1.0).argmin())),
        )
        for (key, series, label, _), j in zip(drawn, anchors):
            # Centred text runs off the axes when the anchor is near an edge, so the label
            # hangs inward from there instead.
            side = "left" if j < 0.25 * len(x) else ("right" if j > 0.75 * len(x) else "center")
            ax.annotate(label, (x[j], series[j]), textcoords="offset points",
                        xytext=({"left": -4, "center": 0, "right": 4}[side], 9),
                        color=_MUTED, fontsize=9, ha=side, zorder=5)

        ax.axvline(1.0, color="#b3b2a8", lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.annotate("median heuristic", (1.0, 1.0), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(5, -11), color=_MUTED, fontsize=9, ha="left")
        ax.plot([mult], [res.best.obj], "o", ms=8, color=_COLORS["objective"], mec="#fcfcfb", mew=2, zorder=4)
        ax.annotate(f"optimum {mult:.2f}x", (mult, res.best.obj), textcoords="offset points", xytext=(0, -16),
                    color=_INK, fontsize=9, ha="center", va="top", zorder=5)

        ax.set_xscale("log", base=2)
        ax.set_xlabel(f"$\\sigma_{lab}$ / median heuristic   (range trimmed to within 10 nats of the optimum)",
                      color=_MUTED, fontsize=9)
        ax.set_ylabel("nats", color=_MUTED, fontsize=9)
        ax.grid(True, color=_GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_GRID)
        ax.tick_params(colors=_MUTED, labelsize=9)
        ax.margins(x=0.05, y=0.12)
        # "best" rather than a fixed corner: the curves fill a different part of the panel
        # for every method and task, and the direct labels have already claimed the gaps.
        ax.legend(loc="best", frameon=False, fontsize=9, labelcolor=_MUTED)

    if len(labels) == 2:
        _plot_surface(axes[0][-1], res, fig)

    fig.suptitle(f"log-det criterion vs bandwidth -- {name} on {task}", color=_INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1.0, 0.95))
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"  landscape plot -> {os.path.relpath(path, ROOT_DIR)}")


def plot_sweep(df: pd.DataFrame, task: str, name: str, mult_logdet: float | None, path: str) -> None:
    """TWA as a function of the bandwidth, with the two candidate bandwidths marked.

    Two series on one axis because both are TWA in the same units: the headline (an argmax
    over 51 window/threshold configurations, which is what the paper table reports) and the
    flat mean over all 510 (no selection, so it moves only when the method genuinely got
    better everywhere).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    x = df["mult"].to_numpy()
    drawn = [
        ("objective", df["TWA"].to_numpy(), "headline TWA"),
        ("logdet", df["TWA_gridmean"].to_numpy(), "grid-mean TWA"),
    ]
    for key, y, label in drawn:
        ax.plot(x, y, color=_COLORS[key], lw=2.0, marker="o", ms=5, label=label)
    anchors = _label_indices([y for _, y, _ in drawn],
                             avoid=tuple(int(np.abs(x - v).argmin()) for v in (1.0, mult_logdet or 1.0)))
    for (key, y, label), j in zip(drawn, anchors):
        side = "left" if j < 0.25 * len(x) else ("right" if j > 0.75 * len(x) else "center")
        ax.annotate(label, (x[j], y[j]), textcoords="offset points",
                    xytext=({"left": -4, "center": 0, "right": 4}[side], 10),
                    color=_MUTED, fontsize=9, ha=side, zorder=5)

    for mult, text, colour in ((1.0, "median heuristic", "#b3b2a8"),
                               (mult_logdet, "log-det criterion", _COLORS["penalty"])):
        if mult is None:
            continue
        ax.axvline(mult, color=colour, lw=1.4, ls=(0, (4, 3)), zorder=1)
        ax.annotate(text, (mult, 1.0), xycoords=("data", "axes fraction"), textcoords="offset points",
                    xytext=(5, -11), color=_MUTED, fontsize=9, ha="left", zorder=5)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("$\\sigma_o$ / median heuristic", color=_MUTED, fontsize=9)
    # The whole point of the chart is how *little* TWA moves, and a zoomed axis argues the
    # opposite, so the zoom is named on the axis rather than left for the reader to notice.
    ax.set_ylabel(f"TWA (axis zoomed to {df['TWA_gridmean'].min():.2f}-{df['TWA'].max():.2f}; "
                  "the metric runs 0-1)", color=_MUTED, fontsize=9)
    ax.grid(True, color=_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.margins(x=0.05, y=0.18)
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=_MUTED)
    fig.suptitle(f"TWA vs observation bandwidth -- {name} on {task}", color=_INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1.0, 0.95))
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"  sweep plot -> {os.path.relpath(path, ROOT_DIR)}")


# --------------------------------------------------------------------------------- the run
#: ``_evaluate_task`` returns metrics, not the method objects, and the objects hold the
#: fitted state this experiment is about (bandwidths, the scan, the LOO scores). The adapter
#: builds them inside ``EvaluationManager``, so they are captured on the way past rather
#: than plumbed through my_methods/ -- which stays untouched.
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


def _keep_tuning_inputs() -> None:
    """Retain the fit-set blocks on the impl, so the ridge sweep can re-tune without refitting.

    ``_pack`` is handed exactly the two arrays the criterion needs and then drops them. They
    are ``[m, d_o]`` and ``[m, 128]`` -- 2 MB at the largest task -- so keeping them is far
    cheaper than a second evaluation pass.
    """
    original = _LogDetMixin._logdet_tune

    def patched(self, obs, mu):
        self._logdet_obs, self._logdet_mu = obs, mu
        return original(self, obs, mu)

    _LogDetMixin._logdet_tune = patched


def summarise(seed: int | None = None) -> int:
    """Roll every cached (task, method) pair up into the paper table's footing.

    Every cached seed of a cell is averaged, and the seed-to-seed spread of the *delta* is
    printed separately for the cells that have more than one. That second table is not
    decoration: on the smallest task one action method's delta is -0.070 at seed 0 and
    +0.025 / +0.028 at seeds 1 and 2, which is a sign flip of the same size as the effect
    being measured. A single-seed cell on a small task carries no information about the
    criterion at all.
    """
    pattern = "*__seed*.csv" if seed is None else f"*__seed{seed}.csv"
    files = sorted(f for f in pathlib.Path(CACHE_DIR).glob(pattern) if "__sweep__" not in f.name)
    if not files:
        print(f"no cached grids in {CACHE_DIR}", file=sys.stderr)
        return 1

    grids: dict[tuple[str, str, int], pd.DataFrame] = {}
    for f in files:
        task, name, tail = f.name.rsplit("__", 2)
        grids[(task, name, int(tail.removeprefix("seed").removesuffix(".csv")))] = pd.read_csv(f)

    tasks = [t for t in run_mod.ALL_TASKS if any(k[0] == t for k in grids)]
    bases = [n for n in ("kern_cd_obs", "kern_cd_disp", "kern_cd_sig", "kern_cd_flat", "kern_cd_sum")
             if any(k[1] == n for k in grids)]

    for label, get in (
        ("headline TWA (best window+threshold, quantile-averaged)", lambda g: float(headline(g)["TWA"])),
        ("grid-mean TWA (all 510 configs, no selection)", lambda g: float(g["TWA"].mean())),
    ):
        cells: dict[tuple[str, str], list[float]] = {}
        med_rows, ld_rows, n_seeds = [], [], []
        for base_name in bases:
            med_row, ld_row = {"Method": base_name}, {"Method": base_name}
            for task in tasks:
                seeds = sorted(s for (t, n, s) in grids if t == task and n == base_name
                               and (task, base_name + SUFFIX, s) in grids)
                if not seeds:
                    continue
                med = [get(grids[(task, base_name, s)]) for s in seeds]
                ld = [get(grids[(task, base_name + SUFFIX, s)]) for s in seeds]
                med_row[task], ld_row[task] = float(np.mean(med)), float(np.mean(ld))
                cells[(base_name, task)] = [b - a for a, b in zip(med, ld)]
                n_seeds.append((base_name, task, len(seeds)))
            med_rows.append(med_row)
            ld_rows.append(ld_row)
        med_df = pd.DataFrame(med_rows).set_index("Method")
        ld_df = pd.DataFrame(ld_rows).set_index("Method")
        cols = [t for t in tasks if t in med_df.columns and t in ld_df.columns]
        delta = ld_df[cols] - med_df[cols]
        delta["MEAN_d"] = delta[cols].mean(axis=1)
        delta["mean_median"] = med_df[cols].mean(axis=1)
        delta["mean_logdet"] = ld_df[cols].mean(axis=1)

        print(f"\n=== {label} ===")
        print("  cells are log-det-tuned minus median-heuristic, averaged over the cached seeds;\n"
              "  mean_* are the task-averaged levels\n")
        print(delta.to_string(float_format=lambda v: f"{v:+.4f}", na_rep="  --  "))
        arr = delta[cols].to_numpy(dtype=float)
        n = int(np.isfinite(arr).sum())
        print(f"\n  log-det better in {int(np.nansum(arr > 0))}/{n} (task, method) cells; "
              f"mean delta {np.nanmean(arr):+.4f}")

        multi = {k: v for k, v in cells.items() if len(v) > 1}
        if multi:
            print("\n  per-seed deltas where more than one seed is cached:")
            for (base_name, task), vals in sorted(multi.items()):
                print(f"    {base_name:<14} {task:<11} " + "  ".join(f"{v:+.4f}" for v in vals)
                      + f"   (spread {max(vals) - min(vals):.4f})")
    return 0


def run_sweep(task: str, base_name: str, seed: int, force: bool, plot: bool = True) -> int:
    """Score the whole bandwidth axis end-to-end, and place both candidates on it.

    All 15 arms -- the 13 fixed multipliers, the median-heuristic method itself and the
    log-det-tuned one -- go through a single ``_evaluate_task`` call, so the processed
    dataset is built once and every arm sees byte-identical data. Only ``sigma_o`` is swept:
    for an action-channel method the axis is a plane and 169 end-to-end evaluations is a
    different experiment.
    """
    path = os.path.join(CACHE_DIR, f"{task}__{base_name}__sweep__seed{seed}.csv")
    if os.path.exists(path) and not force:
        df = pd.read_csv(path)
        mult_logdet = float(df["mult_logdet"].iloc[0]) if "mult_logdet" in df else None
    else:
        make_variant(base_name)
        arms = [(m, sweep_name(base_name, m)) for m in SWEEP_MULTS]
        for mult, _ in arms:
            make_sweep_variant(base_name, mult)
        names = [n for _, n in arms] + [base_name, base_name + SUFFIX]
        _install_impl_capture()
        print(f"\nsweeping sigma_o over {len(SWEEP_MULTS)} multipliers on {task} "
              f"({len(names)} arms in one dataset build) ...")
        results = run_mod._evaluate_task(names, task, seed, {})
        impls = dict(_LAST_IMPLS)

        rows = []
        for mult, n in arms:
            h = headline(grid(n, task, results))
            rows.append(dict(mult=mult, TWA=float(h["TWA"]), Accuracy=float(h["Accuracy"]),
                             det_time=float(h["Det. Time"]), TWA_gridmean=float(h["TWA_gridmean"]),
                             sigma_o=float(impls[n]._sigma_o)))
        df = pd.DataFrame(rows).sort_values("mult")
        mult_logdet = float(impls[base_name + SUFFIX]._logdet_result.mults[0])
        # Carried in the CSV, not just returned: the plot needs it on a cached re-run and
        # re-deriving it would mean refitting the method.
        df["mult_logdet"] = mult_logdet
        df.to_csv(path, index=False)
        # Both reference arms are in the same run, so their rows are exact rather than
        # interpolated off the swept grid.
        for label, n in (("median heuristic (1.00x)", base_name), (f"log-det ({mult_logdet:.2f}x)",
                                                                   base_name + SUFFIX)):
            h = headline(grid(n, task, results))
            print(f"  {label:<28} TWA {float(h['TWA']):.4f}   grid-mean {float(h['TWA_gridmean']):.4f}")

    print(f"\n=== TWA vs sigma_o, {base_name} on {task} (seed {seed}) ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best = df.loc[df["TWA"].idxmax()]
    at_one = df.loc[(df["mult"] - 1.0).abs().idxmin()]
    print(f"\n  best swept multiplier {best['mult']:.3g}x: TWA {best['TWA']:.4f}   "
          f"(at the median heuristic: {at_one['TWA']:.4f}; headroom {best['TWA'] - at_one['TWA']:+.4f})")
    if plot:
        plot_sweep(df, task, base_name, mult_logdet,
                   str(HERE / f"{task}__{base_name}__sweep.png"))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="logdet_kernel", description=__doc__)
    parser.add_argument("--task", default="push_chair", choices=run_mod.ALL_TASKS)
    parser.add_argument("--methods", nargs="+", default=["kern_cd_obs"],
                        help="base method name(s); each is run alongside its _logdet twin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="ignore the cache")
    parser.add_argument("--check-only", action="store_true", help="verify the criterion's algebra and exit")
    parser.add_argument("--summary", action="store_true", help="roll up every cached grid and exit")
    parser.add_argument("--all-seeds", action="store_true",
                        help="--summary: average over every cached seed instead of only --seed")
    parser.add_argument("--grid-n", type=int, default=None, help="grid points per bandwidth")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-ridge-scan", action="store_true",
                        help="skip the ridge / tau sensitivity tables")
    parser.add_argument("--sweep", action="store_true",
                        help="score the whole sigma_o axis end-to-end instead (the control arm)")
    args = parser.parse_args(argv)

    if args.summary:
        return summarise(None if args.all_seeds else args.seed)

    check_criterion()
    if args.check_only:
        return 0

    discover()
    unknown = [m for m in args.methods if m not in REGISTRY]
    if unknown:
        print(f"unknown method(s) {unknown}; registered: {sorted(REGISTRY)}", file=sys.stderr)
        return 1

    os.makedirs(CACHE_DIR, exist_ok=True)
    if args.sweep:
        for base_name in args.methods:
            run_sweep(args.task, base_name, args.seed, args.force, plot=not args.no_plot)
        return 0

    for base_name in args.methods:
        cls = make_variant(base_name)
        if args.grid_n:
            cls.logdet_grid_n = args.grid_n

    names = [n for b in args.methods for n in (b, b + SUFFIX)]
    cache = {n: os.path.join(CACHE_DIR, f"{args.task}__{n}__seed{args.seed}.csv") for n in names}

    grids: dict[str, pd.DataFrame] = {}
    impls: dict = {}
    if args.force or not all(os.path.exists(p) for p in cache.values()):
        _install_impl_capture()
        _keep_tuning_inputs()
        print(f"\nevaluating {names} on {args.task} (seed {args.seed}) ...")
        results = run_mod._evaluate_task(names, args.task, args.seed, {})
        impls = dict(_LAST_IMPLS)
        for n in names:
            grids[n] = grid(n, args.task, results)
            grids[n].to_csv(cache[n], index=False)
        print(f"cached {len(names)} grids -> {os.path.relpath(CACHE_DIR, ROOT_DIR)}/")
    else:
        for n in names:
            grids[n] = pd.read_csv(cache[n])
        print(f"\nloaded cached grids for {names} on {args.task} "
              f"(--force to re-run; the tuning diagnostics need a re-run)")

    if impls:
        report_tuning(args.task, impls)
        report_loo_shift(args.task, impls)
        if not args.no_ridge_scan:
            report_ridge_sensitivity(impls)
            report_tau_robustness(impls)
        for base_name in args.methods:
            res = getattr(impls.get(base_name + SUFFIX), "_logdet_result", None)
            if res is None:
                continue
            res.scan.to_csv(os.path.join(CACHE_DIR, f"{args.task}__{base_name}__scan__seed{args.seed}.csv"),
                            index=False)
            if not args.no_plot:
                plot_landscape(res, args.task, base_name,
                               str(HERE / f"{args.task}__{base_name}__landscape.png"))

    print(f"\n=== headline metrics, {args.task} (seed {args.seed}) ===")
    print("best (window, threshold style) by quantile-averaged TWA, as in paper_table.py\n")
    table = pd.DataFrame({n: headline(grids[n]) for n in names}).T
    table.index.name = "Method"
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n--- log-det-tuned minus median-heuristic ---")
    for base_name in args.methods:
        a, b = headline(grids[base_name]), headline(grids[base_name + SUFFIX])
        deltas = {k: float(b[k]) - float(a[k]) for k in ("TWA", "Accuracy", "Det. Time", "TWA_gridmax", "TWA_gridmean")}
        print(f"  {base_name:<16} " + "  ".join(f"{k} {v:+.4f}" for k, v in deltas.items()))

    print("\n--- paired by (threshold, quantile, window): TWA delta over the whole grid ---")
    keys = ["Threshold", "Quantile", "Window"]
    for base_name in args.methods:
        merged = grids[base_name].merge(grids[base_name + SUFFIX], on=keys, suffixes=("_md", "_ld"))
        d = merged["TWA_ld"] - merged["TWA_md"]
        print(f"  {base_name:<16} n={len(d)}  mean {d.mean():+.4f}  median {d.median():+.4f}  "
              f"win {np.mean(d > 0):.0%} / tie {np.mean(d == 0):.0%} / loss {np.mean(d < 0):.0%}  "
              f"[{d.min():+.4f}, {d.max():+.4f}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
