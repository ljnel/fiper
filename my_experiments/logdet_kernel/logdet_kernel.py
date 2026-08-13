"""Regularized log-determinant kernel objective -- ``specs/experiments/logdet_kernel_objective.md``.

``specs/methods.md`` sets every kernel bandwidth by the median heuristic, each one
independently. The alternative studied here scores a whole bandwidth vector at once,

    J(theta) = (1/n) logdet(K_theta + lam*n*I) + beta * log(1 - CVaR_alpha(q(theta))),

with ``q_i(theta)`` the leave-one-out kern_cd score of fit point ``i``. The first term
rewards a kernel whose Gram matrix has volume -- i.e. one that keeps the fit points apart
-- and the second penalizes a kernel under which the worst ``alpha`` fraction of the fit
set already looks novel. The claim being tested is that the two together balance
``sigma_o`` against ``sigma_A``, which the median heuristic cannot do because it never
looks at the two kernels at the same time.

How this is measured
--------------------
Both halves of the question live on one grid: ``sigma_o`` and ``sigma_A`` each swept over
2^-4 .. 2^4 times their median heuristic in half-octaves (17 x 17 = 289 cells,
``sigma_phi`` left at the median heuristic).

* The **criterion** is evaluated on every cell analytically, and cheaply. The packed
  feature block is ``[o/sigma_o | mu_A/sigma_A]``, so a cell's kernel is
  ``exp(-1/2 (D_o/sigma_o^2 + D_A/sigma_A^2))`` for two squared-distance matrices that do
  not depend on the bandwidths at all: build those once, and a cell costs one Cholesky.
  ``logdet`` comes off that factor's diagonal and the LOO scores off its inverse (the
  identity in ``kern_cd_core._loo_scores``), so no scoring pass and no solver is involved.
  ``tau`` and ``CVaR_alpha`` are read off the sorted scores. ``q_i`` is ``kern_cd``'s own
  leave-one-*step*-out score by default; ``--loo episode`` holds out the whole episode
  instead, from the same factor (:func:`_block_loo`).
* The **headline TWA** is evaluated on every cell too, as a full FIPER evaluation -- the
  same path ``my_methods/run.py`` takes, through a variant class that scales each
  bandwidth by its multiplier. On pretzel that is ~0.6 s per cell.
* The **mean off-diagonal value of each kernel factor** rides along, from the same two
  distance matrices. It is what separates "the criterion balanced the two bandwidths" from
  "the criterion switched one of them off" -- see :func:`factor_grid`.

The one substantive claim this makes about ``kern_cd`` -- that a cell of
:meth:`Calibration.cell` is the kernel the method would have fitted -- is checked against
the method rather than assumed (``--check-only``; agreement to 1e-13 on pretzel).

Sweeping ``alpha`` and ``beta`` is then free: the criterion decomposes into a
``beta``-independent ``logdet`` and one ``CVaR`` per ``alpha``, and the TWA of whatever
cell a given ``(alpha, beta)`` selects is already in the landscape. So the study is not
"tune, then evaluate" but "evaluate everything once, then ask what each rule would have
picked" -- which also gives the oracle (the grid's best cell) to place both rules against.

Scope, and what it costs
------------------------
``kern_cd_flat``, seed 0, all five tasks: 5 x 289 = 1445 end-to-end FIPER evaluations
(about 100 minutes on an RTX 5090, dominated by stacking at ~11 s a cell) against 1445
analytic criterion evaluations (seconds). The study started on pretzel alone, as the spec
asks, and widened because pretzel turned out to be the wrong place to ask the question:
its whole TWA landscape spans 0.716-0.758 and its balanced accuracy is 0.900 in every one
of the 289 cells, so no tuning rule can be told from any other there.

What it found
-------------
**1. At the setting the spec starts from, the criterion does not improve the headline.**
Headline TWA against the median heuristic, at ``alpha = 0.05``, ``beta = 1``:

    pretzel -0.004   push_t -0.008   push_chair -0.049   sorting +0.024   stacking -0.016

Four of five tasks lose; the mean is -0.011 TWA. It is not that the bandwidths do not
matter -- the best cell of the same grid beats the median heuristic by +0.005, +0.021,
+0.081, +0.062 and +0.024 -- so there is real headroom on every task and the criterion is
not finding it. Nor is it that the two knobs were set badly: the best ``(alpha, beta)`` in
the whole 7 x 9 sweep, *chosen by the test-set TWA it produces*, recovers a little over
half of that headroom (+0.001, +0.016, +0.028, +0.042, +0.019 against +0.005, +0.021,
+0.081, +0.062, +0.024) -- and that rule cannot be run in practice, so it is a ceiling, not
a result.

**2. ``beta`` is an exchange rate between two quantities that have no common scale, and it
is not the same exchange rate twice.** The ``beta`` at which the two terms are of equal
median size over the grid is

    push_chair 0.10,   push_t 2.3,   sorting 8.2,   stacking 10.7,   pretzel 19.8,

a 200x spread. At ``beta = 1`` the barrier is under a tenth of the log-determinant on 25%
to 55% of the grid, depending on the task -- so on most tasks ``beta = 1`` is not a balance
between the two terms at all, it is the log-determinant with a rounding error attached, and
the picks show it: at ``beta = 1`` the criterion moves ``sigma_o`` to 0.25-0.5x the median
heuristic on four of five tasks, because a narrower kernel is a larger determinant.

**3. Where the criterion does win, it wins by deleting the action channel.** On sorting,
its +0.024, it picks ``sigma_A`` = 16x the median heuristic, at which the mean off-diagonal
action factor is 0.998 -- the product kernel's action factor is uniformly 1, i.e. the method
has reverted to ``kern_cd_obs``. Pretzel's pick is the same degeneracy (``sigma_A`` 16x,
factor 0.998) and merely does not pay off. This is the mechanism ``hyperparam_sensitivity``
documents from the other direction, and it is the opposite of the thing being tested: the
criterion was supposed to *balance* the observation and action bandwidths.

**4. There is no ``(alpha, beta)`` that is right on more than one task.** Over the 289
cells, the Spearman correlation between ``J(theta)`` and headline TWA at the headline
setting is -0.14, +0.08, -0.10, -0.41, +0.20 -- i.e. no relationship, and negative more
often than not. Each task does have a region of the ``(alpha, beta)`` box where the
criterion correlates decently (best +0.20, +0.50, +0.51, +0.41, +0.73), but the regions are
in opposite corners: push_t wants small ``alpha`` and large ``beta``, push_chair, sorting
and stacking want the reverse. Since the sign of the correlation flips across tasks at
every fixed setting, this is not a knob that needs tuning; it is a criterion whose ordering
of kernels has no stable relation to the benchmark's.

**5. Reusing ``tau`` as the threshold is not a separate idea.** ``tau`` is by construction
the ``(1-alpha)``-quantile of the LOO scores, and FIPER's conformal thresholds are quantiles
of exactly those scores, so "use ``tau``" means "keep the ``1-alpha`` column of FIPER's
quantile grid instead of averaging over all ten". It helps both rules a little on the three
tasks where it does anything (the median heuristic gains +0.032, +0.014, +0.006 on
push_chair, sorting and stacking, and +-0.000 on the other two), and it does not change
which rule wins: the criterion still trails the median heuristic on the same four tasks and
leads on sorting, by margins within 0.02 of the ones above.

**6. None of this is an artefact of leaving out one step rather than one episode.** The
default ``q_i`` is ``kern_cd``'s own, i.e. leave-one-*step*-out, which is the weaker
definition: the fit set is pooled steps, so holding out one step leaves its temporal
neighbours in and they re-interpolate it. ``--loo episode`` holds out the whole episode
instead (closed form in :func:`_block_loo`, checked against refitting with the episode
deleted). The scores do inflate -- the median CVaR over the grid goes 0.19 -> 0.73 on
stacking and 0.65 -> 0.85 on push_t, and barely moves on pretzel and push_chair -- but the
criterion picks the *same cell* on three of five tasks, the headline goes from -0.011 to
-0.008 TWA on average and stays negative on four of five, and the ``beta`` that balances
the two terms still spans 0.085 to 18.4 across tasks. Point 3 strengthens: under
episode-LOO, stacking's pick moves to ``sigma_A`` = 16x as well, so three of five picks now
have the action channel switched off. One caveat in the other direction: ``tau`` under
episode-LOO is no longer a quantile of the scores FIPER thresholds on, so the "reuse
``tau`` as the threshold" property of the spec does not survive this change.

**What this does not settle.** One seed and one of the four action-channel kernels;
``sigma_phi`` is left at the median heuristic throughout, so the criterion was never asked
to balance all three bandwidths at once. The grid is half-octaves over +-4 octaves, so a
pick is resolved to within a factor of 1.4. push_chair's fit set is 49 steps, which is few
enough that its LOO scores -- and therefore its CVaR -- are themselves noisy. Points 2 and 3
are structural and would survive all of that; point 1 is five numbers at one seed.

Running it
----------
    pixi run python my_experiments/logdet_kernel/logdet_kernel.py                 # pretzel, kern_cd_flat
    pixi run python my_experiments/logdet_kernel/logdet_kernel.py --check-only    # verify the machinery
    pixi run python my_experiments/logdet_kernel/logdet_kernel.py --task push_t
    pixi run python my_experiments/logdet_kernel/logdet_kernel.py --report        # cached cells only
    # the whole study, from the cache:
    pixi run python my_experiments/logdet_kernel/logdet_kernel.py \
        --task pretzel push_t push_chair sorting stacking --report
    # the same, with the LOO score taken over whole episodes (finding 6):
    pixi run python my_experiments/logdet_kernel/logdet_kernel.py \
        --task pretzel push_t push_chair sorting stacking --loo episode --report

Per-cell metric grids and criterion grids are cached under
``my_experiments/logdet_kernel/cache/``; ``--force`` re-runs them. The two LOO definitions
cache side by side (``--loo episode`` output carries a ``__loo_episode`` tag) and share the
same TWA landscape, which does not depend on how ``q_i`` is defined. Nothing in
``my_methods/`` is modified, but FIPER's own eval class writes a small pickle per cell
into ``data/<task>/results/<cell>/``; every name this file creates starts with ``ldk_``,
so the litter is removable with ``rm -rf data/*/results/ldk_*``.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import textwrap
import time
from dataclasses import dataclass
from types import SimpleNamespace

HERE = pathlib.Path(__file__).resolve().parent
ROOT_DIR = str(HERE.parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402
from sklearn.metrics.pairwise import euclidean_distances  # noqa: E402

from my_methods import run as run_mod  # noqa: E402
from my_methods.base import REGISTRY, build_cfg, discover  # noqa: E402

CACHE_DIR = HERE / "cache"

#: Half-octaves over +-4 octaves: a 256x range on each bandwidth, wide enough to bracket
#: both degenerate regimes (kernel ~ I at the bottom, kernel ~ all-ones at the top).
EXPONENTS = tuple(np.arange(-4.0, 4.5, 0.5))
MULTS = tuple(2.0**e for e in EXPONENTS)

#: Tail fractions. 1.0 is the mean -- CVaR with no tail selection at all -- and is kept as
#: the degenerate end of the sweep.
ALPHAS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)

#: Weights on the CVaR barrier. beta = 1 is the spec's starting point; the sweep is
#: geometric because the two terms are both logs, so what matters is their ratio.
BETAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)

#: The tail fraction the beta sweep and the headline rows run at, fixed in advance. It has
#: to be fixed in advance: picking it by the TWA it produces would be tuning the criterion
#: on the test set, and the sweep tables would then have nothing left to say. 0.05 is the
#: natural choice -- a 5% tail, and ``tau`` then lands exactly on FIPER's 0.95 quantile.
ALPHA_DEFAULT = 0.05

#: Cells evaluated per ``_evaluate_task`` call (the processed dataset is rebuilt per call,
#: and a group is the unit of crash recovery, so this is a rebuild-time / blast-radius
#: trade). Also caps how many m x m Cholesky factors are alive at once.
GROUP = 16

#: Defaults from ``specs/methods.md``.
LAM, N_COMPONENTS = 1.0e-5, 128


# ------------------------------------------------------------------- the cheap criterion
@dataclass
class Calibration:
    """Everything about a fit set that does not depend on the bandwidths being swept."""

    task: str
    method: str
    seed: int
    m: int
    sigma_o: float  # median heuristic
    sigma_phi: float
    sigma_A: float
    D_o: np.ndarray  # [m, m] squared distances between observation embeddings
    D_A: np.ndarray  # [m, m] squared distances between mean embeddings mu_A
    episode: np.ndarray  # [m] index of the fit episode each row came from

    def cell(self, mult_o: float, mult_A: float, blocked: bool = False) -> tuple[float, np.ndarray]:
        """``((1/m) logdet(K + lam*m*I), LOO scores)`` at one point of the grid.

        The product kernel is a single RBF on the packed block ``[o/sigma_o | mu/sigma_A]``
        (see ``kern_cd_core``), so its exponent is a bandwidth-weighted sum of the two
        precomputed distance matrices and the whole cell costs one Cholesky.

        ``blocked`` switches from leave-one-*step*-out to leave-one-*episode*-out; see
        :func:`_block_loo` for why the distinction matters and what it costs.
        """
        K = np.exp(-0.5 * (self.D_o / (mult_o * self.sigma_o) ** 2 + self.D_A / (mult_A * self.sigma_A) ** 2))
        lam_m = LAM * self.m
        L = np.linalg.cholesky(K + lam_m * np.eye(self.m))
        logdet = 2.0 * float(np.sum(np.log(np.diag(L)))) / self.m
        L_inv = solve_triangular(L, np.eye(self.m), lower=True)
        if blocked:
            return logdet, _block_loo(L_inv, self.episode, lam_m)
        return logdet, 1.0 / np.einsum("ki,ki->i", L_inv, L_inv) - lam_m


def _block_loo(L_inv: np.ndarray, episode: np.ndarray, lam_m: float) -> np.ndarray:
    """Leave-one-*episode*-out support scores, from the same Cholesky factor.

    The fit set is pooled steps, and consecutive steps of one episode are nearly duplicates
    of each other. Holding out a single step therefore leaves its own neighbours in the fit
    set, which re-interpolate it almost perfectly: the step-wise score says "this point is
    not novel" when all it has established is "this point is not novel *given the rest of
    its own episode*". Holding out the whole episode is the honest version, and it is what
    the CVaR term needs, since that term is supposed to measure how novel the fit set
    already looks.

    With ``A = (K + lam*m*I)^-1 = (L L^T)^-1``, the block analogue of ``1/A_ii`` is
    ``diag(inv(A_BB))``: the posterior covariance of block ``B`` with ``B`` removed is the
    inverse of ``A`` restricted to ``B``. ``A_BB = (L^-1 e_B)^T (L^-1 e_B)`` is read straight
    off the columns of ``L^-1``, so no full inverse is formed and the cost is
    ``m * sum_B |B|^2`` -- a few hundred megaflops even at stacking's m = 2760.
    :func:`check_block_loo` verifies it against refitting with the episode deleted.
    """
    loo = np.empty(L_inv.shape[1])
    for b in np.unique(episode):
        idx = np.flatnonzero(episode == b)
        block = L_inv[:, idx]
        loo[idx] = np.diag(np.linalg.inv(block.T @ block)) - lam_m
    return loo


def prepare(task: str, method: str, seed: int, mult_phi: float = 1.0) -> Calibration:
    """Build the fit set's distance matrices and median-heuristic bandwidths.

    This walks the method's own feature pipeline -- the mask that drops push_t's failed
    calibration episodes, ``_obs``, ``_parts``, the RFF draw, ``_embed`` -- and stops just
    before the bandwidths would be applied, so nothing here reimplements the method.
    """
    import torch

    from shared_utils.hydra_utils import load_config
    from tasks import TaskManager

    discover()
    cls = REGISTRY[method]
    cfg = load_config("task", task, return_only_subdict=False)
    device = run_mod._device()
    tm = TaskManager(
        cfg,
        task,
        run_mod.BASE_CONFIG_PATH,
        os.path.join(run_mod.BASE_DATA_PATH, task),
        required_tensors=list(cls.tensors),
        optional_tensors=list(cls.optional_tensors),
        device=device,
    )
    dataset = tm.get_rollout_dataset(load_dataset_if_exists=False)
    eval_cfg = build_cfg(cls, run_mod.BASE_CONFIG_PATH)
    calib = dataset.get_subset(
        subset="calibration",
        required_tensors=list(cls.tensors),
        optional_tensors=list(cls.optional_tensors),
        required_actions=list(cls.actions),
        optional_actions=list(cls.optional_actions),
        normalize_tensors=eval_cfg.normalize_tensors,
        with_success_labels=True,
        history=0,
        return_episode_lengths=True,
    )

    impl = cls()
    impl.p = SimpleNamespace(lam=LAM, n_components=N_COMPONENTS, seed=seed)
    impl.dataset, impl.device = dataset, device
    impl._rng = np.random.default_rng(seed)

    lengths = np.asarray(calib["episode_lengths"], int)
    successful = np.asarray(calib["successful"], bool)
    keep = np.repeat(successful, lengths)
    # Episode id per surviving fit row -- the block structure the LOO score can be taken
    # over. push_t's failed calibration episodes are dropped by `keep`, exactly as in fit().
    episode = np.repeat(np.arange(len(lengths)), lengths)[keep]
    obs = impl._obs(calib["obs_embeddings"])[keep]
    sigma_o = impl._median(obs)
    chunks = impl._chunks(calib["action_preds"])[torch.as_tensor(keep, device=device)]
    parts = impl._fit_parts(chunks)
    impl._fit_phi(parts)
    if mult_phi != 1.0:  # same draw at a different bandwidth; see hyperparam_sensitivity
        impl._sigma_phi *= mult_phi
        impl._phi_W = impl._phi_W / mult_phi
    mu = impl._embed(parts)
    return Calibration(
        task=task,
        method=method,
        seed=seed,
        m=len(obs),
        sigma_o=sigma_o,
        sigma_phi=float(impl._sigma_phi),
        sigma_A=impl._median(mu),
        D_o=euclidean_distances(obs, obs, squared=True),
        D_A=euclidean_distances(mu, mu, squared=True),
        episode=episode,
    )


def cvar(scores: np.ndarray, alpha: float) -> tuple[float, float]:
    """``(CVaR_alpha, tau)`` of the LOO scores, by sorting rather than by a solver.

    The Rockafellar-Uryasev objective ``tau + (1/(alpha n)) sum (q_i - tau)_+`` is convex
    and piecewise linear in ``tau`` with kinks at the ``q_i``, so its minimum sits at the
    ``(1-alpha)``-quantile and its value is the mean of the worst ``alpha n`` scores --
    with the boundary score weighted fractionally when ``alpha n`` is not an integer.
    :func:`check_cvar` verifies both statements against a brute-force minimisation.
    """
    s = np.sort(scores)[::-1]
    n = len(s)
    k = int(np.floor(alpha * n))  # scores fully inside the tail
    if k >= n:
        return float(s.mean()), float(s[-1])
    total = s[:k].sum() + (alpha * n - k) * s[k]
    return float(total / (alpha * n)), float(s[k])


def criterion_grid(cal: Calibration, alphas=ALPHAS, blocked: bool = False) -> pd.DataFrame:
    """One row per grid cell: the logdet term, and ``(tau, CVaR)`` for every alpha."""
    rows = []
    for eo, mo in zip(EXPONENTS, MULTS):
        for ea, ma in zip(EXPONENTS, MULTS):
            logdet, loo = cal.cell(mo, ma, blocked=blocked)
            row = {"exp_o": eo, "exp_A": ea, "mult_o": mo, "mult_A": ma, "logdet": logdet,
                   "loo_mean": float(loo.mean()), "loo_max": float(loo.max())}
            for a in alphas:
                c, t = cvar(loo, a)
                row[f"cvar@{a:g}"], row[f"tau@{a:g}"] = c, t
            rows.append(row)
    return pd.DataFrame(rows)


def factor_grid(cal: Calibration) -> pd.DataFrame:
    """The mean off-diagonal value of each kernel factor, per cell.

    How much work a factor is doing, in one number, exactly as
    ``hyperparam_sensitivity._mean_factor`` measures it. The kernel is a product, so a
    factor averaging ~1 has been switched off: ``k = k_o * 1`` is the observation-only
    baseline. That distinction is the whole difference between "the criterion balanced the
    two bandwidths" and "the criterion deleted one of them", and TWA cannot tell them apart.
    """
    rows = []
    for eo, mo in zip(EXPONENTS, MULTS):
        for ea, ma in zip(EXPONENTS, MULTS):
            rows.append({"exp_o": eo, "exp_A": ea,
                         "k_o": _mean_off_diag(cal.D_o, mo * cal.sigma_o),
                         "k_A": _mean_off_diag(cal.D_A, ma * cal.sigma_A)})
    return pd.DataFrame(rows)


def _mean_off_diag(D: np.ndarray, sigma: float, cap: int = 800) -> float:
    """Mean off-diagonal ``exp(-D / 2 sigma^2)`` on (a capped corner of) the fit set."""
    A = D[:cap, :cap]
    n = len(A)
    K = np.exp(-A / (2.0 * sigma**2))
    return float((K.sum() - np.trace(K)) / (n * (n - 1)))


def objective(grid: pd.DataFrame, alpha: float, beta: float) -> np.ndarray:
    """``J(theta)`` over the whole grid, for one ``(alpha, beta)``.

    ``1 - CVaR`` is clipped away from zero rather than allowed to produce ``-inf``: a
    kernel whose worst tail scores are numerically 1 is maximally novelty-declaring, and
    the barrier is meant to reject it, not to poison the comparison with a NaN.
    """
    slack = np.clip(1.0 - grid[f"cvar@{alpha:g}"].to_numpy(dtype=float), 1e-12, None)
    return grid["logdet"].to_numpy(dtype=float) + beta * np.log(slack)


def select(grid: pd.DataFrame, alpha: float, beta: float) -> pd.Series:
    """The cell the criterion picks, with its criterion values attached."""
    obj = objective(grid, alpha, beta)
    row = grid.iloc[int(np.argmax(obj))].copy()
    row["alpha"], row["beta"] = alpha, beta
    row["objective"] = float(obj.max())
    row["logdet_term"] = float(row["logdet"])
    row["barrier_term"] = float(obj.max() - row["logdet"])
    return row


# ------------------------------------------------------------------------------- the checks
def check_cvar(seed: int = 0) -> None:
    """The sorting shortcut must reproduce the Rockafellar-Uryasev minimisation."""
    rng = np.random.default_rng(seed)
    for n in (97, 500):
        q = rng.beta(2.0, 5.0, size=n)
        for alpha in (0.03, 0.1, 0.37, 1.0):
            c, tau = cvar(q, alpha)
            taus = np.unique(np.concatenate([q, np.linspace(0, 1, 20001)]))
            ru = taus + np.maximum(q[None, :] - taus[:, None], 0).sum(axis=1) / (alpha * n)
            assert abs(ru.min() - c) < 1e-9, f"CVaR mismatch at n={n}, alpha={alpha}: {ru.min()} vs {c}"
            assert abs(taus[int(np.argmin(ru))] - tau) < 1e-3 or abs(ru[np.argmin(np.abs(taus - tau))] - c) < 1e-9, (
                f"tau is not the RU minimiser at n={n}, alpha={alpha}"
            )
    print("CVaR OK: sorting reproduces min_tau [tau + (1/alpha n) sum (q-tau)_+] and its argmin")


def check_grid(cal: Calibration) -> None:
    """The cheap grid's cell at 1x must reproduce an ordinary ``fit()`` of the method.

    Everything downstream rests on the claim that a cell of :meth:`Calibration.cell` is the
    kernel the method would actually have fitted, so that claim is checked against the
    method rather than assumed: refit ``kern_cd`` through its own code path at the median
    heuristic and compare the leave-one-out scores it produces.
    """
    logdet, loo = cal.cell(1.0, 1.0)
    err = float(np.abs(np.sort(loo) - np.sort(_refit_loo(cal))).max())
    assert err < 1e-6, f"cheap grid LOO scores differ from {cal.method}.fit by {err:.2e}"
    print(f"grid OK: LOO scores at 1x match {cal.method}.fit to {err:.1e}; (1/m) logdet = {logdet:.4f}")


def check_block_loo(cal: Calibration, max_m: int = 700) -> None:
    """Leave-one-episode-out, closed form, must equal refitting with the episode deleted."""
    if cal.m > max_m:
        print(f"block-LOO check skipped: m = {cal.m} > {max_m} (the brute force is O(m^3) per episode)")
        return
    mult_o, mult_A = 0.5, 2.0  # off the median heuristic, so no factor is accidentally 1
    _, fast = cal.cell(mult_o, mult_A, blocked=True)
    K = np.exp(-0.5 * (cal.D_o / (mult_o * cal.sigma_o) ** 2 + cal.D_A / (mult_A * cal.sigma_A) ** 2))
    lam_m = LAM * cal.m
    brute = np.empty(cal.m)
    for b in np.unique(cal.episode):
        idx = np.flatnonzero(cal.episode == b)
        out = np.flatnonzero(cal.episode != b)
        M = K[np.ix_(out, out)] + lam_m * np.eye(len(out))
        brute[idx] = np.diag(K[np.ix_(idx, idx)] - K[np.ix_(idx, out)] @ np.linalg.solve(M, K[np.ix_(out, idx)]))
    err = float(np.abs(fast - brute).max())
    assert err < 1e-6, f"block-LOO closed form differs from the brute-force refit by {err:.2e}"
    _, point = cal.cell(mult_o, mult_A)
    print(f"block-LOO OK: matches refitting with the episode deleted to {err:.1e}; "
          f"over {len(np.unique(cal.episode))} episodes the median score rises "
          f"{np.median(point):.4f} -> {np.median(fast):.4f}")


def _refit_loo(cal: Calibration) -> np.ndarray:
    """LOO scores from the method's own ``fit``, for :func:`check_grid`."""
    from shared_utils.hydra_utils import load_config
    from tasks import TaskManager

    cls = REGISTRY[cal.method]
    cfg = load_config("task", cal.task, return_only_subdict=False)
    device = run_mod._device()
    tm = TaskManager(
        cfg, cal.task, run_mod.BASE_CONFIG_PATH, os.path.join(run_mod.BASE_DATA_PATH, cal.task),
        required_tensors=list(cls.tensors), optional_tensors=list(cls.optional_tensors), device=device,
    )
    dataset = tm.get_rollout_dataset(load_dataset_if_exists=False)
    eval_cfg = build_cfg(cls, run_mod.BASE_CONFIG_PATH)
    calib = dataset.get_subset(
        subset="calibration", required_tensors=list(cls.tensors), optional_tensors=list(cls.optional_tensors),
        required_actions=list(cls.actions), optional_actions=list(cls.optional_actions),
        normalize_tensors=eval_cfg.normalize_tensors, with_success_labels=True, history=0,
        return_episode_lengths=True,
    )
    impl = cls()
    impl.p = SimpleNamespace(lam=LAM, n_components=N_COMPONENTS, seed=cal.seed)
    impl.dataset, impl.device = dataset, device
    impl.fit(calib)
    return impl.loo_scores


# --------------------------------------------------------------- the TWA landscape (arms)
class _ScaleMixin:
    """Scales sigma_o and sigma_A by fixed multiples of their median heuristic.

    ``_pack`` is the one place where both bandwidths are known and neither has been used
    yet; it is also called on every scoring pass, so the scaling is guarded to happen once.
    """

    ldk_mult_o = 1.0
    ldk_mult_A = 1.0

    def _pack(self, obs, mu):
        if not getattr(self, "_ldk_done", False):
            self._ldk_done = True
            self._sigma_o = float(self._sigma_o) * float(self.ldk_mult_o)
            if mu is not None:
                self._sigma_A = float(self._sigma_A) * float(self.ldk_mult_A)
        return super()._pack(obs, mu)


def arm_name(method: str, exp_o: float, exp_A: float) -> str:
    """Registry key and cache key, in octaves off the median heuristic."""
    return f"ldk_{method.removeprefix('kern_cd_')}__o{exp_o:+g}__A{exp_A:+g}"


def make_arm(method: str, exp_o: float, exp_A: float):
    """Register (once) the variant class for one grid cell."""
    name = arm_name(method, exp_o, exp_A)
    if name in REGISTRY:
        return REGISTRY[name]
    base = REGISTRY[method]
    return type(
        "Cell" + name.replace("ldk_", "").replace("__", "_").replace("+", "p").replace("-", "m").replace(".", "d"),
        (_ScaleMixin, base),
        {"name": name, "params": dict(base.params, lam=LAM, n_components=N_COMPONENTS),
         "ldk_mult_o": 2.0**exp_o, "ldk_mult_A": 2.0**exp_A},
    )


def headline(df: pd.DataFrame, quantile: float | None = None) -> pd.Series:
    """The headline metrics at ``paper_table.py``'s footing: quantile-averaged, best cell.

    ``TWA_gridmean`` -- the flat mean over all 510 (style, quantile, window) rows -- rides
    along because the headline is an argmax over 51 noisy numbers and can move on its own.

    Passing ``quantile`` restricts to one of FIPER's conformal quantiles instead of
    averaging over all ten. That is how the criterion's own ``tau`` enters: ``tau`` is by
    construction the ``(1-alpha)``-quantile of the LOO scores, and FIPER's thresholds are
    quantiles of those same scores, so "reuse ``tau`` as the threshold" *is* "keep only the
    ``1-alpha`` column of the quantile grid".
    """
    if quantile is not None:
        df = df[np.isclose(df["Quantile"], quantile)]
        if df.empty:
            return pd.Series(dict.fromkeys(
                ["TWA", "Accuracy", "Det. Time", "TWA_gridmean", "Window", "Threshold"], np.nan))
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


def _cache_path(task: str, name: str, seed: int) -> str:
    return str(CACHE_DIR / f"{task}__{name}__seed{seed}.csv")


def twa_landscape(task: str, method: str, seed: int, force: bool = False,
                  cache_only: bool = False, group: int = GROUP) -> dict:
    """Every cell's full (style, quantile, window) metric grid, evaluated end to end.

    The raw grids are kept rather than reduced on the spot: the headline reduction averages
    over the ten conformal quantiles, and the ``tau``-as-threshold variant needs one of them
    on its own.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    discover()
    cells = [(eo, ea) for eo in EXPONENTS for ea in EXPONENTS]
    have, todo = {}, []
    for eo, ea in cells:
        path = _cache_path(task, arm_name(method, eo, ea), seed)
        if os.path.exists(path) and not force:
            have[(eo, ea)] = pd.read_csv(path)
        else:
            todo.append((eo, ea))

    if todo and not cache_only:
        print(f"\nevaluating {len(todo)} cell(s) of the TWA landscape on {task} "
              f"({len(have)} cached) ...")
        started = time.time()
        for i in range(0, len(todo), group):
            batch = todo[i : i + group]
            if not _evaluate_group(task, method, seed, batch, have):
                for cell in batch:  # one bad cell should not cost the group
                    _evaluate_group(task, method, seed, [cell], have)
            print(f"  {min(i + group, len(todo))}/{len(todo)} cells, {time.time() - started:.0f}s elapsed")
    return have


def reduce_landscape(cells: dict, quantile: float | None = None) -> pd.DataFrame:
    """The headline metrics of every cell, as one row per grid point."""
    rows = []
    for eo in EXPONENTS:
        for ea in EXPONENTS:
            row = {"exp_o": eo, "exp_A": ea, "mult_o": 2.0**eo, "mult_A": 2.0**ea}
            df = cells.get((eo, ea))
            row.update(headline(df, quantile).to_dict() if df is not None
                       else dict.fromkeys(["TWA", "Accuracy", "Det. Time", "TWA_gridmean"], np.nan))
            rows.append(row)
    return pd.DataFrame(rows)


def _evaluate_group(task: str, method: str, seed: int, group: list, have: dict) -> bool:
    names = []
    for eo, ea in group:
        make_arm(method, eo, ea)
        names.append(arm_name(method, eo, ea))
    try:
        results = run_mod._evaluate_task(names, task, seed, {})
    except Exception as exc:  # noqa: BLE001 -- a cell that cannot be evaluated is a result
        print(f"  !! {names if len(names) > 1 else names[0]} failed: "
              f"{type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
        return False
    for (eo, ea), name in zip(group, names):
        df = pd.DataFrame(run_mod._rows(name, task, results))
        df.to_csv(_cache_path(task, name, seed), index=False)
        have[(eo, ea)] = df
    return True


# ------------------------------------------------------------------------------ reporting
def merge(crit: pd.DataFrame, twa: pd.DataFrame) -> pd.DataFrame:
    """Criterion values and headline metrics on one row per cell."""
    return crit.merge(twa.drop(columns=["mult_o", "mult_A"]), on=["exp_o", "exp_A"], how="left")


def _cell(df: pd.DataFrame, exp_o: float, exp_A: float) -> pd.Series:
    return df[(df["exp_o"] == exp_o) & (df["exp_A"] == exp_A)].iloc[0]


def sweep(df: pd.DataFrame, alphas=ALPHAS, betas=BETAS) -> pd.DataFrame:
    """What each ``(alpha, beta)`` picks, and what that pick scores."""
    rows = []
    for a in alphas:
        for b in betas:
            pick = select(df, a, b)
            rows.append({
                "alpha": a, "beta": b,
                "exp_o": pick["exp_o"], "exp_A": pick["exp_A"],
                "mult_o": pick["mult_o"], "mult_A": pick["mult_A"],
                "logdet": pick["logdet"], "cvar": pick[f"cvar@{a:g}"], "tau": pick[f"tau@{a:g}"],
                "k_o": pick.get("k_o", np.nan), "k_A": pick.get("k_A", np.nan),
                "TWA": pick.get("TWA", np.nan), "Accuracy": pick.get("Accuracy", np.nan),
                "Det. Time": pick.get("Det. Time", np.nan), "TWA_gridmean": pick.get("TWA_gridmean", np.nan),
            })
    return pd.DataFrame(rows)


def report_landscape(df: pd.DataFrame, cal: Calibration) -> dict:
    """Print the grid's extent, and return its two reference cells: median heuristic, best TWA."""
    med = _cell(df, 0.0, 0.0)
    best = df.loc[df["TWA"].idxmax()]
    print(f"\n=== {cal.method} on {cal.task}, seed {cal.seed}: the grid ===")
    print(f"  fit set m = {cal.m};  median heuristic  sigma_o {cal.sigma_o:.4g}  "
          f"sigma_phi {cal.sigma_phi:.4g}  sigma_A {cal.sigma_A:.4g}")
    print(f"  grid: {len(EXPONENTS)} x {len(EXPONENTS)} = {len(df)} cells, "
          f"2^{EXPONENTS[0]:g} .. 2^{EXPONENTS[-1]:g} x the median heuristic in half-octaves")
    print(f"  (1/m) logdet term over the grid: {df['logdet'].min():.3f} .. {df['logdet'].max():.3f}")
    print(f"  headline TWA over the grid:      {df['TWA'].min():.3f} .. {df['TWA'].max():.3f}   "
          f"(median {df['TWA'].median():.3f})")
    print(f"  median heuristic cell (1x, 1x):  TWA {med['TWA']:.3f}   Accuracy {med['Accuracy']:.3f}   "
          f"Det. Time {med['Det. Time']:.3f}")
    print(f"  best cell in the grid:           TWA {best['TWA']:.3f}   at sigma_o {best['mult_o']:g}x, "
          f"sigma_A {best['mult_A']:g}x")
    return {"median": med, "best": best}


def report_alpha(df: pd.DataFrame, med: pd.Series) -> pd.DataFrame:
    """The alpha sweep at beta = 1, as the spec asks for it first."""
    out = sweep(df, betas=(1.0,))
    print("\n=== alpha sweep at beta = 1 ===")
    print("  (what the criterion picks, and what that pick actually scores; k_A is the mean "
          "off-diagonal\n   action factor, so k_A ~ 1 means the action channel is inert)")
    show = out[["alpha", "mult_o", "mult_A", "logdet", "cvar", "tau", "k_A", "TWA", "Accuracy", "Det. Time"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}", na_rep="  --  "))
    print(f"  median heuristic, for reference: mult_o 1, mult_A 1, TWA {med['TWA']:.4f}")
    return out


def report_beta(df: pd.DataFrame, med: pd.Series, alpha: float) -> pd.DataFrame:
    """The beta sweep, at the tail fraction fixed in advance (see ``ALPHA_DEFAULT``)."""
    out = sweep(df, alphas=(alpha,))
    print(f"\n=== beta sweep at alpha = {alpha:g} ===")
    show = out[["beta", "mult_o", "mult_A", "logdet", "cvar", "tau", "k_A", "TWA", "Accuracy", "Det. Time"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}", na_rep="  --  "))
    print(f"  median heuristic, for reference: mult_o 1, mult_A 1, TWA {med['TWA']:.4f}")
    return out


def report_balance(df: pd.DataFrame, alpha: float) -> dict:
    """How large the two terms are relative to each other, and at what beta they match.

    ``beta`` is the only thing setting the exchange rate between a log-determinant and a
    log-slack, and neither is scale-free: the logdet term carries the data's dimension and
    conditioning, the barrier carries how novel the fit set already looks under the median
    heuristic. So the ``beta`` at which the two terms are comparable is a property of the
    task, and if it moves across tasks then no single ``beta`` -- 1 included -- can be the
    right setting everywhere. That is a claim about the criterion, not about tuning it, so
    it is measured rather than inferred from the sweeps.
    """
    logdet = df["logdet"].to_numpy(float)
    barrier = np.log(np.clip(1.0 - df[f"cvar@{alpha:g}"].to_numpy(float), 1e-12, None))
    med_l, med_b = float(np.median(np.abs(logdet))), float(np.median(np.abs(barrier)))
    out = {
        "median |logdet term|": med_l,
        "median |barrier| at beta=1": med_b,
        "beta that makes them equal": med_l / med_b if med_b else np.inf,
        "cells where the barrier is <10% of the logdet at beta=1":
            float(np.mean(np.abs(barrier) < 0.1 * np.abs(logdet))),
    }
    print(f"\n=== how the two terms compare (alpha = {alpha:g}) ===")
    for k, v in out.items():
        print(f"  {k:<58} {v:.3f}")
    return out


def report_proxy(df: pd.DataFrame, alphas=ALPHAS, betas=BETAS) -> pd.DataFrame:
    """Is the criterion a useful proxy for TWA at all?

    A tuning criterion is only worth anything if cells it likes are cells that score well,
    so the rank correlation between ``J(theta)`` and TWA over the 289 cells is reported for
    every ``(alpha, beta)``. This is a much stronger statement than the argmax alone: the
    argmax is one cell out of 289 and can land well by luck.
    """
    from scipy.stats import spearmanr

    ok = df["TWA"].notna().to_numpy()
    rows = []
    for a in alphas:
        for b in betas:
            obj = objective(df, a, b)[ok]
            rho = float(spearmanr(obj, df["TWA"].to_numpy()[ok]).statistic)
            rows.append({"alpha": a, "beta": b, "spearman": rho})
    out = pd.DataFrame(rows)
    print("\n=== rank correlation between the criterion and headline TWA, over all cells ===")
    print(out.pivot(index="alpha", columns="beta", values="spearman").to_string(
        float_format=lambda v: f"{v:+.3f}"))
    best = out.loc[out["spearman"].idxmax()]
    print(f"  best: alpha {best['alpha']:g}, beta {best['beta']:g} -> rho {best['spearman']:+.3f};  "
          f"logdet alone -> rho {spearmanr(df['logdet'].to_numpy()[ok], df['TWA'].to_numpy()[ok]).statistic:+.3f}")
    return out


def report_tau(crit: pd.DataFrame, cells: dict, alpha: float, beta: float) -> pd.DataFrame | None:
    """Reuse ``tau`` as the classification threshold instead of averaging over quantiles.

    ``tau`` is the ``(1-alpha)``-quantile of the LOO scores and FIPER's conformal thresholds
    are quantiles of exactly those scores, so this is not a new threshold rule at all -- it
    is the instruction to keep the ``1-alpha`` column of the quantile grid rather than the
    mean over all ten. Whether it helps is then a question about the quantile grid, and the
    median heuristic is scored the same way so the two differ only in the bandwidths.
    """
    q = round(1.0 - alpha, 4)
    df_q = merge(crit, reduce_landscape(cells, quantile=q))
    if df_q["TWA"].isna().all():
        print(f"\n=== tau as the threshold: alpha = {alpha:g} needs quantile {q:g}, which is not "
              f"in FIPER's grid -- skipped ===")
        return None
    med_q, pick_q = _cell(df_q, 0.0, 0.0), select(df_q, alpha, beta)
    print(f"\n=== tau as the threshold (quantile {q:g} only, not the 10-quantile average) ===")
    out = pd.DataFrame([
        {"rule": "median heuristic", "mult_o": 1.0, "mult_A": 1.0, "TWA": med_q["TWA"],
         "Accuracy": med_q["Accuracy"], "Det. Time": med_q["Det. Time"]},
        {"rule": f"criterion (alpha={alpha:g}, beta={beta:g})", "mult_o": pick_q["mult_o"],
         "mult_A": pick_q["mult_A"], "TWA": pick_q["TWA"], "Accuracy": pick_q["Accuracy"],
         "Det. Time": pick_q["Det. Time"]},
    ])
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df_q


def summary_row(task: str, method: str, seed: int, kind: str, label: str, cell: pd.Series) -> dict:
    """One row of the summary table. ``kind`` is the machine-readable key the plots filter on."""
    return {
        "Task": task, "Method": method, "Seed": seed, "Kind": kind, "Rule": label,
        "sigma_o (x med)": float(cell["mult_o"]), "sigma_A (x med)": float(cell["mult_A"]),
        "k_A": float(cell.get("k_A", np.nan)),
        "TWA": float(cell["TWA"]), "Accuracy": float(cell["Accuracy"]),
        "Det. Time": float(cell["Det. Time"]), "TWA_gridmean": float(cell["TWA_gridmean"]),
    }


# ---------------------------------------------------------------------------------- plots
# Repo house style: near-white surface, near-black ink, muted labels. Magnitude gets a
# single-hue sequential ramp; a signed quantity gets a two-hue diverging ramp with a
# neutral midpoint.
_INK, _MUTED, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
# Three categorical hues, checked with the dataviz skill's validator against this
# surface: all-pairs CVD dE 9.2 (deutan) and normal-vision dE 23.8, both above floor.
_ACCENT, _SECOND, _THIRD = "#eb6834", "#2b6cb0", "#1baf7a"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=8)


def _save(fig, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{stem}.{ext}", dpi=150, facecolor=_SURFACE, bbox_inches="tight")
    print(f"  plot -> {os.path.relpath(HERE / (stem + '.png'), ROOT_DIR)} (+ .pdf)")


def _field(ax, df: pd.DataFrame, values: np.ndarray, cmap: str, label: str,
           floor_pct: float | None = None):
    """One (sigma_o, sigma_A) landscape as a mesh in log2 coordinates.

    ``floor_pct`` clips the bottom of the colour range at that percentile. The criterion
    needs it: its barrier term dives to -12 in the corner where every fit point looks
    novel, and a range stretched to that swamps the structure around the maximum, which is
    the only part of the surface an argmax rule ever visits.
    """
    piv = df.assign(_v=values).pivot(index="exp_A", columns="exp_o", values="_v")
    X, Y, Z = piv.columns.to_numpy(float), piv.index.to_numpy(float), piv.to_numpy(float)
    kw = {} if floor_pct is None else {"vmin": float(np.nanpercentile(Z, floor_pct))}
    mesh = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="nearest", **kw)
    bar = ax.figure.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
    bar.set_label(label, color=_MUTED, fontsize=8)
    bar.ax.tick_params(colors=_MUTED, labelsize=7)
    bar.outline.set_visible(False)
    ticks = [e for e in EXPONENTS if e == int(e)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{2.0**e:g}" for e in ticks])
    ax.set_yticklabels([f"{2.0**e:g}" for e in ticks])
    ax.set_xlabel(r"$\sigma_o$  ($\times$ median heuristic)", color=_MUTED, fontsize=9)
    ax.set_ylabel(r"$\sigma_A$  ($\times$ median heuristic)", color=_MUTED, fontsize=9)
    _style(ax)
    return mesh


def _marks(ax, picks: list[tuple[float, float, str, str, str]]):
    """Overlay the reference points every landscape shares."""
    for x, y, marker, color, label in picks:
        ax.plot(x, y, marker, color=color, markersize=11, markeredgewidth=2.0,
                markerfacecolor="none" if marker in ("o", "s") else color, label=label, zorder=5)


def plot_landscape(df: pd.DataFrame, cal: Calibration, alpha: float, beta: float, stem: str) -> None:
    """The criterion's landscape beside the TWA landscape it is standing in for.

    The whole question is whether the argmax of the left panel lands where the right panel
    is high. Both panels carry the same three marks -- the median heuristic, the
    criterion's pick, the grid's best TWA -- so the comparison is a matter of looking at
    where the marks sit rather than of reading two colour bars against each other.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pick = select(df, alpha, beta)
    best = df.loc[df["TWA"].idxmax()]
    marks = [
        (0.0, 0.0, "X", _INK, "median heuristic"),
        (pick["exp_o"], pick["exp_A"], "o", _ACCENT, rf"criterion ($\alpha$={alpha:g}, $\beta$={beta:g})"),
        (best["exp_o"], best["exp_A"], "s", _SECOND, "best TWA in the grid"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9))
    _field(axes[0], df, df["logdet"].to_numpy(float), "Blues", r"$n^{-1}\log\det(K_\theta+\lambda I)$")
    axes[0].set_title(r"the log-determinant term alone", color=_INK, fontsize=11, loc="left")
    _field(axes[1], df, objective(df, alpha, beta), "Blues", r"$J(\theta)$", floor_pct=20.0)
    axes[1].set_title(rf"the criterion  ($\alpha$={alpha:g}, $\beta$={beta:g})",
                      color=_INK, fontsize=11, loc="left")
    _field(axes[2], df, df["TWA"].to_numpy(float), "Greens", "TWA")
    axes[2].set_title("what it is standing in for", color=_INK, fontsize=11, loc="left")
    for ax in axes:
        _marks(ax, marks)

    caption = textwrap.wrap(
        f"{len(df)} cells; both bandwidths swept over 2^{EXPONENTS[0]:g}..2^{EXPONENTS[-1]:g} times "
        f"their median heuristic, sigma_phi left at it. Headline TWA at the median heuristic "
        f"{_cell(df, 0, 0)['TWA']:.3f}, at the criterion's pick {pick['TWA']:.3f}, at the best cell "
        f"in the grid {best['TWA']:.3f}. The middle panel's colour range is clipped below its 20th "
        f"percentile, so the structure near the maximum survives the barrier's dive.", width=104)
    fig.tight_layout(rect=(0, 0, 1, 0.85 - 0.028 * len(caption)), w_pad=3.0)
    fig.text(0.005, 0.99, f"Where the criterion looks, and where TWA actually is  --  "
                          f"{cal.method} on {cal.task}", color=_INK, fontsize=13, ha="left", va="top")
    fig.text(0.005, 0.93, "\n".join(caption), color=_MUTED, fontsize=9.5, ha="left", va="top",
             linespacing=1.5)
    # One legend for all three panels, in the header band: inside a panel there is no corner
    # that is free of a mark on every task.
    leg = fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right",
                     bbox_to_anchor=(0.995, 1.0), fontsize=9, framealpha=0.0, ncol=1,
                     labelcolor=_MUTED)
    leg.set_frame_on(False)
    _save(fig, stem)
    plt.close(fig)


def plot_sweeps(df: pd.DataFrame, cal: Calibration, alpha_star: float, stem: str) -> None:
    """What the two knobs move: the bandwidths picked, and the TWA that follows.

    Bandwidths and TWA are different measures on different scales, so they get their own
    panels rather than a second y-axis. The median heuristic is the horizontal reference in
    both rows; on the TWA row the grid's best and worst cells bound what any rule could
    have achieved on this grid.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a_sweep = sweep(df, betas=(1.0,)).sort_values("alpha")
    b_sweep = sweep(df, alphas=(alpha_star,)).sort_values("beta")
    med, best = _cell(df, 0.0, 0.0), df.loc[df["TWA"].idxmax()]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.6), sharex="col")
    for col, (data, key, title) in enumerate([
        (a_sweep, "alpha", r"tail fraction $\alpha$   ($\beta = 1$)"),
        (b_sweep, "beta", rf"barrier weight $\beta$   ($\alpha = {alpha_star:g}$)"),
    ]):
        x = data[key].to_numpy(float)
        top, bot = axes[0, col], axes[1, col]
        top.axhline(0.0, color=_GRID, lw=1.5, zorder=1)
        top.plot(x, np.log2(data["mult_o"]), "-o", color=_ACCENT, lw=2, ms=5, label=r"$\sigma_o$")
        top.plot(x, np.log2(data["mult_A"]), "-s", color=_SECOND, lw=2, ms=5, label=r"$\sigma_A$")
        top.set_xscale("log")
        top.set_yticks([e for e in EXPONENTS if e == int(e)])
        top.set_yticklabels([f"{2.0**e:g}x" for e in EXPONENTS if e == int(e)])
        top.set_ylim(min(EXPONENTS) - 0.4, max(EXPONENTS) + 0.4)
        top.set_ylabel("bandwidth picked\n" + r"($\times$ median heuristic; the 1x rule is the line)",
                       color=_MUTED, fontsize=9)
        top.set_title(title, color=_INK, fontsize=11, loc="left")
        _style(top)
        if col == 0:
            leg = top.legend(loc="lower left", fontsize=9, framealpha=0.92, facecolor=_SURFACE,
                             edgecolor=_GRID, labelcolor=_MUTED, ncol=2)
            leg.get_frame().set_linewidth(0.8)

        bot.axhspan(df["TWA"].min(), df["TWA"].max(), color=_GRID, alpha=0.55, lw=0,
                    label="range over the grid")
        bot.axhline(med["TWA"], color=_INK, lw=1.6, ls="--", label="median heuristic")
        bot.axhline(best["TWA"], color=_SECOND, lw=1.6, ls=":", label="best cell in the grid")
        bot.plot(x, data["TWA"], "-o", color=_ACCENT, lw=2, ms=5, label="the criterion's pick")
        bot.set_xscale("log")
        bot.set_xticks(x)
        bot.set_xticklabels([f"{v:g}" for v in x])
        bot.set_xlabel(r"$\beta$" if key == "beta" else r"$\alpha$", color=_MUTED, fontsize=9)
        bot.set_ylabel("headline TWA", color=_MUTED, fontsize=9)
        _style(bot)
        if col == 0:
            leg = bot.legend(loc="lower left", fontsize=8, framealpha=0.92, facecolor=_SURFACE,
                             edgecolor=_GRID, labelcolor=_MUTED, ncol=2)
            leg.get_frame().set_linewidth(0.8)

    fig.suptitle(f"What the criterion's two knobs actually move  --  {cal.method} on {cal.task}",
                 color=_INK, fontsize=13, x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, stem)
    plt.close(fig)


def plot_proxy(df: pd.DataFrame, cal: Calibration, alpha: float, beta: float, stem: str) -> None:
    """Criterion against TWA, cell by cell: does liking a kernel predict scoring well?"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    ok = df["TWA"].notna().to_numpy()
    obj, twa = objective(df, alpha, beta)[ok], df["TWA"].to_numpy(float)[ok]
    rho = float(spearmanr(obj, twa).statistic)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    sc = ax.scatter(obj, twa, c=df["exp_A"].to_numpy(float)[ok], cmap="Blues", s=26,
                    edgecolors=_SURFACE, linewidths=0.6, zorder=3)
    bar = fig.colorbar(sc, ax=ax, pad=0.02)
    bar.set_label(r"$\sigma_A$ ($\log_2$ of the multiplier)", color=_MUTED, fontsize=8)
    bar.ax.tick_params(colors=_MUTED, labelsize=7)
    bar.outline.set_visible(False)
    med = _cell(df, 0.0, 0.0)
    ax.axhline(med["TWA"], color=_INK, lw=1.4, ls="--", zorder=2)
    ax.text(obj.min(), med["TWA"], " median heuristic", color=_MUTED, fontsize=8, va="bottom")
    pick = select(df, alpha, beta)
    ax.plot(pick["objective"], pick["TWA"], "o", color=_ACCENT, ms=12, markerfacecolor="none",
            markeredgewidth=2.0, zorder=4)
    ax.annotate("the criterion's pick", (pick["objective"], pick["TWA"]), textcoords="offset points",
                xytext=(-18, 14), ha="right", color=_ACCENT, fontsize=8.5, zorder=6,
                bbox={"facecolor": _SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
                arrowprops={"arrowstyle": "-", "color": _ACCENT, "lw": 1.0, "shrinkB": 8})
    ax.set_xlabel(rf"$J(\theta)$   ($\alpha$={alpha:g}, $\beta$={beta:g})", color=_MUTED, fontsize=9)
    ax.set_ylabel("headline TWA", color=_MUTED, fontsize=9)
    _style(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.text(0.005, 0.99, f"Does liking a kernel predict scoring well?   Spearman $\\rho$ = {rho:+.2f}",
             color=_INK, fontsize=12, ha="left", va="top")
    fig.text(0.005, 0.92, f"{int(ok.sum())} cells, {cal.method} on {cal.task}. A criterion worth "
                          "tuning with has to rank kernels the way TWA does.",
             color=_MUTED, fontsize=8.5, ha="left", va="top")
    _save(fig, stem)
    plt.close(fig)


def plot_across_tasks(summary: pd.DataFrame, alpha: float, stem: str) -> None:
    """The study in one figure: which bandwidths each rule picks, and what that costs.

    Top, one bandwidth plane per task, in multiples of that task's median heuristic -- so
    the median heuristic is the origin everywhere by construction and the arrow is exactly
    the move the criterion makes. One panel per task rather than one shared scatter: the
    grid is discrete and the picks genuinely coincide across tasks, so a shared scatter
    cannot label its own marks. Bottom, the headline TWA those moves buy, as displacements
    from the median heuristic -- the summary table's numbers, direct-labelled.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = list(dict.fromkeys(summary["Task"]))
    kinds = [("criterion", rf"criterion ($\alpha$={alpha:g}, $\beta$=1)", _ACCENT, "o"),
             ("criterion_best", "criterion, best knobs (oracle)", _THIRD, "^"),
             ("oracle", "best bandwidths in the grid (oracle)", _SECOND, "s")]
    by = {(r["Task"], r["Kind"]): r for _, r in summary.iterrows()}

    fig = plt.figure(figsize=(2.55 * len(tasks) + 0.6, 7.0))
    gs = fig.add_gridspec(2, len(tasks), height_ratios=[1.25, 1], top=0.78, bottom=0.08,
                          left=0.06, right=0.98, hspace=0.30, wspace=0.18)
    # Every second octave: at 8.5pt, "0.0625 0.125 0.25" do not fit side by side.
    ticks = [e for e in EXPONENTS if e == int(e) and int(e) % 2 == 0]
    for j, task in enumerate(tasks):
        ax = fig.add_subplot(gs[0, j])
        ax.axhline(0, color=_GRID, lw=1.2, zorder=1)
        ax.axvline(0, color=_GRID, lw=1.2, zorder=1)
        for kind, label, color, marker in kinds:
            row = by.get((task, kind))
            if row is None:
                continue
            x, y = np.log2(row["sigma_o (x med)"]), np.log2(row["sigma_A (x med)"])
            if kind == "criterion":  # the move the criterion makes, drawn as the move it is
                ax.annotate("", (x, y), (0, 0), arrowprops={
                    "arrowstyle": "->", "color": color, "lw": 1.4, "alpha": 0.55, "shrinkA": 7})
            ax.plot(x, y, marker, color=color, ms=9, markerfacecolor="none", markeredgewidth=2.0,
                    label=label, zorder=4)
        ax.plot(0, 0, "X", color=_INK, ms=12, label="median heuristic", zorder=5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([f"{2.0**e:g}" for e in ticks])
        ax.set_yticklabels([f"{2.0**e:g}" for e in ticks] if j == 0 else [])
        ax.set_xlim(min(EXPONENTS) - 0.5, max(EXPONENTS) + 0.5)
        ax.set_ylim(min(EXPONENTS) - 0.5, max(EXPONENTS) + 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\sigma_o$  ($\times$ median)", color=_MUTED, fontsize=8.5)
        if j == 0:
            ax.set_ylabel(r"$\sigma_A$  ($\times$ median heuristic)", color=_MUTED, fontsize=9)
        ax.set_title(task, color=_INK, fontsize=10.5, loc="left")
        _style(ax)
        if j == 0:
            handles, labels_ = ax.get_legend_handles_labels()

    ax = fig.add_subplot(gs[1, :])
    height, y0 = 0.26, np.arange(len(tasks))
    for i, (kind, label, color, _) in enumerate(kinds):
        vals = [by.get((t, kind), {}).get("dTWA", np.nan) for t in tasks]
        pos = y0 + (1 - i) * height
        ax.barh(pos, vals, height=height * 0.86, color=color, label=label, zorder=3)
        for y, v in zip(pos, vals):
            if np.isfinite(v):
                ax.text(v + (0.0006 if v >= 0 else -0.0006), y, f"{v:+.3f}", va="center",
                        ha="left" if v >= 0 else "right", color=_MUTED, fontsize=8, zorder=4)
    ax.axvline(0, color=_INK, lw=1.4, zorder=2)
    ax.set_yticks(y0)
    ax.set_yticklabels(tasks)
    ax.set_xlabel("headline TWA, as a change from the median heuristic", color=_MUTED, fontsize=9)
    ax.set_title("and what that is worth", color=_INK, fontsize=11, loc="left")
    ax.margins(x=0.1)
    _style(ax)
    # No second legend: the bars carry the same three colours as the marks above, in the
    # same order, and a box here has nowhere to sit that does not cover a bar.

    fig.text(0.005, 0.99, "The log-determinant criterion against the median heuristic, "
                          f"{summary['Method'].iloc[0]} on {len(tasks)} tasks",
             color=_INK, fontsize=13, ha="left", va="top")
    fig.text(0.005, 0.945, "Which bandwidths each rule picks, per task. Both oracle rules are "
                           "chosen by the test-set TWA they produce and cannot be run in practice; "
                           "they bound what the criterion could reach.",
             color=_MUTED, fontsize=9, ha="left", va="top")
    leg = fig.legend(handles, labels_, loc="upper left", bbox_to_anchor=(0.005, 0.90), ncol=4,
                     fontsize=9, labelcolor=_MUTED)
    leg.set_frame_on(False)
    _save(fig, stem)
    plt.close(fig)


def plot_rho(rhos: dict, alpha: float, stem: str) -> None:
    """Where in the ``(alpha, beta)`` box the criterion agrees with TWA -- on each task.

    This is the strong form of the question. The argmax comparison is one cell out of 289
    and can land well by luck; a rank correlation over the whole grid says whether the
    criterion orders kernels the way the benchmark does. Signed quantity with a meaningful
    zero, so: diverging ramp, neutral midpoint, one symmetric scale shared by all panels so
    the tasks can be read against each other.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("ldk_div", [_ACCENT, "#f2f1ec", _SECOND])
    names = list(rhos)
    span = max(abs(float(r["spearman"].min())) for r in rhos.values() if len(r))
    span = max(span, max(abs(float(r["spearman"].max())) for r in rhos.values() if len(r)))

    fig, axes = plt.subplots(1, len(names), figsize=(2.7 * len(names) + 2.2, 3.9), squeeze=False)
    for ax, name in zip(axes[0], names):
        piv = rhos[name].pivot(index="alpha", columns="beta", values="spearman")
        Z = piv.to_numpy(float)
        X, Y = np.log2(piv.columns.to_numpy(float)), np.log2(piv.index.to_numpy(float))
        mesh = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="nearest", vmin=-span, vmax=span)
        # The headline setting, and the best cell: two rings, so "the criterion is a decent
        # proxy somewhere" and "it is a decent proxy where we ran it" stay separate claims.
        j, i = np.unravel_index(int(np.argmax(Z)), Z.shape)
        ax.plot(X[i], Y[j], "s", color=_INK, ms=9, markerfacecolor="none", markeredgewidth=1.8)
        ax.plot(0.0, np.log2(alpha), "o", color=_INK, ms=9, markerfacecolor="none", markeredgewidth=1.8)
        ax.set_xticks(X)
        ax.set_xticklabels([f"{v:g}" for v in piv.columns], rotation=90)
        ax.set_yticks(Y)
        ax.set_yticklabels([f"{v:g}" for v in piv.index])
        ax.set_xlabel(r"$\beta$", color=_MUTED, fontsize=9)
        ax.set_title(f"{name}   (best {Z.max():+.2f})", color=_INK, fontsize=10, loc="left")
        _style(ax)
        if ax is axes[0][0]:
            ax.set_ylabel(r"$\alpha$", color=_MUTED, fontsize=9)
    # The colour bar gets its own axes rather than stealing space from the last panel: with
    # `ax=axes[0]` it is laid out over the panels, and a later subplots_adjust cannot move it.
    fig.subplots_adjust(left=0.055, right=0.88, top=0.68, bottom=0.22, wspace=0.28)
    bar = fig.colorbar(mesh, cax=fig.add_axes([0.905, 0.22, 0.013, 0.46]))
    bar.set_label(r"Spearman $\rho$ between $J(\theta)$ and TWA, over the 289 cells",
                  color=_MUTED, fontsize=8)
    bar.ax.tick_params(colors=_MUTED, labelsize=7)
    bar.outline.set_visible(False)
    fig.text(0.005, 0.99, "Does the criterion rank kernels the way the benchmark does?",
             color=_INK, fontsize=13, ha="left", va="top")
    fig.text(0.005, 0.87, f"Circle: the setting used for the headline rows ($\\alpha$={alpha:g}, "
                          r"$\beta$=1). Square: the best cell on that task. Blue = agrees, "
                          "orange = disagrees.", color=_MUTED, fontsize=9, ha="left", va="top")
    _save(fig, stem)
    plt.close(fig)


# ----------------------------------------------------------------------------------- main
def run_task(task: str, method: str, seed: int, force: bool, cache_only: bool,
             no_plots: bool = False, alpha: float = ALPHA_DEFAULT,
             group: int = GROUP,
             loo: str = "step") -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Everything for one (task, method): the grid, the sweeps, the plots, the summary rows."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cal = prepare(task, method, seed)
    tag = "" if loo == "step" else f"__loo_{loo}"
    crit_path = CACHE_DIR / f"{task}__{method}__criterion{tag}__seed{seed}.csv"
    if crit_path.exists() and not force:
        crit = pd.read_csv(crit_path)
    else:
        started = time.time()
        crit = criterion_grid(cal, blocked=(loo == "episode"))
        crit.to_csv(crit_path, index=False)
        print(f"criterion grid: {len(crit)} cells in {time.time() - started:.1f}s -> "
              f"{os.path.relpath(crit_path, ROOT_DIR)}")

    fac_path = CACHE_DIR / f"{task}__{method}__factors__seed{seed}.csv"
    if fac_path.exists() and not force:
        fac = pd.read_csv(fac_path)
    else:
        fac = factor_grid(cal)
        fac.to_csv(fac_path, index=False)
    crit = crit.merge(fac, on=["exp_o", "exp_A"], how="left")

    cells = twa_landscape(task, method, seed, force=force, cache_only=cache_only, group=group)
    df = merge(crit, reduce_landscape(cells))
    refs = report_landscape(df, cal)
    med, best = refs["median"], refs["best"]

    report_alpha(df, med)
    report_beta(df, med, alpha)
    report_balance(df, alpha)
    rho = report_proxy(df)

    full = sweep(df)
    full.insert(0, "Task", task)
    full.insert(1, "Method", method)
    full.to_csv(CACHE_DIR / f"{task}__{method}__sweep{tag}__seed{seed}.csv", index=False)

    # The best (alpha, beta) in the whole sweep, chosen by the TWA it produces. That is an
    # oracle over the criterion's own knobs -- it cannot be run in practice -- and it is
    # here as an upper bound: if even this does not clear the median heuristic, no setting
    # of the two knobs does.
    knob_best = full.loc[full["TWA"].idxmax()] if full["TWA"].notna().any() else None
    rows = [
        summary_row(task, method, seed, "median", "median heuristic", med),
        summary_row(task, method, seed, "criterion", f"criterion (alpha={alpha:g}, beta=1)",
                    select(df, alpha, 1.0)),
    ]
    if knob_best is not None:
        rows.append(summary_row(
            task, method, seed, "criterion_best",
            f"criterion, best knobs (alpha={knob_best['alpha']:g}, beta={knob_best['beta']:g})",
            knob_best))
    rows.append(summary_row(task, method, seed, "oracle", "best bandwidths in grid", best))

    df_q = report_tau(crit, cells, alpha, 1.0)
    if df_q is not None:
        rows += [
            summary_row(task, method, seed, "median_tau", "median heuristic, tau threshold",
                        _cell(df_q, 0.0, 0.0)),
            summary_row(task, method, seed, "criterion_tau",
                        f"criterion (alpha={alpha:g}, beta=1), tau threshold", select(df_q, alpha, 1.0)),
        ]

    if not no_plots:
        stem = f"{task}__{method}{tag}"
        plot_landscape(df, cal, alpha, 1.0, f"{stem}__landscape")
        plot_sweeps(df, cal, alpha, f"{stem}__sweeps")
        plot_proxy(df, cal, alpha, 1.0, f"{stem}__proxy")
    return df, rows, rho


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="logdet_kernel", description=__doc__)
    parser.add_argument("--task", nargs="+", default=["pretzel"], choices=run_mod.ALL_TASKS)
    parser.add_argument("--method", nargs="+", default=["kern_cd_flat"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT,
                        help="tail fraction for the beta sweep and the headline rows")
    parser.add_argument("--loo", default="step", choices=["step", "episode"],
                        help="hold out one step (kern_cd's own definition) or one whole episode")
    parser.add_argument("--group", type=int, default=GROUP,
                        help="cells per evaluation call; lower it on the big tasks to cap memory")
    parser.add_argument("--report", action="store_true", help="report cached cells only")
    parser.add_argument("--force", action="store_true", help="ignore the cache")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="run the checks and exit")
    args = parser.parse_args(argv)

    started = time.time()
    check_cvar()
    if args.check_only:
        cal = prepare(args.task[0], args.method[0], args.seed)
        check_grid(cal)
        check_block_loo(cal)
        return 0

    rows, rhos = [], {}
    for task in args.task:
        for method in args.method:
            _, new, rho = run_task(task, method, args.seed, args.force, args.report, args.no_plots,
                                   alpha=args.alpha, group=args.group, loo=args.loo)
            rows += new
            rhos[task if len(args.method) == 1 else f"{task} / {method}"] = rho

    summary = pd.DataFrame(rows)
    # Every rule is judged against the median heuristic on its own task, which is the only
    # comparison the spec asks for; TWA levels are not comparable across tasks.
    base = summary[summary["Rule"] == "median heuristic"].set_index(["Task", "Method"])["TWA"]
    summary.insert(summary.columns.get_loc("Accuracy"), "dTWA",
                   summary["TWA"] - summary.set_index(["Task", "Method"]).index.map(base))
    print("\n=== summary: the effect of the tuning rule on the headline benchmark scores ===")
    print("  dTWA is against the median heuristic on the same task; the 'tau threshold' rows use "
          "only\n  FIPER's 1-alpha quantile instead of the 10-quantile average")
    print(summary.drop(columns=["Kind"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    path = CACHE_DIR / ("summary.csv" if args.loo == "step" else f"summary__loo_{args.loo}.csv")
    summary.to_csv(path, index=False)
    print(f"\n-> {os.path.relpath(path, ROOT_DIR)}")
    if not args.no_plots and summary["Task"].nunique() > 1:
        stem = f"summary__{'_'.join(args.method)}" + ("" if args.loo == "step" else f"__loo_{args.loo}")
        plot_across_tasks(summary, args.alpha, stem)
        plot_rho(rhos, args.alpha, f"{stem}__rho")
    print(f"total wall clock: {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
