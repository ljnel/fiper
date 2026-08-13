"""Trajectory-aware rewrite of the log-determinant criterion -- follow-up to ``logdet_kernel.py``.

That study found the criterion of ``specs/experiments/logdet_kernel_objective.md`` does not
improve the headline, and found it failing in a specific way: at ``beta = 1`` it shrinks
``sigma_o``, and where it wins it wins by switching the action channel off. Both are
symptoms of the same thing -- the fit set is time steps pooled from trajectories, and
neither term of the criterion is written for that.

Why the pooling breaks it
-------------------------
With ``E`` episodes of ~``T`` steps, ``n = ET``:

* The **median heuristic** is a median over pairs, and only ~``1/E`` of the pairs are
  within-episode, so the near-duplicate mass is a few percent of a distribution and a
  median ignores it.
* The **log-determinant** is ``(1/n) sum_i log(mu_i + lam)`` over the *spectrum*. Only ~``E``
  directions are between-episode; the other ``n - E`` -- a fraction ``1 - 1/T`` -- sit at
  ``mu_i ~ 0`` because consecutive steps are near-duplicates, each contributing ``log lam``.
  Shrinking a bandwidth lifts one of those by up to ``log(1/lam) ~ 11.5`` nats. So the term
  is ~95% a measure of *temporal resolution*, not of how well the kernel separates
  situations, and buying it is what drives ``sigma_o`` down.

``logdet_kernel``'s ``--loo episode`` fixed the *risk* side and left the *volume* side
alone, which is why it moved so little. This file rewrites all three parts.

The criterion
-------------
    J(theta) = V_w(theta) + gamma * C(theta) + beta * log(1 - CVaR_alpha(g(theta)))

**V -- volume over the independent unit.** Bin the steps into units (a whole episode, or a
phase bin of ``w`` steps), give each unit its kernel mean embedding, and take the
determinant of the ``U x U`` Gram matrix of those embeddings:

    Gbar_uu' = mean_{i in u, j in u'} k(x_i, x_j)      for u != u'
    Gbar_uu  = mean_{i != j in u}    k(x_i, x_j)       (self pairs removed)
    V = logdet(I + Gbar/rho) / (U log(1 + 1/rho))  in [0, 1].

The ``i != j`` diagonal is load-bearing. With the self pairs left in, a narrow kernel keeps
a free ``Gbar_uu -> 1/n_u`` and the term is still buyable by shrinking; with them removed,
``sigma -> 0`` sends ``Gbar -> 0`` and ``V -> 0``, the worst attainable value. What is left
rewards ``Gbar_uu`` large (wide enough that a trajectory's own steps look like one object)
and ``Gbar_uu'`` small (narrow enough that distinct trajectories stay apart) -- the balance
the step-level determinant never expressed. ``w = 1`` recovers a step-level determinant, so
the width sweep interpolates continuously back to the original criterion.

**The barrier -- risk per trajectory, on the statistic FIPER thresholds.** ``q`` is the
leave-one-*episode*-out score (``logdet_kernel._block_loo``), aggregated the way the
benchmark aggregates it -- rolling mean over ``H`` steps, worst window in the episode --
and the tail is taken over the ``E`` episode statistics, not the ``n`` steps:

    g_b = max_t mean(q_{b,t-H+1..t}),    CVaR_alpha(g) over b = 1..E.

The pooled-step tail has an effective sample size of ``E`` anyway (the worst 5% of steps
are one or two episodes) and over-weights long episodes; this version does not, and its
``tau`` is a threshold at a false-alarm rate of ``alpha`` *per nominal trajectory*, which
is the operating quantity, rather than a per-step quantile.

**C -- the contrast that closes the degenerate direction.** Pooled trajectories supply
negatives for free: take the observation of one step and the action chunk of another,
``(o_bt, A_b't')``. A support estimator must call that novel. ``C`` is the mean increase in
score the swap causes, against the matched control ``(o_bt, A_bt)``. This is the only term
that sees a deleted channel: at ``sigma_A -> inf`` the action factor is identically 1, the
swapped point's kernel row *equals* the control's, and ``C -> 0`` exactly, while ``V`` and
the barrier are both indifferent. Two negative sets ride along: cross-episode swaps (the
term in ``J``) and same-episode swaps far in time (a diagnostic -- the near-miss case, and
the one the benchmark is actually detecting).

Two things about C were wrong before they were measured, and both are asserted in
``--check-only`` now:

* **Both episodes have to be held out**, the donor's as well as the source's. With the
  donor left in the fit set the swapped point sits at zero action-distance from an actual
  fit point, scores *lower* than its control, and C ends up rewarding the inert action
  channel it exists to reject -- measured on push_chair as C rising monotonically in
  ``sigma_A`` to its maximum at ``sigma_A = 16x``.
* **C is a mean gap, not an AUC.** An AUC is scale-free, so at ``sigma_A = 4096x`` it reads
  1.0 off a 1e-7 separation: it reports the direction of the action channel's effect
  perfectly while the effect itself vanishes. The mean gap decays to 0 at both ends --
  ``sigma_A -> inf`` (identical rows) and ``sigma_A -> 0`` (both points maximally novel) --
  and peaks where the action channel actually carries information. The AUC is kept as a
  diagnostic column.

All three terms are dimensionless and O(1), so ``beta`` and ``gamma`` are weights of order
one rather than the task-dependent exchange rate ``logdet_kernel`` found spanning 200x.

How this is measured
--------------------
The same 17 x 17 grid over ``(sigma_o, sigma_A)`` as ``logdet_kernel``, so **the 1445 cached
FIPER evaluations are reused unchanged** -- the TWA landscape does not depend on how the
criterion is defined, and the whole study is a re-scoring. Every quantity above comes off
one Cholesky per cell, plus two gemms for the negatives. The ladder re-scored on that grid:

    A0  logdet + pooled-step tail of step-LOO scores      (the original criterion)
    A1  logdet + pooled-step tail of episode-LOO scores   (``logdet_kernel --loo episode``)
    A2  V      + pooled-step tail of step-LOO scores      (volume fix alone)
    A3  V      + per-episode windowed tail                (volume + risk fixes)
    A4  V + gamma C + per-episode windowed tail           (all three)
    A5  gamma C + per-episode windowed tail               (contrast alone)

plus **A6**, which is A4 with ``beta`` and ``gamma`` set by :func:`balanced_knobs` -- the
values making the three terms equal in median magnitude over the grid. That rule reads only
the criterion's own values, never TWA, so unlike the sweep's argmax it is runnable on a new
task. All of it is measured against the median heuristic and the grid's best cell. The
acceptance test is not the argmax -- one cell in 289 can land well by luck -- but the
Spearman correlation between ``J`` and TWA over all 289 cells, reported for every variant
and every knob setting.

What it found
-------------
**1. The volume fix is the one that matters, and it flips the sign of the headline.**
Mean TWA against the median heuristic over the five tasks, at ``alpha = 0.05``:

    A0 -0.011 (1 of 5 tasks win)  ->  A2 +0.009 (3 of 5)  ->  A3 +0.009 (3 of 5)

A0 reproduces ``logdet_kernel``'s -0.011 and 1-of-5 exactly, as it should. Swapping the
step-level determinant for the episode-level V is what moves it; per task A3 scores -0.011,
+0.016, +0.024, +0.021, -0.004. That is about a quarter of the oracle's +0.039, so the
diagnosis was right about the mechanism, and the fix is real but partial.

**2. ``beta`` is transferable now.** The ``beta`` that balances the two terms is

    push_chair 0.145,  stacking 0.183,  push_t 0.217,  sorting 0.331,  pretzel 0.516,

a 3.6x spread, against 0.099 to 19.8 -- 200x -- for the same measurement on the original
logdet (``logdet_kernel``'s finding 2, reproduced here in the same table). Dimensionless
terms did what they were supposed to. ``beta = 1`` is still not neutral, though: the log
barrier is unbounded below while V is confined to [0, 1], so the balance sits near 0.2, and
at ``beta = 1`` the criterion is the barrier with a rounding error attached -- the original's
failure mode with the roles of the two terms exchanged.

**3. V alone makes the degeneracy worse, and C is the only thing that fixes it.** Number of
tasks whose pick has the action channel inert (mean off-diagonal action factor > 0.9):

    A0 2/5,  A1 3/5,  A2 4/5,  A3 5/5,  A4 1/5,  A6 0/5.

V by itself deletes the action channel on *every* task -- worse than the criterion it
replaces -- which is what a volume term must do, since nothing in it can see the difference
between a balanced kernel and a one-channel one. With the contrast term weighted by the
balance rule, no pick is degenerate on any task, and the picks sit at an action factor of
0.08-0.14 against the oracle's 0.001-0.08: closer to the oracle than any other rule here,
and the thing the spec set out to test.

**4. But the criterion still is not a proxy for TWA.** Mean Spearman between ``J`` and
headline TWA over the 289 cells: A0 -0.075, A4 -0.026, A6 +0.000. Per task A6 gives +0.13,
+0.03, -0.02, -0.37, +0.23 -- sorting still anti-correlates hard, and the sign still flips
across tasks. So the three fixes changed *where the argmax lands* without making the
surface underneath it trustworthy, and this is ``logdet_kernel``'s finding 4 surviving all
of them. A rule that recovers a quarter of the headroom off a surface with zero mean rank
correlation is doing so partly by luck.

**5. ``gamma`` inherits the disease ``beta`` was cured of.** ``median |C|`` spans 0.001
(push_chair) to 0.037 (pretzel) and the balancing ``gamma`` spans 3 to 272. C's magnitude is
how much the action channel matters on a task, which is exactly what is unknown in advance.
A6 is runnable -- its knobs need no TWA -- but they are doing a lot of work, and at the
pre-registered ``gamma = 1`` the contrast term is a no-op on four of five tasks.

**6. The remaining knobs barely matter.** Over ``alpha`` in [0.01, 1], ``H`` in [1, 20] and
``rho`` in [1e-4, 1e-1], the TWA of the pick moves by less than 0.01 on four of five tasks.
Unit width is the same story: ``w = 1`` -- a step-level determinant with the self pairs
removed -- differs from whole-episode units by at most 0.05 and not consistently in sign. So
what matters is the *definition* of the volume term, not the bin width inside it.

**What this does not settle.** One seed, ``kern_cd_flat`` only, ``sigma_phi`` at the median
heuristic throughout, and a half-octave grid, so a pick is resolved to a factor of 1.4 --
the same limits as ``logdet_kernel``. push_chair's fit set is 49 steps over 10 episodes,
which makes its per-episode tail a statistic of ~10 numbers. Findings 2, 3 and 5 are
structural; 1 and 4 are five numbers at one seed.

Running it
----------
    pixi run python my_experiments/logdet_kernel/trajectory_criterion.py --check-only
    pixi run python my_experiments/logdet_kernel/trajectory_criterion.py --task pretzel
    pixi run python my_experiments/logdet_kernel/trajectory_criterion.py \
        --task pretzel push_t push_chair sorting stacking

Cell grids are cached under ``cache/`` as ``<task>__<method>__traj__seed<seed>.csv``;
``--force`` re-runs them. All five tasks from scratch is 100 minutes on an RTX 5090 (5 x 289
cells, dominated by stacking and sorting at ~6 s a cell, all of it CPU linear algebra); from
the cache the whole report is seconds. Nothing outside this file is written to, and no FIPER
evaluation is run: if a cell of the TWA landscape is missing from the cache it is reported
as NaN rather than evaluated.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import textwrap
import time
from dataclasses import dataclass

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402

import logdet_kernel as ldk  # noqa: E402
from logdet_kernel import EXPONENTS, LAM, MULTS, ROOT_DIR, Calibration, cvar  # noqa: E402

CACHE_DIR = ldk.CACHE_DIR

#: Unit widths for V, in steps; 0 means the whole episode. 1 is the degenerate end -- a
#: step-level determinant -- and is kept so the sweep interpolates back to the original.
WIDTHS = (1, 4, 16, 0)

#: Ridge on the unit Gram matrix. It sets the resolution floor of V exactly as ``lam`` does
#: for the step-level determinant; the default follows kern_cd's convention of scaling with
#: the number of rows, and the rest of the sweep brackets it by two orders either way.
RHOS = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)

#: Rolling-window lengths for the episode statistic, from FIPER's own window grid (1..50).
WINDOWS = (1, 3, 5, 10, 20)

#: Weights on the contrast term. 0 is the ablation.
GAMMAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

ALPHAS, BETAS = ldk.ALPHAS, ldk.BETAS
ALPHA_DEFAULT = ldk.ALPHA_DEFAULT

#: Headline knobs, fixed in advance for the same reason ``logdet_kernel`` fixes alpha:
#: choosing them by the TWA they produce would be tuning on the test set.
BETA_DEFAULT, GAMMA_DEFAULT, H_DEFAULT, WIDTH_DEFAULT = 1.0, 1.0, 5, 0

#: Mismatched pairs per cell, split evenly between the two negative kinds. The cost is two
#: m x m x N_NEG gemms per cell, so this trades the contrast term's Monte Carlo error
#: (~1/sqrt(N)) against ~1 s per cell on the largest task.
N_NEG = 1000


# --------------------------------------------------------------------------- the new terms
def unit_index(episode: np.ndarray, width: int) -> np.ndarray:
    """Contiguous bins of ~``width`` steps within each episode (``width = 0``: whole episode)."""
    unit = np.empty(len(episode), int)
    u = 0
    for b in np.unique(episode):
        idx = np.flatnonzero(episode == b)
        n_bins = 1 if width == 0 else max(1, len(idx) // max(1, width))
        for part in np.array_split(np.arange(len(idx)), n_bins):
            unit[idx[part]] = u
            u += 1
    return unit


def unit_gram(K: np.ndarray, unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gram matrix of the units' kernel mean embeddings, with the self pairs removed.

    The blocks are contiguous, so the double block sum is two ``reduceat`` passes. The
    diagonal is the *unbiased* within-unit mean: leaving the ``k(x_i,x_i) = 1`` terms in
    would hand a narrow kernel a free ``1/n_u`` and reopen the degeneracy V exists to close.
    """
    starts = np.flatnonzero(np.r_[True, np.diff(unit) != 0])
    counts = np.diff(np.r_[starts, len(unit)]).astype(float)
    S = np.add.reduceat(np.add.reduceat(K, starts, axis=0), starts, axis=1)
    G = S / np.outer(counts, counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        diag = np.where(counts > 1, (np.diag(S) - counts) / (counts * (counts - 1)), 1.0)
    np.fill_diagonal(G, diag)
    return G, counts


def volume(G: np.ndarray, rhos=RHOS) -> dict[float, float]:
    """``logdet(I + G/rho) / (U log(1 + 1/rho))`` in [0, 1], for each ridge.

    Removing the self pairs can cost the Gram matrix positive definiteness by ``O(1/n_u)``,
    so the eigenvalues are clipped at zero rather than the ridge being inflated -- that
    keeps V comparable across cells, which is the only thing it is used for.
    """
    ev = np.clip(np.linalg.eigvalsh(G), 0.0, None)
    return {r: float(np.log1p(ev / r).sum() / (len(ev) * np.log1p(1.0 / r))) for r in rhos}


def episode_stat(q: np.ndarray, episode: np.ndarray, window: int) -> np.ndarray:
    """Each episode's worst ``window``-step rolling mean of the LOO scores."""
    out = []
    for b in np.unique(episode):
        x = q[episode == b]
        h = min(window, len(x))
        c = np.cumsum(np.r_[0.0, x])
        out.append(float(((c[h:] - c[:-h]) / h).max()))
    return np.asarray(out)


#: Donor episodes per source episode, for the cross-episode negatives. This caps the number
#: of distinct held-out pairs -- and therefore of block inverses per cell -- at ~5E rather
#: than at E^2; it costs nothing statistically, since the pairs are still drawn at random.
N_DONORS = 4


@dataclass
class Negatives:
    """Mismatched ``(observation, action chunk)`` pairs, fixed across cells so C is paired."""

    i: np.ndarray  # step supplying the observation
    j: np.ndarray  # step supplying the action chunk
    cross: np.ndarray  # True: donor from another episode; False: same episode, far in time


def make_negatives(cal: Calibration, n_neg: int = N_NEG, seed: int = 0) -> Negatives:
    """Draw the two negative sets: cross-episode swaps, and same-episode swaps far in time."""
    rng = np.random.default_rng(seed)
    ep = cal.episode
    ids = np.unique(ep)
    blocks = {b: np.flatnonzero(ep == b) for b in ids}
    donors = {b: rng.choice(ids[ids != b], size=min(N_DONORS, len(ids) - 1), replace=False)
              for b in ids} if len(ids) > 1 else {}
    half = n_neg // 2
    i = rng.integers(0, cal.m, size=2 * half)
    j = np.empty_like(i)
    cross = np.r_[np.ones(half, bool), np.zeros(half, bool)]
    for k, (src, is_cross) in enumerate(zip(i, cross)):
        b = ep[src]
        if is_cross and len(blocks) > 1:
            j[k] = rng.choice(blocks[rng.choice(donors[b])])
            continue
        idx = blocks[b]
        pos = int(np.flatnonzero(idx == src)[0])
        far = idx[np.abs(np.arange(len(idx)) - pos) >= max(1, len(idx) // 4)]
        # A short episode may have no distant step; fall back to its farthest one.
        j[k] = rng.choice(far) if len(far) else idx[0 if pos > len(idx) // 2 else -1]
        cross[k] = False
    return Negatives(i=i, j=j, cross=cross)


def blocks_of(episode: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """The row indices of each episode, and the episode each row belongs to."""
    ids = np.unique(episode)
    blocks = [np.flatnonzero(episode == b) for b in ids]
    return blocks, np.searchsorted(ids, episode)


def block_loo(L_inv: np.ndarray, blocks: list[np.ndarray], lam_m: float
              ) -> tuple[np.ndarray, list[np.ndarray]]:
    """Leave-one-episode-out scores and the ``inv(A_BB)`` they come from (kept for C)."""
    loo = np.empty(L_inv.shape[1])
    inv = []
    for idx in blocks:
        col = L_inv[:, idx]
        A_inv = np.linalg.inv(col.T @ col)
        loo[idx] = np.diag(A_inv) - lam_m
        inv.append(A_inv)
    return loo, inv


def probe_scores(D_o: np.ndarray, D_A: np.ndarray, i: np.ndarray, j: np.ndarray,
                 hold: np.ndarray, L_inv: np.ndarray, blocks: list[np.ndarray],
                 sig_o: float, sig_A: float) -> np.ndarray:
    """kern_cd score of each probe ``(o_i, A_j)``, with the episodes in ``hold`` held out.

    The probe's kernel row against fit point ``l`` is
    ``exp(-1/2 (D_o[i,l]/sigma_o^2 + D_A[j,l]/sigma_A^2))`` -- both distance matrices are
    already built, so no feature is recomputed. Zeroing that row on the held-out block and
    applying ``(K_-B + lam I)^-1 = A_-B,-B - A_-B,B A_BB^-1 A_B,-B`` gives the score from
    the *same* factor as the LOO scores, which is what makes C a paired comparison;
    :func:`check_negatives` verifies it by feeding matched pairs in and recovering
    :func:`block_loo`.

    **Both** the source and the donor episode are held out, which is not a refinement but a
    correctness requirement: with the donor left in the fit set, a swapped point sits at
    zero action-distance from an actual fit point and so scores *lower* than its matched
    control, and C then rewards exactly the inert action channel it exists to reject
    (measured: C rising monotonically to 1/2 as sigma_A -> inf, 0.01 at sigma_A = 1/16x).
    """
    Kn = np.exp(-0.5 * (D_o[i] / sig_o**2 + D_A[j] / sig_A**2))
    groups: dict[tuple, list[int]] = {}
    for k, h in enumerate(hold):
        groups.setdefault(tuple(sorted(set(int(b) for b in h))), []).append(k)
    groups = {key: np.asarray(rows) for key, rows in groups.items()}
    removed = {key: np.concatenate([blocks[b] for b in key]) for key in groups}
    for key, rows in groups.items():
        Kn[np.ix_(rows, removed[key])] = 0.0
    U = L_inv @ Kn.T  # [m, N]
    quad = np.einsum("ki,ki->i", U, U)  # k~^T A k~
    W = L_inv.T @ U  # [m, N] = A k~
    for key, rows in groups.items():
        R = removed[key]
        col = L_inv[:, R]
        w = W[np.ix_(R, rows)]
        quad[rows] -= np.einsum("ai,ai->i", w, np.linalg.inv(col.T @ col) @ w)
    return 1.0 - quad


# ------------------------------------------------------------------------------- the grid
def cell_stats(cal: Calibration, mult_o: float, mult_A: float, neg: Negatives,
               units: dict[int, np.ndarray], alphas=ALPHAS, windows=WINDOWS,
               widths=WIDTHS, rhos=RHOS) -> dict:
    """Every criterion quantity at one grid cell, from one Cholesky factor."""
    sig_o, sig_A = mult_o * cal.sigma_o, mult_A * cal.sigma_A
    K = np.exp(-0.5 * (cal.D_o / sig_o**2 + cal.D_A / sig_A**2))
    lam_m = LAM * cal.m
    L = np.linalg.cholesky(K + lam_m * np.eye(cal.m))
    L_inv = solve_triangular(L, np.eye(cal.m), lower=True)

    row = {"mult_o": mult_o, "mult_A": mult_A,
           "logdet": 2.0 * float(np.sum(np.log(np.diag(L)))) / cal.m}

    for w in widths:  # V, at every unit width and ridge
        G, _ = unit_gram(K, units[w])
        for r, v in volume(G, rhos).items():
            row[f"V@w{w}@r{r:g}"] = v
        row[f"U@w{w}"] = G.shape[0]

    q_step = 1.0 / np.einsum("ki,ki->i", L_inv, L_inv) - lam_m
    blocks, block_of = blocks_of(cal.episode)
    q_ep, _ = block_loo(L_inv, blocks, lam_m)
    for a in alphas:  # the two pooled-step tails: A0's and A1's
        row[f"cvar_s@{a:g}"], row[f"tau_s@{a:g}"] = cvar(q_step, a)
        row[f"cvar_e@{a:g}"], row[f"tau_e@{a:g}"] = cvar(q_ep, a)
    for h in windows:  # the per-episode windowed tail
        g = episode_stat(q_ep, cal.episode, h)
        for a in alphas:
            row[f"cvar_g@{a:g}@H{h}"], row[f"tau_g@{a:g}@H{h}"] = cvar(g, a)

    # The matched control goes through the *same* path, on the *same* held-out pair of
    # episodes, rather than reusing q_ep: the two agree to 1e-14, but only identical
    # arithmetic makes the comparison tie exactly when the action factor is 1, which is the
    # degeneracy C has to report as chance.
    hold = np.c_[block_of[neg.i], block_of[neg.j]]
    both = probe_scores(cal.D_o, cal.D_A, np.r_[neg.i, neg.i], np.r_[neg.j, neg.i],
                        np.r_[hold, hold], L_inv, blocks, sig_o, sig_A)
    s_neg, matched = both[: len(neg.i)], both[len(neg.i) :]
    win = (s_neg > matched).astype(float) + 0.5 * (s_neg == matched)
    for tag, sel in (("cross", neg.cross), ("lag", ~neg.cross)):
        row[f"C_{tag}"] = float((s_neg - matched)[sel].mean()) if sel.any() else np.nan
        row[f"auc_{tag}"] = float(win[sel].mean()) if sel.any() else np.nan
    return row


def traj_grid(cal: Calibration, n_neg: int = N_NEG) -> pd.DataFrame:
    """One row per grid cell, for the whole 17 x 17 sweep."""
    neg = make_negatives(cal, n_neg)
    units = {w: unit_index(cal.episode, w) for w in WIDTHS}
    rows = []
    started = time.time()
    for eo, mo in zip(EXPONENTS, MULTS):
        for ea, ma in zip(EXPONENTS, MULTS):
            row = cell_stats(cal, mo, ma, neg, units)
            rows.append({"exp_o": eo, "exp_A": ea, **row})
        print(f"  sigma_o = {mo:g}x done, {time.time() - started:.0f}s elapsed", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- the variants
@dataclass(frozen=True)
class Variant:
    """One rung of the ablation ladder: which volume term, which tail, how much contrast."""

    key: str
    label: str
    vol: str | None  # None: no volume term
    tail: str  # "s", "e" or "g"
    gamma: float

    def columns(self, alpha: float, h: int, w: int, rho: float) -> tuple[str | None, str]:
        vol = None if self.vol is None else (
            "logdet" if self.vol == "logdet" else f"V@w{w}@r{rho:g}")
        tail = f"cvar_g@{alpha:g}@H{h}" if self.tail == "g" else f"cvar_{self.tail}@{alpha:g}"
        return vol, tail


LADDER = (
    Variant("A0", "logdet + step tail (original)", "logdet", "s", 0.0),
    Variant("A1", "logdet + episode-LOO step tail", "logdet", "e", 0.0),
    Variant("A2", "V + step tail", "V", "s", 0.0),
    Variant("A3", "V + per-episode windowed tail", "V", "g", 0.0),
    Variant("A4", "V + gamma C + per-episode tail", "V", "g", GAMMA_DEFAULT),
    Variant("A5", "gamma C + per-episode tail", None, "g", GAMMA_DEFAULT),
)


def objective(grid: pd.DataFrame, var: Variant, alpha: float, beta: float,
              gamma: float | None = None, h: int = H_DEFAULT, w: int = WIDTH_DEFAULT,
              rho: float = 1.0e-3) -> np.ndarray:
    """``J(theta)`` over the whole grid, for one variant at one knob setting."""
    g = var.gamma if gamma is None else gamma
    vol_col, tail_col = var.columns(alpha, h, w, rho)
    out = np.zeros(len(grid))
    if vol_col is not None:
        out = out + grid[vol_col].to_numpy(float)
    if g:
        out = out + g * grid["C_cross"].to_numpy(float)
    slack = np.clip(1.0 - grid[tail_col].to_numpy(float), 1e-12, None)
    return out + beta * np.log(slack)


def select(grid: pd.DataFrame, var: Variant, **kw) -> pd.Series:
    """The cell a variant picks, with the criterion's value attached."""
    obj = objective(grid, var, **kw)
    row = grid.iloc[int(np.argmax(obj))].copy()
    row["objective"] = float(obj.max())
    return row


def spearman(grid: pd.DataFrame, var: Variant, **kw) -> float:
    """Rank correlation between the criterion and headline TWA over all cells."""
    from scipy.stats import spearmanr

    ok = grid["TWA"].notna().to_numpy()
    if ok.sum() < 3:
        return np.nan
    return float(spearmanr(objective(grid, var, **kw)[ok], grid["TWA"].to_numpy()[ok]).statistic)


# ------------------------------------------------------------------------------- the checks
def check_unit_gram(seed: int = 0) -> None:
    """The reduceat block sums must equal a brute-force mean over pairs."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(40, 5))
    K = np.exp(-0.5 * ((X[:, None] - X[None]) ** 2).sum(-1))
    unit = np.repeat(np.arange(8), 5)
    G, counts = unit_gram(K, unit)
    for u in range(8):
        iu = np.flatnonzero(unit == u)
        for v in range(8):
            iv = np.flatnonzero(unit == v)
            blk = K[np.ix_(iu, iv)]
            want = (blk.sum() - np.trace(blk)) / (len(iu) * (len(iu) - 1)) if u == v else blk.mean()
            assert abs(G[u, v] - want) < 1e-12, f"unit_gram wrong at ({u},{v})"
    assert (counts == 5).all()
    # The whole point of the i != j diagonal: a vanishing bandwidth must send V to 0.
    K_narrow = np.eye(40)
    G_n, _ = unit_gram(K_narrow, unit)
    assert abs(volume(G_n, (1e-3,))[1e-3]) < 1e-9, "V does not vanish for K = I"
    print("unit_gram OK: block means match brute force; V(K = I) = 0")


def check_negatives(cal: Calibration) -> None:
    """Feeding the *matched* pairs through the negative path must recover the LOO scores."""
    mult_o, mult_A = 0.5, 2.0  # off the median heuristic, so no factor is accidentally 1
    sig_o, sig_A = mult_o * cal.sigma_o, mult_A * cal.sigma_A
    K = np.exp(-0.5 * (cal.D_o / sig_o**2 + cal.D_A / sig_A**2))
    lam_m = LAM * cal.m
    L = np.linalg.cholesky(K + lam_m * np.eye(cal.m))
    L_inv = solve_triangular(L, np.eye(cal.m), lower=True)
    blocks, block_of = blocks_of(cal.episode)
    q_ep, _ = block_loo(L_inv, blocks, lam_m)
    ref = ldk._block_loo(L_inv, cal.episode, lam_m)
    assert np.abs(q_ep - ref).max() < 1e-9, "block_loo disagrees with logdet_kernel._block_loo"
    take = np.arange(0, cal.m, max(1, cal.m // 200))
    hold = np.c_[block_of[take], block_of[take]]  # one episode held out: the LOO case
    got = probe_scores(cal.D_o, cal.D_A, take, take, hold, L_inv, blocks, sig_o, sig_A)
    err = float(np.abs(got - q_ep[take]).max())
    assert err < 1e-8, f"matched probes differ from the leave-episode-out scores by {err:.2e}"
    # Two episodes held out has no closed form to check against, so it is checked against a
    # refit with both deleted -- the same brute force logdet_kernel.check_block_loo uses.
    K = np.exp(-0.5 * (cal.D_o / sig_o**2 + cal.D_A / sig_A**2))
    pairs = [(0, 1), (1, len(blocks) - 1)]
    src = np.array([blocks[a][0] for a, _ in pairs])
    don = np.array([blocks[b][-1] for _, b in pairs])
    got2 = probe_scores(cal.D_o, cal.D_A, src, don, np.array(pairs), L_inv, blocks, sig_o, sig_A)
    for p, (a, b) in enumerate(pairs):
        out = np.setdiff1d(np.arange(cal.m), np.r_[blocks[a], blocks[b]])
        k = np.exp(-0.5 * (cal.D_o[src[p], out] / sig_o**2 + cal.D_A[don[p], out] / sig_A**2))
        M = K[np.ix_(out, out)] + lam_m * np.eye(len(out))
        want = 1.0 - float(k @ np.linalg.solve(M, k))
        assert abs(got2[p] - want) < 1e-8, f"two-block probe off by {abs(got2[p] - want):.2e}"
    print(f"probes OK: matched probes reproduce the leave-episode-out scores to {err:.1e}; "
          f"two-episode holdout matches a refit with both deleted")


def check_degeneracy(cal: Calibration) -> None:
    """At a huge sigma_A the action channel is inert, so C must be exactly chance."""
    neg = make_negatives(cal, 200)
    units = {w: unit_index(cal.episode, w) for w in WIDTHS}
    # Exactly at the limit the action factor is identically 1, so a swapped point's kernel
    # row *is* its matched control's: every pair ties and C is exactly 0.
    inert = cell_stats(cal, 1.0, np.inf, neg, units, alphas=(0.05,), windows=(5,))
    assert inert["C_cross"] == 0.0, f"C is {inert['C_cross']:.3g}, not 0, at sigma_A = inf"
    assert inert["auc_cross"] == 0.5, f"AUC is {inert['auc_cross']:.4f}, not chance, at sigma_A = inf"
    # ... and it has to *decay* to that, which is why C is the mean gap rather than the AUC:
    # the AUC is scale-free, so at sigma_A = 4096x it reads 1.0 off a 1e-7 separation and
    # would rate the inert action channel as informative as any other.
    wide = cell_stats(cal, 1.0, 2.0**12, neg, units, alphas=(0.05,), windows=(5,))
    assert abs(wide["C_cross"]) < 1e-4, f"C is {wide['C_cross']:.3g} at sigma_A = 4096x"
    narrow = cell_stats(cal, 1.0, 2.0**-8, neg, units, alphas=(0.05,), windows=(5,))
    assert abs(narrow["C_cross"]) < 0.05, f"C is {narrow['C_cross']:.3g} at sigma_A = 1/256x"
    floor = cell_stats(cal, 2.0**-8, 2.0**-8, neg, units, alphas=(0.05,), windows=(5,))
    v = floor[f"V@w0@r{1e-3:g}"]
    assert v < 1e-3, f"V is {v:.3g}, not ~0, at sigma -> 0"
    print(f"degeneracies OK: C = {inert['C_cross']:.1g} at sigma_A = inf, "
          f"{wide['C_cross']:.1e} at 4096x, {narrow['C_cross']:.1e} at 1/256x "
          f"(AUC there: {inert['auc_cross']:.2f}, {wide['auc_cross']:.2f}, {narrow['auc_cross']:.2f});"
          f"  V = {v:.1e} at sigma = 1/256x")


# ------------------------------------------------------------------------------ reporting
def load_grid(task: str, method: str, seed: int, force: bool, n_neg: int
              ) -> tuple[pd.DataFrame, Calibration | None]:
    """The criterion grid, the cached TWA landscape and the kernel factors, on one frame."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{task}__{method}__traj__seed{seed}.csv"
    if path.exists() and not force:
        crit = pd.read_csv(path)
        cal = None
    else:
        cal = ldk.prepare(task, method, seed)
        print(f"{task}: m = {cal.m} steps over {len(np.unique(cal.episode))} episodes; "
              f"median heuristic sigma_o {cal.sigma_o:.4g} sigma_A {cal.sigma_A:.4g}", flush=True)
        crit = traj_grid(cal, n_neg)
        crit.to_csv(path, index=False)
        print(f"criterion grid -> {os.path.relpath(path, ROOT_DIR)}")
    fac = pd.read_csv(CACHE_DIR / f"{task}__{method}__factors__seed{seed}.csv")
    cells = ldk.twa_landscape(task, method, seed, cache_only=True)
    df = crit.merge(fac, on=["exp_o", "exp_A"], how="left")
    df = df.merge(ldk.reduce_landscape(cells).drop(columns=["mult_o", "mult_A"]),
                  on=["exp_o", "exp_A"], how="left")
    return df, cal


def knob_defaults(alpha: float) -> dict:
    return {"alpha": alpha, "beta": BETA_DEFAULT, "h": H_DEFAULT,
            "w": WIDTH_DEFAULT, "rho": RHOS[1]}


def balanced_knobs(df: pd.DataFrame, alpha: float, h: int, w: int, rho: float
                   ) -> tuple[float, float]:
    """``(beta, gamma)`` making all three terms equal in median magnitude over the grid.

    A runnable rule, unlike the sweep's argmax: it reads only the criterion's own values, so
    it needs no TWA and could be applied to a new task. It exists because the log barrier is
    unbounded below while V and C are bounded, so ``beta = 1`` cannot be neutral -- the
    question is only whether the ratio is *stable across tasks*, which is what
    ``logdet_kernel`` found the original two terms failing at by 200x.
    """
    vol = float(np.median(np.abs(df[f"V@w{w}@r{rho:g}"].to_numpy(float))))
    con = float(np.median(np.abs(df["C_cross"].to_numpy(float))))
    bar = float(np.median(np.abs(np.log(np.clip(
        1.0 - df[f"cvar_g@{alpha:g}@H{h}"].to_numpy(float), 1e-12, None)))))
    return vol / max(bar, 1e-12), vol / max(con, 1e-12)


def report_ladder(df: pd.DataFrame, task: str, method: str, seed: int, alpha: float) -> list[dict]:
    """The ablation ladder: what each variant picks, what it scores, how well it ranks."""
    med = df[(df["exp_o"] == 0) & (df["exp_A"] == 0)].iloc[0]
    best = df.loc[df["TWA"].idxmax()]
    kw = knob_defaults(alpha)
    print(f"\n=== {method} on {task}: the ladder (alpha = {alpha:g}, beta = {BETA_DEFAULT:g}, "
          f"gamma = {GAMMA_DEFAULT:g}, H = {H_DEFAULT}, whole-episode units) ===")
    rows = [{"Task": task, "Method": method, "Seed": seed, "Variant": "--",
             "Rule": "median heuristic", "mult_o": 1.0, "mult_A": 1.0, "k_A": med["k_A"],
             "TWA": med["TWA"], "dTWA": 0.0, "Accuracy": med["Accuracy"],
             "Det. Time": med["Det. Time"], "spearman": np.nan}]
    b_bal, g_bal = balanced_knobs(df, alpha, kw["h"], kw["w"], kw["rho"])
    for var in LADDER:
        extra = ((("A6", dict(kw, beta=b_bal, gamma=g_bal)),) if var.key == "A4" else ())
        for key, k in ((var.key, kw),) + extra:
            pick = select(df, var, **k)
            label = (var.label if key == var.key else
                     f"{var.label}, beta = {b_bal:.3g} gamma = {g_bal:.3g} (balanced)")
            rows.append({"Task": task, "Method": method, "Seed": seed, "Variant": key,
                         "Rule": label, "mult_o": pick["mult_o"], "mult_A": pick["mult_A"],
                         "k_A": pick["k_A"], "TWA": pick["TWA"], "dTWA": pick["TWA"] - med["TWA"],
                         "Accuracy": pick["Accuracy"], "Det. Time": pick["Det. Time"],
                         "spearman": spearman(df, var, **k)})
    rows.append({"Task": task, "Method": method, "Seed": seed, "Variant": "oracle",
                 "Rule": "best bandwidths in grid", "mult_o": best["mult_o"],
                 "mult_A": best["mult_A"], "k_A": best["k_A"], "TWA": best["TWA"],
                 "dTWA": best["TWA"] - med["TWA"], "Accuracy": best["Accuracy"],
                 "Det. Time": best["Det. Time"], "spearman": np.nan})
    out = pd.DataFrame(rows)
    print(out.drop(columns=["Task", "Method", "Seed"]).to_string(
        index=False, float_format=lambda v: f"{v:.4f}", na_rep="  --  "))
    print("  k_A ~ 1 means the action factor is uniformly 1, i.e. the action channel is inert")
    return rows


def report_knobs(df: pd.DataFrame, task: str, alpha: float) -> pd.DataFrame:
    """Sweeps over every knob of the full criterion, one at a time from the defaults."""
    kw = knob_defaults(alpha)
    var = LADDER[4]  # A4, the full criterion
    rows = []
    for name, values in (("alpha", ALPHAS), ("beta", BETAS), ("gamma", GAMMAS),
                         ("h", WINDOWS), ("w", WIDTHS), ("rho", RHOS)):
        for v in values:
            k = dict(kw, **{name: v})
            pick = select(df, var, **k)
            rows.append({"knob": name, "value": v, "mult_o": pick["mult_o"],
                         "mult_A": pick["mult_A"], "k_A": pick["k_A"], "TWA": pick["TWA"],
                         "spearman": spearman(df, var, **k)})
    out = pd.DataFrame(rows)
    print(f"\n=== {task}: one knob at a time, from the defaults (variant A4) ===")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}", na_rep="  --  "))
    return out


def report_balance(df: pd.DataFrame, alpha: float) -> dict:
    """How large the three terms are relative to each other over the grid.

    ``logdet_kernel`` found the beta balancing its two terms spanning 200x across tasks,
    which is why beta = 1 was not a balance at all. The rewritten terms are all
    dimensionless, so the same measurement is the test of whether that is fixed.
    """
    kw = knob_defaults(alpha)
    vol = df[f"V@w{kw['w']}@r{kw['rho']:g}"].to_numpy(float)
    con = df["C_cross"].to_numpy(float)
    bar = np.log(np.clip(1.0 - df[f"cvar_g@{alpha:g}@H{kw['h']}"].to_numpy(float), 1e-12, None))
    old = df["logdet"].to_numpy(float)
    old_bar = np.log(np.clip(1.0 - df[f"cvar_s@{alpha:g}"].to_numpy(float), 1e-12, None))
    b_bal, g_bal = balanced_knobs(df, alpha, kw["h"], kw["w"], kw["rho"])
    out = {
        "median |V|": float(np.median(np.abs(vol))),
        "median |C|": float(np.median(np.abs(con))),
        "median |barrier| at beta=1": float(np.median(np.abs(bar))),
        "beta balancing V": b_bal,
        "gamma balancing C": g_bal,
        "beta balancing the old logdet (for reference)":
            float(np.median(np.abs(old)) / max(np.median(np.abs(old_bar)), 1e-12)),
    }
    print(f"\n=== how the terms compare (alpha = {alpha:g}) ===")
    for k, v in out.items():
        print(f"  {k:<50} {v:.3f}")
    return out


# ---------------------------------------------------------------------------------- plots
def plot_landscape(df: pd.DataFrame, task: str, method: str, alpha: float, stem: str) -> None:
    """The rewritten criterion's three terms beside the TWA landscape they stand in for."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kw = knob_defaults(alpha)
    a0, a4 = LADDER[0], LADDER[4]
    old, new = select(df, a0, **kw), select(df, a4, **kw)
    best = df.loc[df["TWA"].idxmax()]
    marks = [
        (0.0, 0.0, "X", ldk._INK, "median heuristic"),
        (old["exp_o"], old["exp_A"], "o", ldk._THIRD, "original criterion (A0)"),
        (new["exp_o"], new["exp_A"], "o", ldk._ACCENT, "rewritten criterion (A4)"),
        (best["exp_o"], best["exp_A"], "s", ldk._SECOND, "best TWA in the grid"),
    ]
    panels = [
        (df[f"V@w{kw['w']}@r{kw['rho']:g}"].to_numpy(float), "Blues", r"$V$", "volume, per episode"),
        (df["C_cross"].to_numpy(float), "Blues", r"$C$", "contrast (cross-episode swaps)"),
        (objective(df, a4, **kw), "Blues", r"$J(\theta)$", "the criterion, A4"),
        (df["TWA"].to_numpy(float), "Greens", "TWA", "what it is standing in for"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.9))
    for ax, (vals, cmap, lab, title) in zip(axes, panels):
        ldk._field(ax, df, vals, cmap, lab, floor_pct=20.0 if lab.startswith("$J") else None)
        ax.set_title(title, color=ldk._INK, fontsize=11, loc="left")
        ldk._marks(ax, marks)
    caption = textwrap.wrap(
        f"{len(df)} cells, the same grid and the same cached FIPER evaluations as "
        f"logdet_kernel. Headline TWA at the median heuristic {df[(df['exp_o'] == 0) & (df['exp_A'] == 0)].iloc[0]['TWA']:.3f}, "
        f"at the original criterion's pick {old['TWA']:.3f}, at the rewritten one's "
        f"{new['TWA']:.3f}, at the best cell {best['TWA']:.3f}. C = 1/2 along the top edge is "
        f"the action channel going inert, which is the degeneracy the contrast term exists to "
        f"reject.", width=132)
    fig.tight_layout(rect=(0, 0, 1, 0.85 - 0.030 * len(caption)), w_pad=3.0)
    fig.text(0.005, 0.99, f"The trajectory-aware criterion  --  {method} on {task}",
             color=ldk._INK, fontsize=13, ha="left", va="top")
    fig.text(0.005, 0.93, "\n".join(caption), color=ldk._MUTED, fontsize=9.5, ha="left",
             va="top", linespacing=1.5)
    leg = fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right",
                     bbox_to_anchor=(0.995, 1.0), fontsize=9, framealpha=0.0,
                     labelcolor=ldk._MUTED)
    leg.set_frame_on(False)
    ldk._save(fig, stem)
    plt.close(fig)


def plot_ladder(summary: pd.DataFrame, alpha: float, stem: str) -> None:
    """The deliverable: what each rung of the ladder does to the headline, on every task."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = list(dict.fromkeys(summary["Task"]))
    keys = [k for k in list(dict.fromkeys(summary["Variant"])) if k not in ("--", "oracle")]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    width = 0.8 / len(keys)
    cmap = plt.get_cmap("Blues")
    for k, key in enumerate(keys):
        sub = summary[summary["Variant"] == key].set_index("Task")
        x = np.arange(len(tasks)) + (k - (len(keys) - 1) / 2) * width
        axes[0].bar(x, [sub.loc[t, "dTWA"] for t in tasks], width,
                    color=cmap(0.30 + 0.10 * k), label=key,
                    edgecolor=ldk._SURFACE, linewidth=0.5)
        axes[1].bar(x, [sub.loc[t, "spearman"] for t in tasks], width,
                    color=cmap(0.30 + 0.10 * k), edgecolor=ldk._SURFACE, linewidth=0.5)
    orc = summary[summary["Variant"] == "oracle"].set_index("Task")
    axes[0].plot(np.arange(len(tasks)), [orc.loc[t, "dTWA"] for t in tasks], "s",
                 color=ldk._SECOND, markersize=8, markerfacecolor="none", markeredgewidth=2.0,
                 linestyle="none", label="oracle (best cell)")
    for ax, title, ylab in (
            (axes[0], "effect on the headline", r"$\Delta$TWA vs the median heuristic"),
            (axes[1], "is the criterion a proxy for TWA at all?", "Spearman rank correlation")):
        ax.axhline(0.0, color=ldk._MUTED, linewidth=0.8)
        ax.set_xticks(np.arange(len(tasks)))
        ax.set_xticklabels(tasks, rotation=15, ha="right")
        ax.set_title(title, color=ldk._INK, fontsize=11, loc="left")
        ax.set_ylabel(ylab, color=ldk._MUTED, fontsize=9)
        ldk._style(ax)
    labels = "   ".join(f"{v.key}: {v.label}" for v in LADDER) + \
        "   A6: A4 with beta set by the term-balance rule"
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.text(0.005, 0.99, "Fixing the criterion for pooled trajectory data, one term at a time",
             color=ldk._INK, fontsize=13, ha="left", va="top")
    fig.text(0.005, 0.925, "\n".join(textwrap.wrap(labels, width=140)),
             color=ldk._MUTED, fontsize=9, ha="left", va="top", linespacing=1.6)
    leg = fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right",
                     bbox_to_anchor=(0.995, 1.0), fontsize=8.5, ncol=4, framealpha=0.0,
                     labelcolor=ldk._MUTED)
    leg.set_frame_on(False)
    ldk._save(fig, stem)
    plt.close(fig)


def plot_knobs(knobs: dict, alpha: float, stem: str) -> None:
    """Every knob of the full criterion, swept one at a time, against TWA and rank correlation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["alpha", "beta", "gamma", "h", "w", "rho"]
    fig, axes = plt.subplots(2, len(names), figsize=(3.0 * len(names), 6.4), sharey="row")
    colors = [ldk._ACCENT, ldk._SECOND, ldk._THIRD, ldk._INK, ldk._MUTED]
    for col, name in enumerate(names):
        for c, (task, df) in zip(colors, knobs.items()):
            sub = df[df["knob"] == name]
            xs = np.arange(len(sub))
            axes[0, col].plot(xs, sub["TWA"], "-o", color=c, markersize=3.5, label=task)
            axes[1, col].plot(xs, sub["spearman"], "-o", color=c, markersize=3.5)
            axes[0, col].set_xticks(xs)
            axes[0, col].set_xticklabels([f"{v:g}" for v in sub["value"]], fontsize=7, rotation=45)
            axes[1, col].set_xticks(xs)
            axes[1, col].set_xticklabels([f"{v:g}" for v in sub["value"]], fontsize=7, rotation=45)
        axes[0, col].set_title(name, color=ldk._INK, fontsize=10, loc="left")
        for r in (0, 1):
            ldk._style(axes[r, col])
    axes[1, 0].axhline(0.0, color=ldk._MUTED, linewidth=0.8)
    axes[0, 0].set_ylabel("TWA of the pick", color=ldk._MUTED, fontsize=9)
    axes[1, 0].set_ylabel("Spearman vs TWA", color=ldk._MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.text(0.005, 0.99, "One knob at a time, from the defaults  --  variant A4",
             color=ldk._INK, fontsize=13, ha="left", va="top")
    fig.text(0.005, 0.935, f"w = 0 is whole-episode units, w = 1 a step-level determinant; "
                           f"defaults alpha = {alpha:g}, beta = {BETA_DEFAULT:g}, "
                           f"gamma = {GAMMA_DEFAULT:g}, H = {H_DEFAULT}, rho = {RHOS[1]:g}.",
             color=ldk._MUTED, fontsize=9, ha="left", va="top")
    leg = fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right",
                     bbox_to_anchor=(0.995, 1.0), fontsize=9, ncol=5, framealpha=0.0,
                     labelcolor=ldk._MUTED)
    leg.set_frame_on(False)
    ldk._save(fig, stem)
    plt.close(fig)


# ----------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="trajectory_criterion", description=__doc__)
    parser.add_argument("--task", nargs="+", default=["pretzel"], choices=ldk.run_mod.ALL_TASKS)
    parser.add_argument("--method", nargs="+", default=["kern_cd_flat"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--n-neg", type=int, default=N_NEG)
    parser.add_argument("--force", action="store_true", help="ignore the criterion cache")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    check_unit_gram()
    if args.check_only:
        cal = ldk.prepare(args.task[0], args.method[0], args.seed)
        print(f"{args.task[0]}: m = {cal.m} steps over {len(np.unique(cal.episode))} episodes")
        check_negatives(cal)
        check_degeneracy(cal)
        return 0

    rows, knobs = [], {}
    for task in args.task:
        for method in args.method:
            df, _ = load_grid(task, method, args.seed, args.force, args.n_neg)
            rows += report_ladder(df, task, method, args.seed, args.alpha)
            knobs[task] = report_knobs(df, task, args.alpha)
            report_balance(df, args.alpha)
            if not args.no_plots:
                plot_landscape(df, task, method, args.alpha, f"{task}__{method}__traj__landscape")

    summary = pd.DataFrame(rows)
    print("\n=== summary: the effect of each rung on the headline benchmark scores ===")
    print(summary.drop(columns=["Method", "Seed"]).to_string(
        index=False, float_format=lambda v: f"{v:.4f}", na_rep="  --  "))
    path = CACHE_DIR / "summary__traj.csv"
    summary.to_csv(path, index=False)
    print(f"\n-> {os.path.relpath(path, ROOT_DIR)}")
    if not args.no_plots and summary["Task"].nunique() > 1:
        plot_ladder(summary, args.alpha, "summary__traj__ladder")
        plot_knobs(knobs, args.alpha, "summary__traj__knobs")
    print(f"total wall clock: {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
