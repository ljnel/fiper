"""Can the log-determinant criterion pick ``sigma_phi``? -- a small follow-up to ``logdet_kernel.py``.

That study swept the two bandwidths of the product kernel and left ``sigma_phi`` -- the
width *inside* the RFF map, on chunk features -- at its median heuristic throughout. It is
the one bandwidth the criterion was never asked about, and the one where the median
heuristic is least obviously right: its value spans five orders of magnitude across tasks
(748 on push_t, 0.0042 on sorting) where every other bandwidth stays within a factor of
~20. So if the criterion can see anything the median heuristic cannot, this is where.

The test
--------
``sigma_o`` is pinned at its median heuristic and the grid is ``(sigma_phi, sigma_A)``, 9 x 9
octaves over 2^-4 .. 2^4, on push_t and sorting. Those two are the extremes of the
``sigma_phi`` range and both have a TWA landscape with real headroom.

``sigma_phi`` and ``sigma_A`` have to move together, which is why this is a grid and not a
line: ``mu_A`` is built by ``phi``, so changing ``sigma_phi`` changes the mean embeddings and
therefore the median heuristic for ``sigma_A`` itself. ``mult_A = 1`` here means "the median
heuristic recomputed at this ``sigma_phi``", on both the criterion side and the evaluation
side, so the two never disagree about what the baseline is.

``hyperparam_sensitivity`` already reports that ``sigma_phi`` moves TWA by only 0.011 over a
64x range on push_t, with the optimum at 1x -- but one-at-a-time, on ``kern_cd_sum``, on the
task where ``sigma_phi`` is largest. This asks the same question jointly with ``sigma_A``, on
``kern_cd_flat``, including the task at the other extreme.

What it found
-------------
**``sigma_phi`` matters more than push_t suggests, and push_t was the wrong task to ask
on.** Holding ``sigma_A`` at its median heuristic, ``sigma_phi``'s own range over 2^-4..2^4
is 0.069 TWA on stacking, 0.066 on push_chair, 0.044 on sorting and 0.021 on pretzel --
against 0.0085 on push_t. So the one bandwidth previously reported as inert is not inert;
it was measured on the least sensitive of the five tasks.

**But the criterion still cannot pick it, and fails the same way as before.** At
``beta = 1`` it selects the grid's *smallest* ``sigma_phi``, 2^-4 x the median heuristic, on
**five tasks out of five**. That is an edge solution, not an interior optimum -- the
criterion would go lower if the grid let it. It is the log-determinant's generic preference
for narrowness (finding 2 of ``logdet_kernel``) reappearing on a third axis, with the
barrier again too weak at ``beta = 1`` to oppose it.

Whether that fixed preference helps is task luck:

    pretzel -0.035   push_t -0.004   push_chair +0.036   sorting +0.020   stacking -0.069

mean -0.010 TWA, 2 wins of 5 -- indistinguishable from the -0.011 and 1 of 5 that the
``(sigma_o, sigma_A)`` grid produced. The two wins are the tasks whose true optimum happens
to sit at small ``sigma_phi``; the two large losses are the tasks where the median heuristic
was already at or beside the optimum (stacking's ``sigma_phi``-alone optimum is 1.0x,
pretzel's best cell is at 2x) and where the whole plane holds almost no headroom anyway
(+0.0009 and +0.0006). Picking the edge costs most exactly where the heuristic was right.

So the recommendation this file was written to test does not hold: ``sigma_phi`` is a more
interesting axis than expected, but not one this criterion can navigate.

**Caveat.** ``sigma_o`` is pinned throughout, so this says nothing about a joint
``(sigma_o, sigma_A, sigma_phi)`` search; and one seed, as everywhere in this experiment.

Running it
----------
    pixi run python my_experiments/logdet_kernel/phi_probe.py              # push_t + sorting
    pixi run python my_experiments/logdet_kernel/phi_probe.py --report     # cached cells only

Cells cache under ``cache/`` as ``<task>__ldkphi_*``; FIPER's own per-arm litter lands in
``data/<task>/results/ldkphi_*`` and is removable with ``rm -rf data/*/results/ldkphi_*``.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

# Same bootstrap as logdet_kernel: the sibling import below needs the repo root on the path
# before it can run, so this cannot be deferred to that module's own bootstrap.
if str(pathlib.Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from my_experiments.logdet_kernel.logdet_kernel import (  # noqa: E402
    ALPHA_DEFAULT,
    CACHE_DIR,
    HERE,
    LAM,
    N_COMPONENTS,
    ROOT_DIR,
    _INK,
    _MUTED,
    _ACCENT,
    _SECOND,
    _mean_off_diag,
    _save,
    _style,
    cvar,
    headline,
    prepare,
    run_mod,
)
from my_methods.base import REGISTRY, discover  # noqa: E402

#: Octaves, not half-octaves: this is a probe, and 81 cells is enough to see a trend.
EXPONENTS = tuple(float(e) for e in range(-4, 5))

#: Where sigma_phi is largest and smallest, and both discriminate.
TASKS = ("push_t", "sorting")

BETAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
GROUP = 16


# ---------------------------------------------------------------------------- the arms
class _PhiScaleMixin:
    """Scales ``sigma_phi`` and ``sigma_A`` by fixed multiples of their median heuristic.

    ``_fit_phi`` runs once per fit and sets ``sigma_phi`` immediately before drawing the RFF
    weights, so the rescaling goes there, in the form ``hyperparam_sensitivity`` verifies
    against sklearn: ``W(c*sigma) == W(sigma)/c`` with the offsets untouched. ``sigma_A`` is
    then taken by the unmodified ``fit``, on the mean embeddings that this ``phi`` produced,
    so its multiplier is relative to the median heuristic *at this* ``sigma_phi``.
    """

    ldk_mult_phi = 1.0
    ldk_mult_A = 1.0

    def _fit_phi(self, parts) -> None:
        super()._fit_phi(parts)
        c = float(self.ldk_mult_phi)
        if c != 1.0:
            self._sigma_phi = float(self._sigma_phi) * c
            self._phi_W = self._phi_W / c

    def _pack(self, obs, mu):
        if not getattr(self, "_ldkphi_done", False):
            self._ldkphi_done = True
            if mu is not None:
                self._sigma_A = float(self._sigma_A) * float(self.ldk_mult_A)
        return super()._pack(obs, mu)


def arm_name(exp_phi: float, exp_A: float) -> str:
    return f"ldkphi_flat__P{exp_phi:+g}__A{exp_A:+g}"


def make_arm(exp_phi: float, exp_A: float):
    name = arm_name(exp_phi, exp_A)
    if name in REGISTRY:
        return REGISTRY[name]
    base = REGISTRY["kern_cd_flat"]
    return type(
        "PhiCell" + name.replace("ldkphi_flat__", "").replace("+", "p").replace("-", "m").replace(".", "d"),
        (_PhiScaleMixin, base),
        {"name": name, "params": dict(base.params, lam=LAM, n_components=N_COMPONENTS),
         "ldk_mult_phi": 2.0**exp_phi, "ldk_mult_A": 2.0**exp_A},
    )


# ------------------------------------------------------------------------ the criterion
def criterion_grid(task: str, seed: int, alpha: float, blocked: bool) -> pd.DataFrame:
    """Criterion values over ``(sigma_phi, sigma_A)``, with ``sigma_o`` at its median heuristic.

    One ``prepare`` per ``sigma_phi``: the mean embeddings, and hence ``D_A`` and the median
    heuristic for ``sigma_A``, are all downstream of ``phi``, so they cannot be cached across
    the outer loop the way ``logdet_kernel``'s two-bandwidth grid caches them.
    """
    rows = []
    for ep in EXPONENTS:
        cal = prepare(task, "kern_cd_flat", seed, mult_phi=2.0**ep)
        for ea in EXPONENTS:
            logdet, loo = cal.cell(1.0, 2.0**ea, blocked=blocked)
            c, t = cvar(loo, alpha)
            rows.append({"exp_phi": ep, "exp_A": ea, "mult_phi": 2.0**ep, "mult_A": 2.0**ea,
                         "sigma_phi": cal.sigma_phi, "sigma_A_median": cal.sigma_A,
                         "logdet": logdet, "cvar": c, "tau": t,
                         "k_A": _mean_off_diag(cal.D_A, 2.0**ea * cal.sigma_A)})
    return pd.DataFrame(rows)


def objective(grid: pd.DataFrame, beta: float) -> np.ndarray:
    return grid["logdet"].to_numpy(float) + beta * np.log(np.clip(1.0 - grid["cvar"].to_numpy(float), 1e-12, None))


# ------------------------------------------------------------------------ the landscape
def twa_landscape(task: str, seed: int, force: bool, cache_only: bool) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    discover()
    cells = [(ep, ea) for ep in EXPONENTS for ea in EXPONENTS]
    have, todo = {}, []
    for ep, ea in cells:
        path = CACHE_DIR / f"{task}__{arm_name(ep, ea)}__seed{seed}.csv"
        if path.exists() and not force:
            have[(ep, ea)] = pd.read_csv(path)
        else:
            todo.append((ep, ea))
    if todo and not cache_only:
        print(f"\nevaluating {len(todo)} cell(s) on {task} ({len(have)} cached) ...")
        started = time.time()
        for i in range(0, len(todo), GROUP):
            batch = todo[i : i + GROUP]
            names = [arm_name(*c) for c in batch]
            for c in batch:
                make_arm(*c)
            results = run_mod._evaluate_task(names, task, seed, {})
            for c, name in zip(batch, names):
                df = pd.DataFrame(run_mod._rows(name, task, results))
                df.to_csv(CACHE_DIR / f"{task}__{name}__seed{seed}.csv", index=False)
                have[c] = df
            print(f"  {min(i + GROUP, len(todo))}/{len(todo)} cells, {time.time() - started:.0f}s elapsed")
    rows = []
    for ep, ea in cells:
        row = {"exp_phi": ep, "exp_A": ea}
        df = have.get((ep, ea))
        row.update(headline(df).to_dict() if df is not None
                   else dict.fromkeys(["TWA", "Accuracy", "Det. Time", "TWA_gridmean"], np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- reporting
def _cell(df, ep, ea):
    return df[(df["exp_phi"] == ep) & (df["exp_A"] == ea)].iloc[0]


def report(task: str, df: pd.DataFrame, alpha: float) -> list[dict]:
    med, best = _cell(df, 0.0, 0.0), df.loc[df["TWA"].idxmax()]
    print(f"\n=== kern_cd_flat on {task}: (sigma_phi, sigma_A) at sigma_o = median heuristic ===")
    print(f"  median heuristic sigma_phi {med['sigma_phi']:.4g}, sigma_A {med['sigma_A_median']:.4g}")
    print(f"  TWA over the {len(df)} cells: {df['TWA'].min():.4f} .. {df['TWA'].max():.4f}   "
          f"(median heuristic cell {med['TWA']:.4f})")
    # sigma_phi's own range, holding sigma_A at its (recomputed) median heuristic
    line = df[df["exp_A"] == 0.0].sort_values("exp_phi")
    print(f"  along sigma_phi alone (sigma_A at 1x): TWA {line['TWA'].min():.4f} .. {line['TWA'].max():.4f}"
          f"  = {line['TWA'].max() - line['TWA'].min():.4f} of range, best at "
          f"{line.loc[line['TWA'].idxmax(), 'mult_phi']:g}x")
    print(f"  best cell in the grid: TWA {best['TWA']:.4f} at sigma_phi {best['mult_phi']:g}x, "
          f"sigma_A {best['mult_A']:g}x  (k_A {best['k_A']:.3f})")

    print(f"\n  what the criterion picks (alpha = {alpha:g}):")
    rows = [{"rule": "median heuristic", "beta": np.nan, "mult_phi": 1.0, "mult_A": 1.0,
             "k_A": med["k_A"], "TWA": med["TWA"], "dTWA": 0.0}]
    for beta in BETAS:
        p = df.iloc[int(np.argmax(objective(df, beta)))]
        rows.append({"rule": "criterion", "beta": beta, "mult_phi": p["mult_phi"], "mult_A": p["mult_A"],
                     "k_A": p["k_A"], "TWA": p["TWA"], "dTWA": p["TWA"] - med["TWA"]})
    rows.append({"rule": "best in grid", "beta": np.nan, "mult_phi": best["mult_phi"],
                 "mult_A": best["mult_A"], "k_A": best["k_A"], "TWA": best["TWA"],
                 "dTWA": best["TWA"] - med["TWA"]})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}", na_rep="  -- "))
    return [dict(r, Task=task) for r in rows]


def plot(task: str, df: pd.DataFrame, alpha: float, beta: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    med, best = _cell(df, 0.0, 0.0), df.loc[df["TWA"].idxmax()]
    pick = df.iloc[int(np.argmax(objective(df, beta)))]
    marks = [(0.0, 0.0, "X", _INK, "median heuristic"),
             (pick["exp_phi"], pick["exp_A"], "o", _ACCENT, rf"criterion ($\beta$={beta:g})"),
             (best["exp_phi"], best["exp_A"], "s", _SECOND, "best TWA in the grid")]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    for ax, (vals, cmap, label, title) in zip(axes, [
        (objective(df, beta), "Blues", r"$J(\theta)$", rf"the criterion ($\alpha$={alpha:g}, $\beta$={beta:g})"),
        (df["TWA"].to_numpy(float), "Greens", "TWA", "headline TWA"),
    ]):
        piv = df.assign(_v=vals).pivot(index="exp_A", columns="exp_phi", values="_v")
        mesh = ax.pcolormesh(piv.columns.to_numpy(float), piv.index.to_numpy(float),
                             piv.to_numpy(float), cmap=cmap, shading="nearest")
        bar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
        bar.set_label(label, color=_MUTED, fontsize=8)
        bar.ax.tick_params(colors=_MUTED, labelsize=7)
        bar.outline.set_visible(False)
        for x, y, marker, color, lab in marks:
            ax.plot(x, y, marker, color=color, markersize=11, markeredgewidth=2.0,
                    markerfacecolor="none" if marker in ("o", "s") else color, label=lab, zorder=5)
        ax.set_xticks(EXPONENTS)
        ax.set_yticks(EXPONENTS)
        ax.set_xticklabels([f"{2.0**e:g}" for e in EXPONENTS], fontsize=7)
        ax.set_yticklabels([f"{2.0**e:g}" for e in EXPONENTS], fontsize=7)
        ax.set_xlabel(r"$\sigma_\varphi$  ($\times$ median heuristic)", color=_MUTED, fontsize=9)
        ax.set_ylabel(r"$\sigma_A$  ($\times$ median heuristic at that $\sigma_\varphi$)",
                      color=_MUTED, fontsize=9)
        ax.set_title(title, color=_INK, fontsize=11, loc="left")
        _style(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.text(0.005, 0.99, f"Can the criterion pick $\\sigma_\\varphi$?  --  kern_cd_flat on {task}",
             color=_INK, fontsize=13, ha="left", va="top")
    fig.text(0.005, 0.92, f"{len(df)} cells, $\\sigma_o$ pinned at its median heuristic. TWA at the "
                          f"median heuristic {med['TWA']:.3f}, at the criterion's pick {pick['TWA']:.3f}, "
                          f"at the best cell {best['TWA']:.3f}.",
             color=_MUTED, fontsize=9, ha="left", va="top")
    leg = fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right",
                     bbox_to_anchor=(0.995, 1.0), fontsize=8.5, labelcolor=_MUTED)
    leg.set_frame_on(False)
    _save(fig, f"{task}__phi_probe")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="phi_probe", description=__doc__)
    parser.add_argument("--task", nargs="+", default=list(TASKS), choices=run_mod.ALL_TASKS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--beta", type=float, default=1.0, help="beta used for the plot's mark")
    parser.add_argument("--loo", default="step", choices=["step", "episode"])
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    started, rows = time.time(), []
    for task in args.task:
        tag = "" if args.loo == "step" else f"__loo_{args.loo}"
        path = CACHE_DIR / f"{task}__phi_criterion{tag}__seed{args.seed}.csv"
        if path.exists() and not args.force:
            crit = pd.read_csv(path)
        else:
            crit = criterion_grid(task, args.seed, args.alpha, blocked=(args.loo == "episode"))
            crit.to_csv(path, index=False)
        df = crit.merge(twa_landscape(task, args.seed, args.force, args.report),
                        on=["exp_phi", "exp_A"], how="left")
        rows += report(task, df, args.alpha)
        if not args.no_plots:
            plot(task, df, args.alpha, args.beta)

    summary = pd.DataFrame(rows)
    out = CACHE_DIR / f"phi_probe_summary__seed{args.seed}.csv"
    summary.to_csv(out, index=False)
    print(f"\n-> {os.path.relpath(out, ROOT_DIR)}\ntotal wall clock: {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
