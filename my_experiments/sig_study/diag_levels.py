"""Why does truncation level stop mattering?

Level k of a signature scales like ||path||^k / k!.  Raw action chunks have a
total variation far below 1, while the appended time channel sweeps a full
unit.  So in a linear kernel the signature is *all time channel*, the action
information sits orders of magnitude below it, and the truncation level becomes
a no-op.  This quantifies that and reports the dilation that fixes it.

Also records where the level tensors stop fitting in memory.
"""
from __future__ import annotations

import sys
from math import factorial as np_math_factorial

sys.path.insert(0, "/home/louis/fiper/my_experiments/sig_study")

import numpy as np
import pandas as pd
import torch

from loader import load_task, episodes, TASKS
from sigtools import augment, signature, sig_dim

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LEVEL = 5
NSAMP = 2000
MEM_BUDGET = 8e9  # bytes we are willing to spend on the level tensors


def sample_chunks(ds, actions, n=NSAMP, seed=0):
    rng = np.random.default_rng(seed)
    pool = []
    for ap, ok in episodes(ds, "calibration", actions):
        pool.append(ap.reshape(-1, ap.shape[-2], ap.shape[-1]))
    pool = np.concatenate(pool, 0)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return torch.tensor(pool[idx], dtype=torch.float64, device=DEV)


def total_variation(paths):
    return (paths[:, 1:] - paths[:, :-1]).norm(dim=-1).sum(-1)


def max_level(d, n, bytes_per=8, budget=MEM_BUDGET):
    """Largest level whose flattened signature batch fits in `budget`."""
    for L in range(1, 12):
        if sig_dim(d, L) * n * bytes_per > budget:
            return L - 1
    return 11


if __name__ == "__main__":
    rows = []
    for task in TASKS:
        ds = load_task(task)
        for actions in (("position",), ("all",)):
            X = sample_chunks(ds, actions)
            a = X.shape[-1]
            tv = total_variation(X)
            tv_med = float(tv.median())
            theta = 1.0 / tv_med if tv_med > 0 else 1.0

            # state-channel-only energies (no time channel to mask them)
            L_ok = min(LEVEL, max_level(a, len(X)))
            lv = signature(X, L_ok, flatten=False)
            e_state = [float(l.pow(2).sum(-1).mean().sqrt()) for l in lv]

            # after dilating the state channels so TV ~ 1
            lv_s = signature(X * theta, L_ok, flatten=False)
            e_scaled = [float(l.pow(2).sum(-1).mean().sqrt()) for l in lv_s]

            row = {"task": task, "actions": actions[0], "a": a,
                   "tv_median": tv_med, "tv_p05": float(tv.quantile(0.05)),
                   "tv_p95": float(tv.quantile(0.95)), "theta": theta,
                   "d_with_time": a + 1,
                   "max_level_8GB": max_level(a + 1, 32),
                   "sigdim_L3": sig_dim(a + 1, 3), "sigdim_L4": sig_dim(a + 1, 4),
                   "level_computed": L_ok}
            for k, (es, esc) in enumerate(zip(e_state, e_scaled), start=1):
                row[f"rms_L{k}"] = es
                row[f"rms_scaled_L{k}"] = esc
            rows.append(row)
            print(f"{task:11s} {actions[0]:8s} a={a:2d} TV med {tv_med:.4g} theta {theta:9.2f} | "
                  f"raw " + " ".join(f"L{k+1} {e:.2e}" for k, e in enumerate(e_state)))
            print(f"{'':11s} {'':8s}         "
                  f"          {'':9s}   | scaled " + " ".join(f"L{k+1} {e:.2e}" for k, e in enumerate(e_scaled)),
                  flush=True)
        del ds
    df = pd.DataFrame(rows)
    df.to_csv("/home/louis/fiper/my_experiments/sig_study/diag_levels.csv", index=False)

    print("\n--- time channel vs action channels (linear kernel energy share) ---")
    for _, r in df.iterrows():
        # time channel contributes exactly 1/k! at level k; actions contribute rms_Lk
        share = [r[f"rms_L{k}"] / (r[f"rms_L{k}"] + 1.0 / float(np_math_factorial(k)))
                 for k in range(1, int(r["level_computed"]) + 1)]
        print(f"  {r['task']:11s} {r['actions']:8s} action share of level energy: "
              + " ".join(f"L{k+1} {s:.1e}" for k, s in enumerate(share)))
