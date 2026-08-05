"""Non-signature controls for the same three score families.

The sweep shows truncation level barely moves TWA.  That could mean "level 2 is
enough" or it could mean "the signature is not doing anything you couldn't get
from the raw chunk vector".  These controls separate the two:

  flat    : the chunk flattened to a (H*a)-vector - no signature at all
  disp    : the chunk displacement (endpoint - start), i.e. exactly signature
            level 1, to check whether anything above level 1 ever pays
  sigL2   : signature truncated at 2 (reference point from the sweep)

Same families, same harness, same threshold grid.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/louis/fiper/my_experiments/sig_study")

import numpy as np
import pandas as pd
import torch

from loader import load_task, episodes, TASKS
from harness import evaluate, dataset_stats, best_row, episode_auroc
from scores import SigConfig
from run_sweep import task_meta, calibration_pool, fit_theta

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/home/louis/fiper/my_experiments/sig_study"


def featurize(ap: torch.Tensor, mode: str, cfg: SigConfig | None = None) -> torch.Tensor:
    """ap: (T,B,H,a) -> (T,B,D)."""
    if mode == "flat":
        return ap.reshape(*ap.shape[:2], -1)
    if mode == "disp":
        return ap[:, :, -1, :] - ap[:, :, 0, :]
    if mode == "sig":
        from episode_scores import _feats
        return _feats(ap, cfg)
    raise ValueError(mode)


def var_episode(F: torch.Tensor) -> np.ndarray:
    """E||f||^2 - ||E f||^2 : kernel variance under a linear kernel."""
    return (F.pow(2).sum(-1).mean(-1) - F.mean(1).pow(2).sum(-1)).cpu().numpy()


def mmd_episode(Fx: torch.Tensor, Fy: torch.Tensor) -> np.ndarray:
    Kxx = Fx @ Fx.transpose(-2, -1)
    Kyy = Fy @ Fy.transpose(-2, -1)
    Kxy = Fx @ Fy.transpose(-2, -1)
    return (Kxx.mean((-2, -1)) + Kyy.mean((-2, -1)) - 2 * Kxy.mean((-2, -1))).cpu().numpy()


def score_ep(ap, family, mode, cfg, exec_h, backtrack, ref_feat=None):
    T, B, H, a = ap.shape
    if family == "sigvar":
        return var_episode(featurize(ap, mode, cfg))
    if family == "sigtc":
        off = exec_h * backtrack
        if T <= backtrack or off >= H:
            return np.zeros(T)
        prev = featurize(ap[:-backtrack, :, off:, :], mode, cfg)
        curr = featurize(ap[backtrack:, :, :H - off, :], mode, cfg)
        return np.concatenate([np.zeros(backtrack), mmd_episode(prev, curr)])
    if family == "sigmmd":
        F = featurize(ap, mode, cfg)
        R = ref_feat.unsqueeze(0).expand(T, *ref_feat.shape)
        return mmd_episode(F, R)
    raise ValueError(family)


def run(task, modes=("flat", "disp", "sig"), level=2):
    ds = load_task(task)
    exec_h, pred_h, bt = task_meta(ds)
    stats = dataset_stats(ds)
    actions = ("position",)
    pool = calibration_pool(ds, actions)
    theta = fit_theta(pool)
    rows = []
    for mode in modes:
        cfg = SigConfig(level=level, kernel="lin", time_aug=True, path_scale=theta)
        # reference features for sigmmd, built from the same calibration pool
        refc = pool[:256]
        ref_feat = featurize(refc.unsqueeze(0), mode, cfg)[0]
        for fam in ("sigvar", "sigtc", "sigmmd"):
            cal, cal_ok, tst, tst_ok = [], [], [], []
            for subset, S, OK in (("calibration", cal, cal_ok), ("test", tst, tst_ok)):
                for apn, ok in episodes(ds, subset, actions):
                    x = torch.tensor(apn, dtype=torch.float32, device=DEV)
                    s = score_ep(x, fam, mode, cfg, exec_h, bt, ref_feat)
                    S.append(np.nan_to_num(np.asarray(s, dtype=np.float64),
                                           nan=0.0, posinf=0.0, neginf=0.0))
                    OK.append(ok)
            r = evaluate(cal, cal_ok, tst, stats)
            b = best_row(r)
            dim = featurize(pool[:2].unsqueeze(0), mode, cfg).shape[-1]
            rows.append({"task": task, "mode": mode, "family": fam, "dim": int(dim),
                         "level": level if mode == "sig" else None,
                         "best_TWA": b["TWA"], "TPR": b["TPR"], "TNR": b["TNR"],
                         "auroc": episode_auroc(tst, tst_ok, 1),
                         "at": f"{b['threshold']}/q{b['quantile']}/w{b['window']}"})
            print(f"  {task:11s} {mode:5s} {fam:7s} dim {int(dim):5d} "
                  f"TWA {b['TWA']:.3f} auroc {rows[-1]['auroc']:.3f}", flush=True)
    del ds
    torch.cuda.empty_cache()
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--level", type=int, default=2)
    args = ap.parse_args()
    allr = []
    for t in args.tasks:
        allr += run(t, level=args.level)
    pd.DataFrame(allr).to_csv(f"{OUT}/baselines.csv", index=False)
    print(f"\nwrote baselines.csv")
