"""Pilot: is any signature-based score competitive, and what does it cost?

Runs the three score families over a small grid of truncation levels on the
cheap tasks, reporting best-TWA (full FIPER threshold grid) and episode AUROC
alongside scoring wall-clock.
"""
from __future__ import annotations

import sys, time

sys.path.insert(0, "/home/louis/fiper/my_experiments/sig_study")

import numpy as np
import pandas as pd
import torch

from loader import load_task, episodes
from harness import evaluate, dataset_stats, best_row, episode_auroc
from scores import SigConfig, sig_features
from episode_scores import sigvar_episode, sigtc_episode, sigmmd_episode

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def task_meta(ds):
    md = ds.data["metadata"]
    exec_h = md["actions"].get("action_execution_horizon") or 4
    pred_h = md["actions"]["action_prediction_horizon"]
    return exec_h, pred_h, min(1, pred_h // exec_h - 1)


def build_reference(ds, actions, n_ref=256, seed=0):
    """Pooled reference chunk set from the *successful* calibration rollouts."""
    rng = np.random.default_rng(seed)
    pool = []
    for ap, ok in episodes(ds, "calibration", actions):
        if ok:
            pool.append(ap.reshape(-1, ap.shape[-2], ap.shape[-1]))
    pool = np.concatenate(pool, 0)
    idx = rng.choice(len(pool), size=min(n_ref, len(pool)), replace=False)
    return torch.tensor(pool[idx], dtype=torch.float32, device=DEV)


def score_all(ds, family, cfg, actions, exec_h, backtrack, ref=None, ref_feats=None):
    out = {}
    for subset in ("calibration", "test"):
        scores, succ = [], []
        for ap, ok in episodes(ds, subset, actions):
            x = torch.tensor(ap, dtype=torch.float32, device=DEV)
            if family == "sigvar":
                s = sigvar_episode(x, cfg)
            elif family == "sigtc":
                s = sigtc_episode(x, cfg, exec_h, backtrack)
            elif family == "sigmmd":
                s = sigmmd_episode(x, ref, cfg, ref_feats=ref_feats)
            else:
                raise ValueError(family)
            scores.append(np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0))
            succ.append(ok)
        out[subset] = (scores, succ)
    return out


def run_task(task, levels=(1, 2, 3, 4), families=("sigvar", "sigtc", "sigmmd"),
             actions=("position",), kernels=("lin",)):
    ds = load_task(task)
    exec_h, pred_h, backtrack = task_meta(ds)
    stats = dataset_stats(ds)
    rows = []
    for kernel in kernels:
        for level in levels:
            cfg = SigConfig(level=level, kernel=kernel, time_aug=True)
            ref = build_reference(ds, actions)
            ref_feats = sig_features(ref, cfg) if kernel == "lin" else None
            for fam in families:
                t0 = time.time()
                res = score_all(ds, fam, cfg, actions, exec_h, backtrack, ref, ref_feats)
                st = time.time() - t0
                t1 = time.time()
                r = evaluate(*res["calibration"], res["test"][0], stats)
                et = time.time() - t1
                b = best_row(r)
                a = ds.data["action_preds"].shape[-1] if actions == ("all",) else 3
                rows.append({
                    "task": task, "family": fam, "kernel": kernel, "level": level,
                    "feat_dim": cfg.feat_dim(a),
                    "best_TWA": b["TWA"], "TPR": b["TPR"], "TNR": b["TNR"],
                    "at": f"{b['threshold']}/q{b['quantile']}/w{b['window']}",
                    "auroc": episode_auroc(res["test"][0], res["test"][1], 1),
                    "score_s": round(st, 1), "eval_s": round(et, 1),
                })
                print(f"  {task:10s} {fam:7s} {kernel} L{level} d={rows[-1]['feat_dim']:5d} "
                      f"TWA {b['TWA']:.3f} auroc {rows[-1]['auroc']:.3f} "
                      f"score {st:.1f}s eval {et:.1f}s", flush=True)
    return rows


if __name__ == "__main__":
    tasks = sys.argv[1:] or ["push_chair", "pretzel", "sorting", "push_t"]
    allr = []
    for t in tasks:
        allr += run_task(t)
    df = pd.DataFrame(allr)
    df.to_csv("/home/louis/fiper/my_experiments/sig_study/pilot.csv", index=False)
    print(df.to_string())
