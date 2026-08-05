"""Does a signature-MMD action-chunk score add anything on top of kern_cd?

kern_cd scores the *observation* embeddings; a signature score reads the
*action chunk* distribution.  They see different things, so the interesting
question is not "which is better" but "does OR-ing them beat either alone".

Combination follows EvaluationManager._combine_two_methods exactly: elementwise
max of the threshold-normalised scores for "or", min for "and", matched on
(threshold style, quantile, window).
"""
from __future__ import annotations

import os
import pickle
import sys

sys.path.insert(0, "/home/louis/fiper/my_experiments/sig_study")

import numpy as np
import pandas as pd
import torch

from loader import load_task, episodes, TASKS
from harness import (EVAL_CFG, dataset_stats, evaluate, best_row,
                     normalized_test_scores)
from evaluation.utils import calculate_metrics
from scores import SigConfig, sig_features
from run_sweep import (task_meta, calibration_pool, fit_theta, score_episodes)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/home/louis/fiper/my_experiments/sig_study"


def load_kerncd(task):
    p = f"/home/louis/fiper/data/{task}/results/kern_cd/eval_results.pkl"
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def best_from_metrics(nested, stats, patience=0):
    """nested[style][q][w] -> normalised scores; return best TWA row."""
    best = None
    for style, byq in nested.items():
        for q, byw in byq.items():
            for w, sc in byw.items():
                m = calculate_metrics(sc, stats, patience)
                row = {"threshold": style, "quantile": q, "window": w,
                       "TWA": float(m["TWA"]), "TPR": float(m["TPR"]),
                       "TNR": float(m["TNR"])}
                if best is None or row["TWA"] > best["TWA"]:
                    best = row
    return best


def combine(n1, n2, op):
    out = {}
    for style in n1:
        if style not in n2:
            continue
        out[style] = {}
        for q in n1[style]:
            if q not in n2[style]:
                continue
            out[style][q] = {}
            for w in n1[style][q]:
                if w not in n2[style][q]:
                    continue
                a, b = n1[style][q][w], n2[style][q][w]
                f = np.maximum if op == "or" else np.minimum
                out[style][q][w] = [f(np.asarray(x), np.asarray(y))
                                    for x, y in zip(a, b)]
    return out


def run(task, family, cfg_kw):
    kc = load_kerncd(task)
    if kc is None:
        print(f"  {task}: no saved kern_cd result, skipping")
        return []

    ds = load_task(task)
    exec_h, pred_h, bt = task_meta(ds)
    stats = dataset_stats(ds)
    actions = ("position",)
    pool = calibration_pool(ds, actions)
    theta = fit_theta(pool)
    cfg = SigConfig(path_scale=theta, **cfg_kw)

    ref = pool
    ref_feats = sig_features(ref[:256], cfg) if family == "sigmmd" else None
    cal, cal_ok = score_episodes(ds, "calibration", family, cfg, actions,
                                 exec_h, bt, ref[:256], ref_feats)
    tst, tst_ok = score_episodes(ds, "test", family, cfg, actions,
                                 exec_h, bt, ref[:256], ref_feats)
    cal = [np.nan_to_num(s) for s in cal]
    tst = [np.nan_to_num(s) for s in tst]

    sig_norm = normalized_test_scores(cal, cal_ok, tst)
    kc_norm = kc["test_scores_by_threshold"]

    # episode lengths must line up for the elementwise combination
    L1 = [len(x) for x in sig_norm["ct_quantile"][0.9][1]]
    L2 = [len(x) for x in kc_norm["ct_quantile"][0.9][1]]
    if L1 != L2:
        print(f"  {task}: episode length mismatch (sig {L1[:3]} vs kern_cd {L2[:3]}), skipping")
        return []

    rows = []
    b_sig = best_from_metrics(sig_norm, stats)
    b_kc = best_from_metrics(kc_norm, stats)
    rows.append({"task": task, "method": f"{family}({cfg.tag()})", **b_sig})
    rows.append({"task": task, "method": "kern_cd", **b_kc})
    for op in ("or", "and"):
        b = best_from_metrics(combine(kc_norm, sig_norm, op), stats)
        rows.append({"task": task, "method": f"kern_cd_{op}_{family}", **b})
    for r in rows:
        print(f"  {task:11s} {r['method']:28s} TWA {r['TWA']:.3f} "
              f"TPR {r['TPR']:.2f} TNR {r['TNR']:.2f}  @{r['threshold']}/q{r['quantile']}/w{r['window']}",
              flush=True)
    del ds
    torch.cuda.empty_cache()
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--family", default="sigtc")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--kernel", default="lin")
    args = ap.parse_args()

    cfg_kw = dict(level=args.level, kernel=args.kernel, time_aug=True)
    allr = []
    for t in args.tasks:
        allr += run(t, args.family, cfg_kw)
    df = pd.DataFrame(allr)
    df.to_csv(f"{OUT}/combine_{args.family}_L{args.level}.csv", index=False)
    print(f"\nwrote combine_{args.family}_L{args.level}.csv")
