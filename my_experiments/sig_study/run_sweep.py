"""Main parameter sweep for the signature / signature-kernel viability study.

Stages
------
scale : does dilating the path so TV~1 make the truncation level matter at all?
level : truncation level 1..5 at the fixed good scaling, per score family
kernel: linear vs RBF-on-signature-features vs untruncated PDE signature kernel
aug   : time / basepoint / lead-lag / tensor-normalisation ablation
dims  : position-only vs the full action vector (memory blow-up)

Everything is scored on GPU and evaluated through FIPER's own threshold and
metric code (see harness.py), so best_TWA is comparable to
data/results/complete_results.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/home/louis/fiper/my_experiments/sig_study")

import numpy as np
import pandas as pd
import torch

from loader import load_task, episodes, TASKS
from harness import evaluate, dataset_stats, best_row, episode_auroc
from scores import SigConfig, sig_features
from episode_scores import sigvar_episode, sigtc_episode, sigmmd_episode

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/home/louis/fiper/my_experiments/sig_study"


# --------------------------------------------------------------------------
def task_meta(ds):
    md = ds.data["metadata"]
    exec_h = md["actions"].get("action_execution_horizon") or 4
    pred_h = md["actions"]["action_prediction_horizon"]
    return exec_h, pred_h, min(1, pred_h // exec_h - 1)


def calibration_pool(ds, actions, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    pool = [ap.reshape(-1, ap.shape[-2], ap.shape[-1])
            for ap, ok in episodes(ds, "calibration", actions)]
    pool = np.concatenate(pool, 0)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return torch.tensor(pool[idx], dtype=torch.float32, device=DEV)


def fit_theta(pool):
    """Dilation making the median total variation of a chunk equal to 1."""
    tv = (pool[:, 1:] - pool[:, :-1]).norm(dim=-1).sum(-1)
    med = float(tv.median())
    return 1.0 / med if med > 0 else 1.0


def fit_global_gamma(pool, cfg, n=1024):
    """Median heuristic for the RBF-on-signature-features bandwidth, computed
    once on calibration chunks rather than per step."""
    S = sig_features(pool[:n], cfg)
    sq = torch.cdist(S, S).pow(2)
    v = sq[sq > 0]
    if v.numel() == 0:
        return 1.0
    return float(1.0 / (2.0 * torch.median(v)))


def score_episodes(ds, subset, family, cfg, actions, exec_h, backtrack,
                   ref=None, ref_feats=None):
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
        scores.append(np.nan_to_num(np.asarray(s, dtype=np.float64),
                                    nan=0.0, posinf=0.0, neginf=0.0))
        succ.append(ok)
    return scores, succ


def one_config(ds, stats, family, cfg, actions, exec_h, backtrack, pool,
               n_ref=256, do_eval=True):
    """Score + evaluate one (family, SigConfig).  Returns a result row."""
    ref = pool[:n_ref] if family == "sigmmd" else None
    ref_feats = None
    if family == "sigmmd" and cfg.kernel != "pde":
        ref_feats = sig_features(ref, cfg)

    torch.cuda.synchronize() if DEV == "cuda" else None
    t0 = time.time()
    cal, cal_ok = score_episodes(ds, "calibration", family, cfg, actions,
                                 exec_h, backtrack, ref, ref_feats)
    tst, tst_ok = score_episodes(ds, "test", family, cfg, actions,
                                 exec_h, backtrack, ref, ref_feats)
    torch.cuda.synchronize() if DEV == "cuda" else None
    score_s = time.time() - t0
    n_steps = sum(len(s) for s in cal) + sum(len(s) for s in tst)

    row = {"family": family, "level": cfg.level, "kernel": cfg.kernel,
           "time_aug": cfg.time_aug, "basepoint": cfg.basepoint,
           "lead_lag": cfg.lead_lag, "normalise": cfg.normalise,
           "path_scale": round(cfg.path_scale, 4), "actions": actions[0],
           "n_steps": n_steps, "score_s": round(score_s, 2),
           "us_per_step": round(1e6 * score_s / max(n_steps, 1), 1)}

    if do_eval:
        t1 = time.time()
        rows = evaluate(cal, cal_ok, tst, stats)
        row["eval_s"] = round(time.time() - t1, 1)
        b = best_row(rows)
        row.update({"best_TWA": b["TWA"], "TPR": b["TPR"], "TNR": b["TNR"],
                    "at": f"{b['threshold']}/q{b['quantile']}/w{b['window']}",
                    "auroc": episode_auroc(tst, tst_ok, 1)})
        # score dynamic range: a collapsed score is a red flag the TWA can hide
        allv = np.concatenate(tst)
        row["score_cv"] = float(np.std(allv) / (np.mean(np.abs(allv)) + 1e-30))
    return row


# --------------------------------------------------------------------------
def stage_scale(tasks, levels=(1, 2, 3, 4, 5), families=("sigvar", "sigtc", "sigmmd")):
    """Unscaled vs TV-normalised paths, across truncation levels."""
    out = []
    for task in tasks:
        ds = load_task(task)
        exec_h, pred_h, bt = task_meta(ds)
        stats = dataset_stats(ds)
        actions = ("position",)
        pool = calibration_pool(ds, actions)
        theta = fit_theta(pool)
        for scale_name, scale in (("raw", 1.0), ("tv_norm", theta)):
            for level in levels:
                cfg = SigConfig(level=level, kernel="lin", time_aug=True,
                                path_scale=scale)
                for fam in families:
                    r = one_config(ds, stats, fam, cfg, actions, exec_h, bt,
                                   pool)
                    r.update({"task": task, "stage": "scale", "scaling": scale_name,
                              "theta": round(theta, 4)})
                    out.append(r)
                    print(f"  {task:10s} {scale_name:7s} {fam:7s} L{level} "
                          f"TWA {r['best_TWA']:.3f} auroc {r['auroc']:.3f} "
                          f"cv {r['score_cv']:.2e} {r['us_per_step']:.0f}us/step", flush=True)
        del ds
        torch.cuda.empty_cache()
    return out


def stage_kernel(tasks, level=3):
    """Linear vs RBF-on-features vs untruncated PDE signature kernel."""
    out = []
    for task in tasks:
        ds = load_task(task)
        exec_h, pred_h, bt = task_meta(ds)
        stats = dataset_stats(ds)
        actions = ("position",)
        pool = calibration_pool(ds, actions)
        theta = fit_theta(pool)
        variants = [
            ("lin", dict(kernel="lin")),
            ("rbf_median", dict(kernel="rbf", rbf_gamma="median")),
            ("rbf_global", dict(kernel="rbf", rbf_gamma=None)),   # filled below
            ("pde_d0", dict(kernel="pde", dyadic_order=0)),
            ("pde_d1", dict(kernel="pde", dyadic_order=1)),
        ]
        for name, kw in variants:
            kw = dict(kw)
            cfg = SigConfig(level=level, time_aug=True, path_scale=theta,
                            **{k: v for k, v in kw.items() if v is not None})
            if name == "rbf_global":
                cfg = SigConfig(level=level, time_aug=True, path_scale=theta,
                                kernel="rbf",
                                rbf_gamma=fit_global_gamma(pool * theta, SigConfig(
                                    level=level, time_aug=True, path_scale=theta)))
            for fam in ("sigvar", "sigtc", "sigmmd"):
                if name == "rbf_median" and fam == "sigmmd":
                    continue  # sigmmd needs a fixed bandwidth
                try:
                    r = one_config(ds, stats, fam, cfg, actions, exec_h, bt,
                                   pool)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  {task:10s} {name:10s} {fam:7s} OOM", flush=True)
                    out.append({"task": task, "stage": "kernel", "variant": name,
                                "family": fam, "level": level, "oom": True})
                    continue
                r.update({"task": task, "stage": "kernel", "variant": name,
                          "theta": round(theta, 4)})
                out.append(r)
                print(f"  {task:10s} {name:10s} {fam:7s} TWA {r['best_TWA']:.3f} "
                      f"auroc {r['auroc']:.3f} {r['us_per_step']:.0f}us/step", flush=True)
        del ds
        torch.cuda.empty_cache()
    return out


def stage_aug(tasks, level=3):
    """Path-augmentation ablation at the chosen level."""
    out = []
    variants = [
        ("time", dict(time_aug=True)),
        ("no_time", dict(time_aug=False)),
        ("time+bp", dict(time_aug=True, basepoint=True)),
        ("time+ll", dict(time_aug=True, lead_lag=True)),
        ("time+norm", dict(time_aug=True, normalise=True)),
        ("increments", dict(time_aug=True, increments=True)),
    ]
    for task in tasks:
        ds = load_task(task)
        exec_h, pred_h, bt = task_meta(ds)
        stats = dataset_stats(ds)
        actions = ("position",)
        pool = calibration_pool(ds, actions)
        theta = fit_theta(pool)
        for name, kw in variants:
            cfg = SigConfig(level=level, kernel="lin", path_scale=theta, **kw)
            for fam in ("sigvar", "sigtc", "sigmmd"):
                r = one_config(ds, stats, fam, cfg, actions, exec_h, bt,
                               pool)
                r.update({"task": task, "stage": "aug", "variant": name,
                          "theta": round(theta, 4)})
                out.append(r)
                print(f"  {task:10s} {name:10s} {fam:7s} TWA {r['best_TWA']:.3f} "
                      f"auroc {r['auroc']:.3f}", flush=True)
        del ds
        torch.cuda.empty_cache()
    return out


def stage_dims(tasks, levels=(1, 2, 3)):
    """position-only vs the full action vector: detection gain vs memory."""
    out = []
    for task in tasks:
        ds = load_task(task)
        exec_h, pred_h, bt = task_meta(ds)
        stats = dataset_stats(ds)
        for actions in (("position",), ("all",)):
            pool = calibration_pool(ds, actions)
            theta = fit_theta(pool)
            a = pool.shape[-1]
            for level in levels:
                cfg = SigConfig(level=level, kernel="lin", time_aug=True,
                                path_scale=theta)
                for fam in ("sigvar", "sigtc"):
                    try:
                        r = one_config(ds, stats, fam, cfg, actions, exec_h, bt, pool)
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print(f"  {task:10s} {actions[0]:8s} L{level} {fam:7s} OOM", flush=True)
                        out.append({"task": task, "stage": "dims", "actions": actions[0],
                                    "level": level, "family": fam, "a": a,
                                    "feat_dim": cfg.feat_dim(a), "oom": True})
                        continue
                    r.update({"task": task, "stage": "dims", "a": a,
                              "feat_dim": cfg.feat_dim(a), "theta": round(theta, 4)})
                    out.append(r)
                    print(f"  {task:10s} {actions[0]:8s} L{level} {fam:7s} "
                          f"dim {cfg.feat_dim(a):7d} TWA {r['best_TWA']:.3f} "
                          f"auroc {r['auroc']:.3f} {r['us_per_step']:.0f}us/step", flush=True)
            del pool
            torch.cuda.empty_cache()
        del ds
        torch.cuda.empty_cache()
    return out


STAGES = {"scale": stage_scale, "kernel": stage_kernel, "aug": stage_aug,
          "dims": stage_dims}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--families", nargs="*", default=["sigvar", "sigtc", "sigmmd"])
    args = ap.parse_args()

    kw = {"families": tuple(args.families)} if args.stage == "scale" else ({} if args.stage == "dims" else {"level": args.level})
    rows = STAGES[args.stage](args.tasks, **kw)
    df = pd.DataFrame(rows)
    suffix = "_" + "_".join(args.families) if args.stage == "scale" and len(args.families) < 3 else ""
    path = os.path.join(OUT, f"sweep_{args.stage}{suffix}.csv")
    df.to_csv(path, index=False)
    print(f"\nwrote {path}  ({len(df)} rows)")
