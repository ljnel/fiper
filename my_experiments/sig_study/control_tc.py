"""Control experiment: reimplement FIPER's `tc` (MMD-RBF on flattened
overlapping action chunks) inside this harness and compare against the
published numbers in data/results/complete_results.csv.

If this reproduces `tc`, the harness is trustworthy for the signature variants.
"""
from __future__ import annotations

import sys, time

sys.path.insert(0, "/home/louis/fiper/my_experiments/sig_study")

import numpy as np
import pandas as pd
import torch

from loader import load_task, episodes, TASKS
from harness import evaluate, dataset_stats, best_row, episode_auroc

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def flat_tc_episode(ap: torch.Tensor, exec_h: int, backtrack: int = 1) -> np.ndarray:
    """Exactly TCEval._mmd_rbf with gamma='median', batched over timesteps.

    ap: (T,B,H,a) on device.  prev = ap[t-bt][:, off:], curr = ap[t][:, :H-off],
    each flattened to (B, (H-off)*a); MMD^2 with a per-step median bandwidth.
    """
    T, B, H, a = ap.shape
    off = exec_h * backtrack
    if T <= backtrack or off >= H:
        return np.zeros(T)
    prev = ap[:-backtrack, :, off:, :].reshape(T - backtrack, B, -1)
    curr = ap[backtrack:, :, :H - off, :].reshape(T - backtrack, B, -1)
    Z = torch.cat([prev, curr], dim=1)                       # (T', 2B, D)
    sq = torch.cdist(Z, Z).pow(2)
    # gamma = 1 / (2 * median of the strictly positive squared distances)
    big = sq.flatten(1).clone()
    big[big <= 0] = float("nan")
    med = big.nanmedian(dim=1).values.clamp_min(1e-30)
    g = (1.0 / (2.0 * med)).view(-1, 1, 1)
    K = torch.exp(-g * sq)
    Kxx, Kyy, Kxy = K[:, :B, :B], K[:, B:, B:], K[:, :B, B:]
    vals = Kxx.mean((-2, -1)) + Kyy.mean((-2, -1)) - 2 * Kxy.mean((-2, -1))
    return np.concatenate([np.zeros(backtrack), vals.cpu().numpy()])


def run(task: str) -> dict:
    ds = load_task(task)
    md = ds.data["metadata"]
    exec_h = md["actions"].get("action_execution_horizon") or 4
    pred_h = md["actions"]["action_prediction_horizon"]
    backtrack = min(1, pred_h // exec_h - 1)

    t0 = time.time()
    cal_scores, cal_succ = [], []
    for ap, ok in episodes(ds, "calibration", ("position",)):
        cal_scores.append(flat_tc_episode(torch.tensor(ap, dtype=torch.float32, device=DEV), exec_h, backtrack))
        cal_succ.append(ok)
    test_scores, test_succ = [], []
    for ap, ok in episodes(ds, "test", ("position",)):
        test_scores.append(flat_tc_episode(torch.tensor(ap, dtype=torch.float32, device=DEV), exec_h, backtrack))
        test_succ.append(ok)
    score_time = time.time() - t0

    stats = dataset_stats(ds)
    rows = evaluate(cal_scores, cal_succ, test_scores, stats)
    b = best_row(rows)
    return {"task": task, "best_TWA": b["TWA"], "at": (b["threshold"], b["quantile"], b["window"]),
            "auroc_w1": episode_auroc(test_scores, test_succ, 1),
            "score_time_s": score_time}


if __name__ == "__main__":
    pub = pd.read_csv("/home/louis/fiper/data/results/complete_results.csv")
    pub_tc = pub[pub.Method == "tc"].groupby("Task")["TWA"].max()
    out = []
    for task in TASKS:
        r = run(task)
        r["published_tc_TWA"] = float(pub_tc.get(task, np.nan))
        out.append(r)
        print(f"{task:11s} mine {r['best_TWA']:.3f}  published {r['published_tc_TWA']:.3f}  "
              f"(diff {r['best_TWA']-r['published_tc_TWA']:+.3f})  auroc {r['auroc_w1']:.3f}  "
              f"{r['score_time_s']:.1f}s  @{r['at']}", flush=True)
    pd.DataFrame(out).to_csv("/home/louis/fiper/my_experiments/sig_study/control_tc.csv", index=False)
