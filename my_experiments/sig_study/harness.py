"""Scoring -> FIPER metrics, reusing the repo's own threshold/metric utilities
so the numbers are directly comparable to data/results/complete_results.csv.
"""
from __future__ import annotations

import sys

ROOT = "/home/louis/fiper"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from sklearn.metrics import roc_auc_score

from evaluation.utils import (
    calculate_metrics,
    compute_thresholds,
    normalize_scores_by_threshold,
)

# Mirrors configs/eval/base.yaml.  NB: base.yaml spells the key
# `extend_threshold` while the code reads `extend_thresholds`, so the repo
# effectively always uses the "mean" extension - replicated here on purpose.
EVAL_CFG = {
    "window_sizes": [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 20, 25, 30, 35, 40, 45, 50],
    "quantiles": [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
    "thresholds": ["tvt_quantile", "tvt_cp_band", "ct_quantile"],
    "handle_zero_thresholds": {"style": "add_small_score"},
    "detection_patience": 0,
}
SMALL = 1e-6


def apply_window(scores, window_size):
    """Same aggregation as BaseEvalClass._apply_window_size (uniform sum)."""
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return s
    if window_size == "all":
        return np.cumsum(s)
    w = int(window_size)
    n = len(s)
    if n < w:
        s = np.concatenate([np.zeros(w - n), s])
    c = np.concatenate([[0.0], np.cumsum(s)])
    out = np.array([c[i + 1] - c[max(0, i + 1 - w)] for i in range(len(s))])
    return out[-n:]


def dataset_stats(ds):
    md = ds.data["metadata"]
    return {
        "max_episode_length": max(md["episode_lengths"]),
        "id_rollouts": ds.get_rollout_subtypes(subset="test", subsubset="id"),
        "ood_rollouts": ds.get_rollout_subtypes(subset="test", subsubset="ood"),
        "successful_rollouts": ds.get_rollout_types(subset=["test", "successful"],
                                                    reduce_mask_to=["test"]),
    }


def evaluate(cal_scores, cal_success, test_scores, stats, cfg=EVAL_CFG):
    """cal_scores/test_scores: list of per-episode 1-D score arrays.

    Returns a list of dicts, one per (threshold_style, quantile, window).
    """
    rows = []
    cal_scores = [np.asarray(s, dtype=np.float64) + SMALL for s in cal_scores]
    test_scores = [np.asarray(s, dtype=np.float64) + SMALL for s in test_scores]

    windowed_cal, windowed_test = {}, {}
    for w in cfg["window_sizes"]:
        windowed_cal[w] = [apply_window(s, w) for s in cal_scores]
        windowed_test[w] = [apply_window(s, w) for s in test_scores]

    for style in cfg["thresholds"]:
        for q in cfg["quantiles"]:
            for w in cfg["window_sizes"]:
                succ_cal = [s for s, ok in zip(windowed_cal[w], cal_success) if ok]
                if not succ_cal:
                    continue
                thr = compute_thresholds(style, q, [list(s) for s in succ_cal])
                if isinstance(thr, (list, np.ndarray)) and np.min(thr) <= 0:
                    thr = np.clip(np.asarray(thr, dtype=float), 1e-12, None)
                elif np.isscalar(thr) and thr <= 0:
                    thr = 1e-12
                norm = normalize_scores_by_threshold(windowed_test[w], thr, cfg=cfg)
                m = calculate_metrics(norm, stats, cfg["detection_patience"])
                rows.append({"threshold": style, "quantile": q, "window": w,
                             "TWA": float(m["TWA"]), "TPR": float(m["TPR"]),
                             "TNR": float(m["TNR"]),
                             "accuracy": float(m["accuracy"]),
                             "bal_acc": float(m["balanced_accuracy"]),
                             "det_time": float(m["avg_detection_time"])})
    return rows


def normalized_test_scores(cal_scores, cal_success, test_scores, cfg=EVAL_CFG):
    """Same pipeline as `evaluate`, but returns the threshold-normalised test
    scores nested as [style][quantile][window] - the structure FIPER's
    `_combine_two_methods` consumes, so a signature score can be OR/AND-ed with
    a saved kern_cd result."""
    cal_scores = [np.asarray(s, dtype=np.float64) + SMALL for s in cal_scores]
    test_scores = [np.asarray(s, dtype=np.float64) + SMALL for s in test_scores]
    out = {}
    for style in cfg["thresholds"]:
        out[style] = {}
        for q in cfg["quantiles"]:
            out[style][q] = {}
            for w in cfg["window_sizes"]:
                wc = [apply_window(s, w) for s in cal_scores]
                wt = [apply_window(s, w) for s in test_scores]
                succ_cal = [s for s, ok in zip(wc, cal_success) if ok]
                if not succ_cal:
                    continue
                thr = compute_thresholds(style, q, [list(s) for s in succ_cal])
                if isinstance(thr, (list, np.ndarray)):
                    thr = np.clip(np.asarray(thr, dtype=float), 1e-12, None)
                elif thr <= 0:
                    thr = 1e-12
                out[style][q][w] = normalize_scores_by_threshold(wt, thr, cfg=cfg)
    return out


def episode_auroc(test_scores, successful, window=1):
    """Threshold-free sanity metric: AUROC of max-over-episode windowed score
    against the failure label.  Insensitive to the CP machinery, so it ranks
    score *definitions* rather than threshold styles."""
    y = np.asarray([0 if s else 1 for s in successful])          # 1 = failure
    if y.min() == y.max():
        return float("nan")
    v = np.array([np.max(apply_window(s, window)) for s in test_scores])
    if not np.isfinite(v).all():
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return float(roc_auc_score(y, v))


def best_row(rows, key="TWA"):
    return max(rows, key=lambda r: r[key]) if rows else None
