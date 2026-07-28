"""Do the initial object positions predict task failure? (stacking)

We take the *first* overhead RGB frame of each stacking test rollout -- which
captures the initial layout of the blocks on the table -- and ask whether a
classifier can predict the ground-truth outcome (success vs. failure) from it.

The stacking rgb is 96x192: two 96x96 camera views side by side. The LEFT half
is the top-down / overhead view (see docs/data.md and my_experiments exploration);
that is the one whose pixels encode the initial object positions.

Pipeline:
  1. Load all test rollouts, grab frame 0's overhead crop + label.
  2. Stratified 80/20 train/test split (seed fixed).
  3. Compare: majority-class baseline, logistic regression on pixels,
     and a small CNN.  Report acc / balanced-acc / ROC-AUC / confusion matrix.

Run:  pixi run python my_experiments/predict_failure_from_initial_frame.py
"""

from __future__ import annotations

import glob
import os
import pickle

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(ROOT, "data", "stacking", "rollouts", "test")
CACHE = os.path.join(os.path.dirname(__file__), "_initial_frames_cache.npz")
SEED = 0


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_dataset():
    """Return (X, y) where X is (N, 96, 96, 3) uint8 overhead frame 0,
    y is (N,) int with 1 = failure, 0 = success."""
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        print(f"loaded cache {CACHE}  X={d['X'].shape}")
        return d["X"], d["y"]

    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.pkl")))
    X, y = [], []
    for i, f in enumerate(files):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        frame0 = d["rollout"][0]["rgb"]          # (96, 192, 3)
        overhead = frame0[:, :96]                # left half = top-down view
        X.append(overhead)
        y.append(0 if d["metadata"]["successful"] else 1)
        if (i + 1) % 100 == 0:
            print(f"  loaded {i + 1}/{len(files)}")
    X = np.stack(X).astype(np.uint8)
    y = np.array(y, dtype=np.int64)
    np.savez_compressed(CACHE, X=X, y=y)
    print(f"cached {CACHE}  X={X.shape}  failures={y.sum()}/{len(y)}")
    return X, y


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def report(name, y_true, y_pred, y_score=None):
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        roc_auc_score,
    )

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score) if y_score is not None else float("nan")
    cm = confusion_matrix(y_true, y_pred)  # rows=true, cols=pred; [0]=success,[1]=failure
    print(f"\n=== {name} ===")
    print(f"  accuracy          {acc:.3f}")
    print(f"  balanced accuracy {bacc:.3f}")
    print(f"  ROC-AUC           {auc:.3f}")
    print(f"  confusion (rows=true succ/fail, cols=pred succ/fail):\n{cm}")
    return dict(name=name, acc=acc, bacc=bacc, auc=auc)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def run_logreg(Xtr, ytr, Xte, yte):
    """Logistic regression on standardized, downsampled, flattened pixels."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def feats(X):
        # downsample 96->48 by 2x2 average, grayscale-ish keep channels, flatten
        Xf = X.astype(np.float32).reshape(-1, 48, 2, 48, 2, 3).mean((2, 4))
        return Xf.reshape(len(Xf), -1)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.01, class_weight="balanced"),
    )
    clf.fit(feats(Xtr), ytr)
    score = clf.predict_proba(feats(Xte))[:, 1]
    return report("logreg (pixels)", yte, (score >= 0.5).astype(int), score)


def run_cnn(Xtr, ytr, Xte, yte):
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def to_t(X):
        # NHWC uint8 -> NCHW float in [0,1]
        return torch.from_numpy(X).float().div(255).permute(0, 3, 1, 2)

    # internal train/val split (from the 80% train) for early stopping
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(Xtr))
    nval = len(idx) // 5
    vi, ti = idx[:nval], idx[nval:]
    xt, yt = to_t(Xtr[ti]).to(dev), torch.from_numpy(ytr[ti]).to(dev)
    xv, yv = to_t(Xtr[vi]).to(dev), torch.from_numpy(ytr[vi]).to(dev)
    xe = to_t(Xte).to(dev)

    net = nn.Sequential(
        nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(), nn.BatchNorm2d(16),   # 48
        nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), nn.BatchNorm2d(32),  # 24
        nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.BatchNorm2d(64),  # 12
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(0.5), nn.Linear(64, 2),
    ).to(dev)

    # class weights for imbalance
    w = torch.tensor([1.0 / (ytr == 0).mean(), 1.0 / (ytr == 1).mean()], device=dev).float()
    w = w / w.sum() * 2
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)

    best_val, best_state, patience, bad = 1e9, None, 30, 0
    bs = 64
    for epoch in range(300):
        net.train()
        perm = torch.randperm(len(xt), device=dev)
        for b in range(0, len(xt), bs):
            j = perm[b:b + bs]
            opt.zero_grad()
            loss = lossf(net(xt[j]), yt[j])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vloss = lossf(net(xv), yv).item()
        if vloss < best_val - 1e-4:
            best_val, best_state, bad = vloss, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        score = torch.softmax(net(xe), 1)[:, 1].cpu().numpy()
    return report("small CNN", yte, (score >= 0.5).astype(int), score)


# --------------------------------------------------------------------------- #
def main():
    X, y = load_dataset()
    print(f"\ndataset: {len(y)} episodes, failures={y.sum()} ({y.mean():.1%})")

    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"train={len(ytr)} (fail {ytr.mean():.1%})  test={len(yte)} (fail {yte.mean():.1%})")

    results = []
    # majority-class baseline
    maj = int(ytr.mean() >= 0.5)
    results.append(report("majority baseline", yte, np.full_like(yte, maj), None))
    results.append(run_logreg(Xtr, ytr, Xte, yte))
    results.append(run_cnn(Xtr, ytr, Xte, yte))

    # robustness: 5-fold CV AUC for the logreg over the full 800 (single split is noisy)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xf = X.astype(np.float32).reshape(-1, 48, 2, 48, 2, 3).mean((2, 4)).reshape(len(X), -1)
    cv = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.01, class_weight="balanced"),
    )
    cv_auc = cross_val_score(cv, Xf, y, cv=5, scoring="roc_auc")
    print(f"\nlogreg 5-fold CV ROC-AUC: {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}  {np.round(cv_auc,3)}")

    print("\n\n================ SUMMARY ================")
    print(f"{'model':<22}{'acc':>7}{'bal-acc':>9}{'auc':>7}")
    for r in results:
        print(f"{r['name']:<22}{r['acc']:>7.3f}{r['bacc']:>9.3f}{r['auc']:>7.3f}")
    print("\nBaseline (always-predict-failure) acc =", f"{max(yte.mean(), 1-yte.mean()):.3f}")
    print("If a model's ROC-AUC is ~0.5 and bal-acc ~0.5, the initial layout")
    print("carries little predictive signal for failure; >0.6 suggests it does.")


if __name__ == "__main__":
    main()
