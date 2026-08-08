"""Do the initial block *positions* predict stacking failure?

Unlike predict_failure_from_initial_frame.py (which feeds raw overhead pixels to
a classifier), this script extracts explicit (x, y) centroids for the four
colored blocks from the first overhead frame and classifies on those. That makes
the result interpretable: the model coefficients tell us *which* block and
*which* spatial relation predict failure.

Features per episode:
  - 8 absolute coords: (x, y) for red, blue, green, cyan
  - 6 pairwise block-to-block distances
  - 4 distances from each block to the scene centroid (spread)

Run:  pixi run python my_experiments/failure_eda/predict_failure_from_centroids.py
"""
import itertools
import os

import numpy as np
from matplotlib.colors import rgb_to_hsv

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "_initial_frames_cache.npz")
SEED = 0
COLORS = ["red", "blue", "green", "cyan"]


# --------------------------------------------------------------------------- #
# Centroid extraction (same logic validated in centroid_sanity.py)
# --------------------------------------------------------------------------- #
def centroids(img):
    """Return dict color -> (x, y) or None from a 96x96 overhead frame."""
    hsv = rgb_to_hsv(img.astype(np.float32) / 255)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    not_arm = (np.arange(96)[:, None] > 14)  # drop top strip (robot gripper)
    masks = {
        "red": ((h < 0.05) | (h > 0.95)) & (s > 0.5) & (v > 0.4),
        "green": (h > 0.25) & (h < 0.45) & (s > 0.5) & (v > 0.4),
        "cyan": (h > 0.45) & (h < 0.55) & (s > 0.45) & (v > 0.5),
        "blue": (h > 0.55) & (h < 0.75) & (s > 0.5) & (v > 0.3),
    }
    out = {}
    for c, m in masks.items():
        m = m & not_arm
        if m.sum() < 4:
            out[c] = None
        else:
            ys, xs = np.nonzero(m)
            out[c] = (float(xs.mean()), float(ys.mean()))
    return out


def build_features(X):
    """Return (F, names, n_missing). NaN where a block wasn't found."""
    rows = []
    miss = 0
    for img in X:
        c = centroids(img)
        miss += sum(v is None for v in c.values())
        pts = {k: (v if v is not None else (np.nan, np.nan)) for k, v in c.items()}
        feat = []
        for col in COLORS:                       # 8 absolute coords
            feat += [pts[col][0], pts[col][1]]
        for a, b in itertools.combinations(COLORS, 2):  # 6 pairwise dists
            feat.append(np.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]))
        cx = np.nanmean([pts[c][0] for c in COLORS])     # scene centroid
        cy = np.nanmean([pts[c][1] for c in COLORS])
        for col in COLORS:                       # 4 dists to scene centroid
            feat.append(np.hypot(pts[col][0] - cx, pts[col][1] - cy))
        rows.append(feat)

    names = [f"{c}_{ax}" for c in COLORS for ax in ("x", "y")]
    names += [f"d_{a}_{b}" for a, b in itertools.combinations(COLORS, 2)]
    names += [f"spread_{c}" for c in COLORS]
    return np.array(rows, dtype=np.float32), names, miss


# --------------------------------------------------------------------------- #
def report(name, y_true, y_pred, y_score):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, roc_auc_score)
    print(f"\n=== {name} ===")
    print(f"  accuracy          {accuracy_score(y_true, y_pred):.3f}")
    print(f"  balanced accuracy {balanced_accuracy_score(y_true, y_pred):.3f}")
    print(f"  ROC-AUC           {roc_auc_score(y_true, y_score):.3f}")
    print(f"  confusion (rows=true succ/fail):\n{confusion_matrix(y_true, y_pred)}")
    return dict(name=name, acc=accuracy_score(y_true, y_pred),
                bacc=balanced_accuracy_score(y_true, y_pred),
                auc=roc_auc_score(y_true, y_score))


def main():
    if not os.path.exists(CACHE):
        raise SystemExit("missing cache -- run predict_failure_from_initial_frame.py first")
    d = np.load(CACHE)
    X, y = d["X"], d["y"]
    F, names, miss = build_features(X)
    print(f"features {F.shape}  ({len(names)} per episode)  "
          f"missing-block cells: {miss}/{len(X) * 4}")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Ftr, Fte, ytr, yte = train_test_split(
        F, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"train={len(ytr)} (fail {ytr.mean():.1%})  test={len(yte)} (fail {yte.mean():.1%})")

    results = []
    # baseline
    maj = int(ytr.mean() >= 0.5)
    results.append(report("majority baseline", yte, np.full_like(yte, maj),
                          np.full(len(yte), 0.5)))

    # logistic regression (interpretable)
    logreg = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced"))
    logreg.fit(Ftr, ytr)
    sc = logreg.predict_proba(Fte)[:, 1]
    results.append(report("logreg (centroids)", yte, (sc >= 0.5).astype(int), sc))

    # random forest (nonlinear)
    rf = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=400, max_depth=6,
                               class_weight="balanced", random_state=SEED))
    rf.fit(Ftr, ytr)
    scr = rf.predict_proba(Fte)[:, 1]
    results.append(report("random forest", yte, (scr >= 0.5).astype(int), scr))

    # robust estimate: 5-fold CV AUC on full data
    cv = cross_val_score(logreg, F, y, cv=5, scoring="roc_auc")
    print(f"\nlogreg 5-fold CV ROC-AUC: {cv.mean():.3f} +/- {cv.std():.3f}  {np.round(cv,3)}")

    # interpretability: top coefficients and RF importances
    coef = logreg.named_steps["logisticregression"].coef_[0]
    order = np.argsort(-np.abs(coef))
    print("\ntop logreg coefficients (sign = direction toward FAILURE):")
    for i in order[:8]:
        print(f"  {names[i]:<12} {coef[i]:+.3f}")
    imp = rf.named_steps["randomforestclassifier"].feature_importances_
    print("\ntop random-forest importances:")
    for i in np.argsort(-imp)[:8]:
        print(f"  {names[i]:<12} {imp[i]:.3f}")

    print("\n\n================ SUMMARY ================")
    print(f"{'model':<22}{'acc':>7}{'bal-acc':>9}{'auc':>7}")
    for r in results:
        print(f"{r['name']:<22}{r['acc']:>7.3f}{r['bacc']:>9.3f}{r['auc']:>7.3f}")
    print(f"\nchance ROC-AUC = 0.5; baseline acc = {max(yte.mean(), 1-yte.mean()):.3f}")


if __name__ == "__main__":
    main()
