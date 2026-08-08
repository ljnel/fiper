"""
Gripper-signal failure prediction on stacking (test set only).

Setup
-----
- Source: the 800 stacking `test/` rollouts (the only ones with both classes).
- Binary task: predict FAILURE (positive class = not successful).
- Feature: the measured gripper width `agent_pos[7]` over the first T=20 steps,
  used as a 20-dim trajectory vector. Truncating every episode to a fixed
  prefix removes episode-length as a leak (failures otherwise run to the cap).
- Model: standardized logistic regression.
- Metric: AUROC on a stratified 20% holdout (no threshold).

A length-only logistic baseline is also reported: episode length is a strong
leak, so the gripper AUROC should be read against it, not against 0.5.

Run: pixi run python my_experiments/failure_eda/gripper_failure_logreg.py
"""
import glob
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

T = 20            # prefix length (steps); all test episodes have >= 24
GRIP_AP = 7       # agent_pos index = measured gripper width
SEED = 0
DATA = "data/stacking/rollouts/test/*.pkl"

# --- load: extract the first-T and last-T gripper windows per episode ---
Xfirst, Xlast, lengths, y, strat = [], [], [], [], []
for f in sorted(glob.glob(DATA)):
    d = pickle.load(open(f, "rb"))
    m = d["metadata"]
    grip = np.array([s["agent_pos"][GRIP_AP] for s in d["rollout"]], dtype=float)
    Xfirst.append(grip[:T])                 # early window (deployable, online)
    Xlast.append(grip[-T:])                 # terminal window (post-hoc only)
    lengths.append(m["num_steps"])
    y.append(0 if m["successful"] else 1)   # positive class = failure
    strat.append(f"{m.get('rollout_subtype')}_{m['successful']}")

Xfirst = np.array(Xfirst)
Xlast = np.array(Xlast)
lengths = np.array(lengths).reshape(-1, 1)
y = np.array(y)
print(f"episodes: {len(y)}  failures: {y.sum()}  successes: {(y == 0).sum()}")
print(f"feature: gripper width agent_pos[{GRIP_AP}], T={T} steps -> {T}-dim\n")


def auroc_logreg(X):
    """Stratified 80/20, logistic regression on X, AUROC on the holdout."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=strat
    )
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


print(f"AUROC  gripper FIRST {T} steps (early, online):      {auroc_logreg(Xfirst):.4f}")
print(f"AUROC  gripper LAST  {T} steps (terminal, post-hoc): {auroc_logreg(Xlast):.4f}")
print(f"AUROC  length-only leak baseline:                    {auroc_logreg(lengths):.4f}")
