"""
Shareable summary figure for the gripper-signal failure experiment (stacking).

Three panels:
  A) gripper width aligned to episode START  -> success/failure overlap early
  B) gripper width aligned to episode END    -> they diverge (success holds cube)
  C) AUROC of a logistic regression on the first-20 vs last-20 gripper window,
     with episode length as a leak reference.

Run: pixi run python my_experiments/plot_gripper_summary.py
"""
import glob
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

T = 20          # window length
AP = 7          # agent_pos index = measured gripper width
SEED = 0
L = 40          # steps shown in the trajectory panels
HOLD = 0.062    # cube-width hold mode
GREEN, RED = "#2ca02c", "#d62728"

# --- load ---
gr, y, strat, lens = [], [], [], []
for f in sorted(glob.glob("data/stacking/rollouts/test/*.pkl")):
    d = pickle.load(open(f, "rb")); m = d["metadata"]
    gr.append(np.array([s["agent_pos"][AP] for s in d["rollout"]], float))
    y.append(0 if m["successful"] else 1)
    strat.append(f"{m.get('rollout_subtype')}_{m['successful']}")
    lens.append(m["num_steps"])
y = np.array(y); lens = np.array(lens).reshape(-1, 1)
SUC, FAIL = y == 0, y == 1


def align_start(arrs, L):
    M = np.full((len(arrs), L), np.nan)
    for i, a in enumerate(arrs): M[i, :min(L, len(a))] = a[:L]
    return M


def align_end(arrs, L):
    M = np.full((len(arrs), L), np.nan)
    for i, a in enumerate(arrs): M[i, L - min(L, len(a)):] = a[-L:]
    return M


def auroc(X):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=SEED, stratify=strat)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


a_first = auroc(np.array([a[:T] for a in gr]))
a_last = auroc(np.array([a[-T:] for a in gr]))
a_len = auroc(lens)

fig, ax = plt.subplots(1, 3, figsize=(16, 5))

# Panel A — aligned to start
Ms = align_start(gr, L); t = np.arange(L)
for mask, c, lab in [(SUC, GREEN, "Success"), (FAIL, RED, "Failure")]:
    mu, sd = np.nanmean(Ms[mask], 0), np.nanstd(Ms[mask], 0)
    ax[0].plot(t, mu, color=c, lw=2.5, label=lab); ax[0].fill_between(t, mu - sd, mu + sd, color=c, alpha=.12)
ax[0].axvspan(0, T, color="gray", alpha=.15); ax[0].axhline(HOLD, ls=":", c="k", lw=1, alpha=.5)
ax[0].text(T / 2, 0.045, f"first {T} steps\n(used by 'early')", ha="center", fontsize=9, color="dimgray")
ax[0].set_title("Aligned to START — early phase", fontsize=12, weight="bold")
ax[0].set_xlabel("Step from episode start"); ax[0].set_ylabel("Measured gripper width (m)")
ax[0].legend(loc="upper right"); ax[0].grid(alpha=.3)

# Panel B — aligned to end
Me = align_end(gr, L); te = np.arange(-L + 1, 1)
for mask, c, lab in [(SUC, GREEN, "Success"), (FAIL, RED, "Failure")]:
    mu, sd = np.nanmean(Me[mask], 0), np.nanstd(Me[mask], 0)
    ax[1].plot(te, mu, color=c, lw=2.5, label=lab); ax[1].fill_between(te, mu - sd, mu + sd, color=c, alpha=.12)
ax[1].axvspan(-T + 1, 0, color="gray", alpha=.15); ax[1].axhline(HOLD, ls=":", c="k", lw=1, alpha=.5)
ax[1].text(-T / 2, 0.0565, f"last {T} steps\n(used by 'terminal')", ha="center", fontsize=9, color="dimgray")
ax[1].annotate("holds cube ~0.062", (0, HOLD), (-18, 0.067), fontsize=9, color=GREEN,
               arrowprops=dict(arrowstyle="->", color=GREEN))
ax[1].set_title("Aligned to END — terminal phase", fontsize=12, weight="bold")
ax[1].set_xlabel("Step from episode end"); ax[1].set_ylabel("Measured gripper width (m)")
ax[1].legend(loc="upper left"); ax[1].grid(alpha=.3)

# Panel C — AUROC bars
labels = ["Early\n(first 20)", "Terminal\n(last 20)", "Length\n(leak ref)"]
vals = [a_first, a_last, a_len]
bars = ax[2].bar(labels, vals, color=["#9e9e9e", "#1f77b4", "#d6a800"], edgecolor="k")
ax[2].axhline(0.5, ls="--", c="k", lw=1); ax[2].text(2.35, 0.515, "chance", fontsize=9)
for r, v in zip(bars, vals):
    ax[2].text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", weight="bold")
ax[2].set_ylim(0, 1.08); ax[2].set_ylabel("AUROC (failure prediction)")
ax[2].set_title("Gripper signal: when is it informative?", fontsize=12, weight="bold")
ax[2].grid(axis="y", alpha=.3)

fig.suptitle("Stacking: gripper width predicts failure only in the terminal phase  "
             "(N=800 test episodes, logistic regression, 80/20 holdout)", fontsize=13, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = "/home/louis/fiper/my_experiments/gripper_failure_summary.png"
plt.savefig(out, dpi=140)
print("saved", out, "| AUROC first/last/len:", round(a_first, 3), round(a_last, 3), round(a_len, 3))
