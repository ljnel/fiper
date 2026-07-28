"""Visual comparison: SPARC vs accel_rms as success/failure predictors
on the stacking last-24 EE-position window."""
import glob, numpy as np, importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

spec = importlib.util.spec_from_file_location("m", "my_experiments/ee_smoothness_clf.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

fs = sorted(glob.glob("data/stacking/rollouts/test/*.pkl"))
X, y, names = m.build(fs, trunc=24, side="last")
S = {n: X[:, names.index(n)] for n in ("sparc", "accel_rms")}

OK, BAD = "#2a9d8f", "#e76f51"   # success, failure
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

def dist_panel(a, vals, title):
    s, f = vals[y == 1], vals[y == 0]
    lo, hi = np.percentile(vals, [1, 99])
    bins = np.linspace(lo, hi, 35)
    a.hist(f, bins, density=True, alpha=.55, color=BAD, label="failure")
    a.hist(s, bins, density=True, alpha=.55, color=OK, label="success")
    for v, c in [(s, OK), (f, BAD)]:
        a.axvline(v.mean(), color=c, ls="--", lw=2)
    au = roc_auc_score(y, vals)
    a.set_title(f"{title}\nAUROC = {max(au,1-au):.3f}", fontsize=11)
    a.set_xlabel(title); a.set_ylabel("density"); a.legend(frameon=False)

dist_panel(ax[0], S["sparc"], "sparc")
dist_panel(ax[1], S["accel_rms"], "accel_rms")

# ROC panel
ax[2].plot([0, 1], [0, 1], color="gray", ls=":", lw=1)
for n, c in [("sparc", OK), ("accel_rms", "#264653")]:
    v = S[n]
    if roc_auc_score(y, v) < .5:   # orient so higher score = success
        v = -v
    fpr, tpr, _ = roc_curve(y, v)
    ax[2].plot(fpr, tpr, color=c, lw=2.2,
               label=f"{n}  (AUROC {roc_auc_score(y, v):.3f})")
ax[2].set_title("ROC — last-24 EE window", fontsize=11)
ax[2].set_xlabel("false positive rate"); ax[2].set_ylabel("true positive rate")
ax[2].legend(frameon=False, loc="lower right")

fig.suptitle("Stacking failure prediction from EE-position smoothness "
             "(last-24 window)", fontsize=13, y=1.02)
fig.tight_layout()
out = "my_experiments/sparc_vs_accel.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
