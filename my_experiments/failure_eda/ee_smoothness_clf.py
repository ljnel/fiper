"""Predict stacking success/failure from end-effector position trajectory.

Input signal: the per-step "first action point" projected to the EE position
sub-space (action dims [0,1,2]), i.e. the executed EE-position command over time.

Hypothesis: successful trajectories are smoother than failures.

We engineer smoothness/geometry features per episode and train classifiers,
reporting AUROC on an 80/20 stratified split of the original test set.
"""
import glob, pickle, numpy as np
from scipy.fft import fft
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

unwrap = lambda x: x[0] if isinstance(x, list) else x
TASK = "stacking"
POS_IDX = [0, 1, 2]
SEED = 0


def ee_pos(path):
    ep = pickle.load(open(path, "rb"))
    first = np.stack([unwrap(s["action"])[0] for s in ep["rollout"]])  # (T, A)
    return first[:, POS_IDX].astype(np.float64), bool(ep["metadata"]["successful"])


def eval_auroc(X, y, names, label, drop=()):
    """Stratified 80/20, report eval AUROC for LogReg + HGB on given features."""
    keep = [i for i, n in enumerate(names) if n not in drop]
    Xk = X[:, keep]; kept = [names[i] for i in keep]
    Xtr, Xte, ytr, yte = train_test_split(
        Xk, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"\n=== {label} ===  ({len(kept)} features: {', '.join(kept)})")
    for nm, m in {
        "LogReg (scaled)": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)),
        "HistGradBoost": HistGradientBoostingClassifier(
            random_state=SEED, max_depth=3, learning_rate=0.05, max_iter=400),
    }.items():
        m.fit(Xtr, ytr)
        au = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
        print(f"  {nm:20s} eval AUROC = {au:.4f}")
    return Xte, yte, kept


def sparc(speed, fs=1.0, padlevel=4, fc=10.0, amp_th=0.05):
    """Spectral arc length: dimensionless smoothness of a speed profile.
    More negative = less smooth. (Balasubramanian et al. 2015)"""
    if len(speed) < 3 or speed.max() <= 0:
        return 0.0
    N = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    Mf = np.abs(fft(speed, N))
    Mf = Mf / Mf.max()
    freq = np.arange(N) * fs / N
    # cut off above fc, then trim to where magnitude exceeds amp_th
    mask = freq <= fc
    f, Mf = freq[mask], Mf[mask]
    inx = Mf >= amp_th
    if inx.sum() < 2:
        return 0.0
    fsel, Msel = f[inx], Mf[inx]
    df = np.diff(fsel) / (fsel[-1] - fsel[0] + 1e-12)
    dM = np.diff(Msel)
    return -np.sum(np.sqrt(df ** 2 + dM ** 2))


def features(pos):
    T = len(pos)
    v = np.diff(pos, axis=0)                      # per-step velocity (T-1, 3)
    speed = np.linalg.norm(v, axis=1)            # (T-1,)
    a = np.diff(v, axis=0)                        # acceleration
    j = np.diff(a, axis=0)                        # jerk
    jmag = np.linalg.norm(j, axis=1) if len(j) else np.zeros(1)
    amag = np.linalg.norm(a, axis=1) if len(a) else np.zeros(1)
    path_len = speed.sum() + 1e-12
    disp = np.linalg.norm(pos[-1] - pos[0])
    bbox = np.prod(pos.max(0) - pos.min(0) + 1e-9)
    mean_sp = speed.mean() + 1e-12
    # direction reversals: sign changes of speed derivative (stop-go choppiness)
    dsp = np.diff(speed)
    reversals = int(np.sum(np.diff(np.sign(dsp)) != 0)) if len(dsp) > 1 else 0
    # log dimensionless jerk (duration & amplitude normalized) — classic smoothness
    dur = T
    ldj = np.log((dur ** 3 / (mean_sp ** 2 + 1e-12)) * (jmag ** 2).mean() + 1e-12)
    return {
        # --- smoothness ---
        "jerk_rms": np.sqrt((jmag ** 2).mean()),
        "jerk_max": jmag.max(),
        "log_dimless_jerk": ldj,
        "sparc": sparc(speed),
        "accel_rms": np.sqrt((amag ** 2).mean()),
        "speed_cov": speed.std() / mean_sp,          # coefficient of variation
        "reversals_per_step": reversals / max(T, 1),
        # --- geometry ---
        "path_len": path_len,
        "straightness": disp / path_len,             # 1 = perfectly direct
        "bbox_vol": bbox,
        "disp": disp,
        # --- basic ---
        "duration": T,
        "speed_mean": mean_sp,
        "speed_max": speed.max(),
    }


def build(fs, trunc=None, side="first"):
    """Feature matrix. trunc=K -> use only K steps of each episode;
    side='first' keeps the first K, 'last' keeps the final K."""
    X, y, names = [], [], None
    for p in fs:
        pos, ok = ee_pos(p)
        if trunc is not None:
            pos = pos[:trunc] if side == "first" else pos[-trunc:]
        f = features(pos)
        if names is None:
            names = list(f)
        X.append([f[k] for k in names])
        y.append(int(ok))
    return np.asarray(X), np.asarray(y), names


def main():
    fs = sorted(glob.glob(f"data/{TASK}/rollouts/test/*.pkl"))

    # length-only baseline: failures time out, successes end early
    X, y, names = build(fs)
    print(f"{len(y)} episodes | success={y.sum()} fail={(1-y).sum()} "
          f"| base rate(success)={y.mean():.3f}")
    _, _, _ = eval_auroc(X[:, [names.index("duration")]], y, ["duration"],
                         "LENGTH-ONLY baseline (the confound)")

    # full features (includes length signal -> optimistic / confounded)
    eval_auroc(X, y, names, "ALL features (length leaks in)")

    # FAIR tests of the smoothness hypothesis: fixed-size windows so episode
    # length carries no information. first-K = approach phase; last-K = end phase.
    drop = ("duration", "log_dimless_jerk")  # length-encoding features
    for K in (12, 24, 40):
        for side in ("first", "last"):
            Xw, yw, nw = build(fs, trunc=K, side=side)
            eval_auroc(Xw, yw, nw,
                       f"{side.upper()}-{K} window (no length leak)", drop)

    # univariate detail for the strongest fair setting: last-24
    Xw, yw, nw = build(fs, trunc=24, side="last")
    Xte, yte, kept = eval_auroc(Xw, yw, nw, "LAST-24 detail", drop)
    print("\nPer-feature univariate AUROC (last-24, eval fold):")
    rows = [(k, roc_auc_score(yte, Xte[:, j])) for j, k in enumerate(kept)]
    for k, au in sorted(rows, key=lambda r: -abs(r[1] - 0.5)):
        print(f"  {k:22s} {au:.3f}  (|.5|={abs(au-0.5):.3f})")


if __name__ == "__main__":
    main()
