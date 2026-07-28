"""Sanity-check plot for block-centroid estimation (standalone; does NOT touch
the main predict_failure script). Saves a grid of overhead frames with the
estimated red/blue/green/cyan centroids marked, so the segmentation can be
eyeballed before we trust it as features.

Run:  pixi run python my_experiments/centroid_sanity.py
Out:  my_experiments/centroid_sanity.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb_to_hsv

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_initial_frames_cache.npz")
OUT = os.path.join(HERE, "centroid_sanity.png")
SCATTER_OUT = os.path.join(HERE, "centroid_scatter.png")

# marker draw colors per block
DRAW = dict(red="red", blue="blue", green="lime", cyan="cyan")


def centroids(img):
    """Return {color: (x, y) or None} from the 96x96 overhead frame."""
    hsv = rgb_to_hsv(img.astype(np.float32) / 255)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    rows = np.arange(96)[:, None]
    not_arm = rows > 14  # ignore the top strip where the robot gripper sits

    masks = {
        # hue in turns (0-1): red~0/1, green~0.33, cyan~0.5, blue~0.66
        "red": ((h < 0.05) | (h > 0.95)) & (s > 0.5) & (v > 0.4),
        "green": (h > 0.25) & (h < 0.45) & (s > 0.5) & (v > 0.4),
        # cyan: high saturation gate separates the vivid cube from the
        # washed-out cyan background gradient (the known gotcha)
        "cyan": (h > 0.45) & (h < 0.55) & (s > 0.45) & (v > 0.5),
        "blue": (h > 0.55) & (h < 0.75) & (s > 0.5) & (v > 0.3),
    }
    out = {}
    for c, m in masks.items():
        m = m & not_arm
        if m.sum() < 4:
            out[c] = None
            continue
        ys, xs = np.nonzero(m)
        out[c] = (xs.mean(), ys.mean())
    return out


def scatter_starting_positions(X, y):
    """Scatter every episode's block centroids over the 96x96 frame, colored by
    block. Success and failure are drawn in separate side-by-side panels (shared
    axes) so the two classes are directly comparable instead of overplotted."""
    from matplotlib.lines import Line2D

    OUTCOME = {0: "success", 1: "failure"}
    pts = {(c, o): [] for c in DRAW for o in OUTCOME}
    for img, lab in zip(X, y):
        for c, p in centroids(img).items():
            if p is not None:
                pts[(c, int(lab))].append(p)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), sharex=True, sharey=True)
    for o, ax in zip(OUTCOME, axes):
        n_eps = int((y == o).sum())
        for c, draw in DRAW.items():
            arr = np.array(pts[(c, o)])
            if len(arr) == 0:
                continue
            ax.scatter(arr[:, 0], arr[:, 1], s=20, alpha=0.35, color=draw,
                       edgecolors="none")
        ax.set_xlim(0, 96)
        ax.set_ylim(96, 0)        # image coords: origin top-left, y grows down
        ax.set_aspect("equal")
        ax.set_xlabel("x (px)")
        ax.set_title(f"{OUTCOME[o]}  (n={n_eps} episodes)")
    axes[0].set_ylabel("y (px)")

    color_handles = [Line2D([], [], marker="o", linestyle="none", color=d, label=c)
                     for c, d in DRAW.items()]
    axes[1].legend(handles=color_handles, title="block", loc="upper right",
                   framealpha=0.9)
    fig.suptitle("Initial block centroid positions by outcome", fontsize=13)
    fig.tight_layout()
    fig.savefig(SCATTER_OUT, dpi=110)
    print("saved", SCATTER_OUT)


def main():
    if not os.path.exists(CACHE):
        raise SystemExit(f"missing {CACHE} -- run predict_failure_from_initial_frame.py first")
    d = np.load(CACHE)
    X, y = d["X"], d["y"]

    # show a spread: first 6 successes and first 6 failures
    succ = np.where(y == 0)[0][:6]
    fail = np.where(y == 1)[0][:6]
    idx = np.concatenate([succ, fail])

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax, ep in zip(axes.flat, idx):
        img = X[ep]
        ax.imshow(img)
        for c, pt in centroids(img).items():
            if pt is None:
                continue
            ax.plot(pt[0], pt[1], "o", mfc="none", mec=DRAW[c], mew=2, ms=12)
            ax.plot(pt[0], pt[1], "+", color=DRAW[c], ms=8)
        ax.set_title(f"ep{ep}  {'FAIL' if y[ep] else 'success'}", fontsize=10)
        ax.axis("off")
    fig.suptitle("Estimated block centroids on first overhead frame (sanity check)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print("saved", OUT)

    scatter_starting_positions(X, y)


if __name__ == "__main__":
    main()
