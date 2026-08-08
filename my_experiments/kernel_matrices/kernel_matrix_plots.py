"""Kernel-matrix plots: does an action kernel add anything to the observation kernel?

Draws the Gram matrices of the kernels used by ``my_methods/methods`` over a sample of
test steps ordered success-block-then-failure-block, so that any block structure the
kernel sees is directly visible. One step per test episode, frozen at ``--step-frac``
(default 80%) of episode completion; only the choice of episodes is random.

Two figures per task, three panels each::

    plot 1   K_obs   |  K_sum (time-aligned sum kernel)  |  K_obs * K_sum
    plot 2   K_obs   |  K_sig (mean-signature kernel)    |  K_obs * K_sig

Kernels (hyperparameters follow ``my_methods/methods``; everything data-dependent is fit
on the **calibration** split, exactly as the methods do, and only then evaluated on the
sampled test steps):

``K_obs``   RBF on the raw observation embeddings with the median-heuristic bandwidth
            ``gamma = 1/(2 median^2)``: the kernel of ``kern_cd_rbf``, and the
            observation factor of ``kern_cd_prod``.

``K_sum``   The action factor of ``kern_cd_prod`` verbatim. Each step's chunk batch
            ``A`` of shape (B, H, D) is mapped to ``mu_A = 1/(B sqrt(H)) sum_b (+)_h
            phi(A_bh)``, the direct sum over the horizon of an RFF map approximating
            ``exp(-gamma_a ||u-v||^2)``, normalised to unit norm; ``K_sum = <mu_hat,
            mu_hat'>``. The implied per-chunk kernel is the time-aligned sum kernel
            ``sum_h exp(-gamma_a ||x_h - y_h||^2)``, pooled over the B chunks by a mean
            embedding. ``K_obs * K_sum`` is therefore exactly the ``kern_cd_prod``
            kernel.

``K_sig``   The action block of ``kern_cd_rbf_sig``: the level-``sig_level`` truncated
            signature of each time-augmented, path-dilated chunk, averaged over the B
            chunks, standardised with calibration statistics and dimension-balanced by
            ``1/sqrt(d)``; then an RBF over that block (``--sig-kernel rbf``, default) or
            the normalised linear kernel on the raw mean signatures (``--sig-kernel
            linear``, the plain truncated signature kernel of the chunk-batch mean
            embedding). ``K_obs * K_sig`` is ``kern_cd_rbf_sig``'s joint kernel up to one
            deliberate change: the bandwidth is fit per block by the median heuristic
            rather than once on the concatenation, so each factor is individually well
            scaled and the same ``K_obs`` appears in both figures.

Both action kernels have a unit diagonal, so the products do too and all three panels of
a figure live on the same [0, 1] scale.

Usage::

    python -m my_experiments.kernel_matrix_plots --tasks push_t pretzel --n-steps 100 --seed 0

Note on ``pretzel``: its test split holds only 20 episodes (10 successful, 10 failed), so
under the one-step-per-episode rule its matrix is 20x20 rather than 100x100, and its
calibration split is 10 episodes, all successful.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT_DIR = str(HERE.parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
_SIG_DIR = os.path.join(ROOT_DIR, "my_experiments", "sig_study")
if _SIG_DIR not in sys.path:
    sys.path.insert(0, _SIG_DIR)

from cd.algs.kernels import RBF  # noqa: E402
from datasets.rollout_datasets import ProcessedRolloutDataset  # noqa: E402
from sigtools import augment, signature  # noqa: E402

#: Cap on elements in one RFF intermediate (N*B*H*d), bounding peak memory. Mirrors
#: kern_cd_prod._RFF_ELEM_BUDGET.
RFF_ELEM_BUDGET = 32_000_000
#: Cap on paths per signature() call. Mirrors _KernCDJoint._SIG_BATCH.
SIG_BATCH = 65_536
STD_EPS = 1e-8
#: Points subsampled for the action-space median heuristic (kern_cd_prod._setup_rff).
GAMMA_ACT_SAMPLE = 20_000


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_task(task: str, data_dir: str, config_dir: str) -> ProcessedRolloutDataset:
    ds = ProcessedRolloutDataset(
        task_data_path=os.path.join(data_dir, task),
        base_config_path=config_dir,
        required_tensors=["obs_embeddings", "action_preds"],
    )
    ds.load_dataset()
    assert ds.dataset_loaded, f"no processed rollout cache for task '{task}'"
    # load_dataset() keys tensors as "<name>.pt"; the rest of the codebase wants the
    # bare name (same fix as my_experiments/sig_study/loader.py).
    for k in list(ds.data.keys()):
        if k.endswith(".pt"):
            ds.data[k[:-3]] = ds.data.pop(k)
    return ds


def subset_step_indices(ds: ProcessedRolloutDataset, subset: str) -> np.ndarray:
    """Flat step indices of every step in `subset`, in episode order."""
    starts, ends, _ = ds._filter_start_end_episode_indices(subset=subset)
    return ds._get_slices_from_indices(starts, ends)


def sample_test_steps(
    ds: ProcessedRolloutDataset,
    subset: str,
    n_steps: int,
    step_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """One step per episode from ~n_steps/2 successful and ~n_steps/2 failed episodes.

    Only *which* episodes are used is random; the step itself is frozen at `step_frac` of
    episode completion, so every row of the matrix sits at the same relative point in its
    rollout and any block structure cannot be an artefact of comparing early steps with
    late ones. No episode contributes twice, so a block is short if the subset does not
    hold enough episodes -- pretzel has only 20 test episodes, giving a 20x20 matrix.

    Returns the flat step indices and the size of the success block.
    """
    starts, ends, labels = ds._filter_start_end_episode_indices(subset=subset)
    labels = np.asarray(labels).astype(bool)

    n_fail = n_steps // 2
    n_success = n_steps - n_fail
    picked = []
    for kind, want, mask in (("successful", n_success, labels), ("failed", n_fail, ~labels)):
        episodes = np.flatnonzero(mask)
        if len(episodes) < want:
            print(
                f"warning: {subset} has only {len(episodes)} {kind} episodes; "
                f"using {len(episodes)} of the requested {want} steps"
            )
            want = len(episodes)
        chosen = rng.choice(episodes, size=want, replace=False)
        # Completion is measured over [0, T-1], so step_frac=1.0 is the final step.
        steps = [int(s) + int(round((int(t) - int(s) - 1) * step_frac)) for s, t in zip(starts[chosen], ends[chosen])]
        picked.append(np.sort(np.asarray(steps, dtype=np.int64)))

    return np.concatenate(picked), len(picked[0])


def gather(ds: ProcessedRolloutDataset, idx: np.ndarray, device: str):
    """(obs [n, d_obs] float64, chunks [n, B, H, 3] float32 on `device`) for flat steps."""
    obs = np.asarray(ds.data["obs_embeddings"][idx]).astype(np.float64)
    act_idx = ds._get_action_slices(required_actions=["position"], optional_actions=[])
    ap = ds.data["action_preds"][idx]
    if ap.ndim == 5:  # [N, robots, B, H, A]
        assert ap.shape[1] == 1, "multi-robot rollouts are not supported"
        ap = ap[:, 0]
    chunks = torch.as_tensor(np.asarray(ap[..., act_idx]), dtype=torch.float32, device=device)
    return obs, chunks


# ---------------------------------------------------------------------------
# action kernels
# ---------------------------------------------------------------------------
class SumKernel:
    """kern_cd_prod's action factor: normalised mean embedding of the time-aligned sum
    kernel over a chunk batch, approximated with random Fourier features."""

    def __init__(self, rff_dim: int, gamma_scale: float, seed: int):
        self.rff_dim = int(rff_dim)
        self.gamma_scale = float(gamma_scale)
        self.rng = np.random.default_rng(seed)

    def fit(self, chunks: torch.Tensor) -> "SumKernel":
        D = chunks.shape[-1]
        pts = chunks.reshape(-1, D)
        n = pts.shape[0]
        s = min(n, GAMMA_ACT_SAMPLE)
        # median_distance subsamples to 3000 on the host, so pull a bounded slice off the
        # device first rather than moving all N*B*H points.
        sel = self.rng.choice(n, size=s, replace=False) if n > s else np.arange(n)
        sample = pts[torch.as_tensor(sel, device=pts.device)].cpu().numpy().astype(np.float64)
        self.gamma = RBF(gamma="median").fit(sample).gamma * self.gamma_scale

        d = self.rff_dim
        W = self.rng.standard_normal((D, d)) * np.sqrt(2.0 * self.gamma)
        b = self.rng.uniform(0.0, 2.0 * np.pi, size=d)
        self._W = torch.as_tensor(W, dtype=chunks.dtype, device=chunks.device)
        self._b = torch.as_tensor(b, dtype=chunks.dtype, device=chunks.device)
        return self

    def embed(self, chunks: torch.Tensor) -> np.ndarray:
        """[N, B, H, D] -> [N, H*d] unit-norm chunk-batch embeddings."""
        N, B, H, D = chunks.shape
        d = self.rff_dim
        out = np.empty((N, H * d), dtype=np.float64)
        rows = max(1, RFF_ELEM_BUDGET // (B * H * d))
        for s in range(0, N, rows):
            blk = chunks[s : s + rows]
            n = blk.shape[0]
            u = blk.reshape(-1, D)
            phi = (np.sqrt(2.0 / d) * torch.cos(u @ self._W + self._b)).reshape(n, B, H, d)
            # Direct sum over h: the inner product only ever pairs equal timesteps.
            out[s : s + n] = (phi.sum(dim=1) / (B * np.sqrt(H))).reshape(n, H * d).cpu().numpy()
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)

    def gram(self, chunks: torch.Tensor) -> np.ndarray:
        mu = self.embed(chunks)
        return mu @ mu.T


class SignatureKernel:
    """kern_cd_rbf_sig's action block: mean level-`level` truncated signature of the
    time-augmented, path-dilated chunks, then an RBF (or normalised linear) kernel."""

    def __init__(self, level: int, time_aug: bool, kernel: str, gamma_scale: float):
        self.level = int(level)
        self.time_aug = bool(time_aug)
        self.kernel = kernel
        self.gamma_scale = float(gamma_scale)

    def fit(self, chunks: torch.Tensor) -> "SignatureKernel":
        # Path dilation: scale the position channels so the median chunk total variation
        # is 1, otherwise the higher signature levels are numerically swamped.
        x = chunks.reshape(-1, chunks.shape[-2], chunks.shape[-1])
        tv = float((x[:, 1:] - x[:, :-1]).norm(dim=-1).sum(-1).median())
        self.theta = 1.0 / tv if tv > 0 else 1.0

        mu = self._mean_signature(chunks)
        self._mean, self._std = mu.mean(0), mu.std(0) + STD_EPS
        if self.kernel == "rbf":
            self.gamma = RBF(gamma="median").fit(self._standardise(mu)).gamma * self.gamma_scale
        return self

    def _mean_signature(self, chunks: torch.Tensor) -> np.ndarray:
        """[N, B, H, D] -> [N, F] mean truncated signature over the B chunks."""
        N, B = chunks.shape[0], chunks.shape[1]
        paths = chunks.reshape(N * B, chunks.shape[-2], chunks.shape[-1])
        out = []
        for s in range(0, paths.shape[0], SIG_BATCH):
            aug = augment(paths[s : s + SIG_BATCH], time=self.time_aug, scale=self.theta)
            out.append(signature(aug, self.level, flatten=True))
        sig = torch.cat(out, dim=0) if len(out) > 1 else out[0]
        return sig.reshape(N, B, -1).mean(dim=1).cpu().numpy().astype(np.float64)

    def _standardise(self, mu: np.ndarray) -> np.ndarray:
        # Per-block standardisation + 1/sqrt(dim) balancing, as in _KernCDJoint._assemble_Z.
        z = (mu - self._mean) / self._std
        return z / np.sqrt(z.shape[1])

    def gram(self, chunks: torch.Tensor) -> np.ndarray:
        mu = self._mean_signature(chunks)
        if self.kernel == "rbf":
            return RBF(gamma=self.gamma)(self._standardise(mu))
        # Normalised linear signature kernel; unit diagonal, like the RBF variant.
        mu = mu / np.maximum(np.linalg.norm(mu, axis=1, keepdims=True), 1e-12)
        return mu @ mu.T


# ---------------------------------------------------------------------------
# metrics + plotting
# ---------------------------------------------------------------------------
def block_stats(K: np.ndarray, n_success: int) -> dict[str, float]:
    """Within-block (diagonal excluded) and cross-block mean similarity."""

    def off_diag_mean(M: np.ndarray) -> float:
        return float(M[~np.eye(len(M), dtype=bool)].mean())

    return {
        "ss": off_diag_mean(K[:n_success, :n_success]),
        "ff": off_diag_mean(K[n_success:, n_success:]),
        "sf": float(K[:n_success, n_success:].mean()),
    }


def plot_panels(panels, n_success: int, suptitle: str, out_base: str) -> None:
    """One figure of len(panels) kernel-matrix heatmaps. panels: [(title, K), ...]."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.1 * len(panels), 5.4), constrained_layout=True)
    for ax, (title, K) in zip(np.atleast_1d(axes), panels):
        stats = block_stats(K, n_success)
        # The diagonal is 1 by construction for every kernel here, so it carries no
        # information and would only pull the eye and the colour scale. Drop it.
        shown = np.where(np.eye(len(K), dtype=bool), np.nan, K)
        # Sequential single hue, light -> dark: the values are a magnitude in [0, 1].
        # Robust per-panel limits, because the interesting contrast is often a narrow
        # band near 1 that a fixed [0, 1] scale would flatten.
        lo, hi = np.nanpercentile(shown, [1, 99])
        cmap = matplotlib.colormaps["Blues"].with_extremes(bad="white")
        im = ax.imshow(shown, cmap=cmap, vmin=lo, vmax=min(hi, 1.0), interpolation="nearest")
        for pos in (n_success - 0.5,):
            ax.axhline(pos, color="0.25", lw=1.0)
            ax.axvline(pos, color="0.25", lw=1.0)
        ax.set_title(
            f"{title}\n"
            f"within success {stats['ss']:.3f}   within failure {stats['ff']:.3f}\n"
            f"cross success/failure {stats['sf']:.3f}   "
            f"gap {0.5 * (stats['ss'] + stats['ff']) - stats['sf']:+.3f}",
            fontsize=10,
        )
        n = len(K)
        ax.set_xticks([n_success / 2, (n_success + n) / 2])
        ax.set_xticklabels(["success", "failure"], fontsize=9)
        ax.set_yticks([n_success / 2, (n_success + n) / 2])
        ax.set_yticklabels(["success", "failure"], fontsize=9, rotation=90, va="center")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_color("0.8")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, length=2, color="0.6")
        print(
            f"  {title:<44s} SS {stats['ss']:.3f}  FF {stats['ff']:.3f}  SF {stats['sf']:.3f}  "
            f"gap {0.5 * (stats['ss'] + stats['ff']) - stats['sf']:+.3f}"
        )

    fig.suptitle(
        f"{suptitle}\ndiagonal masked (=1 for every kernel); colour scale is per panel, "
        "1st-99th percentile; similarities are means over the block",
        fontsize=11,
    )
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"wrote {out_base}.pdf / .png")


# ---------------------------------------------------------------------------
def run_task(task: str, args, device: str, t0: float) -> None:
    """Fit the three kernels on `task`'s calibration split and write its two figures."""
    rng = np.random.default_rng(args.seed)
    ds = load_task(task, args.data_dir, args.config_dir)

    # -- fit everything data-dependent on the fit (calibration) split ------------
    fit_idx = subset_step_indices(ds, args.fit_subset)
    if args.fit_max_steps and len(fit_idx) > args.fit_max_steps:
        fit_idx = np.sort(rng.choice(fit_idx, size=args.fit_max_steps, replace=False))
    fit_obs, fit_chunks = gather(ds, fit_idx, device)
    print(f"[{time.time() - t0:5.1f}s] {task}: fit on {len(fit_idx)} {args.fit_subset} steps, chunks {tuple(fit_chunks.shape)}")

    obs_kernel = RBF(gamma=RBF(gamma="median").fit(fit_obs).gamma * args.gamma_obs_scale)
    sum_kernel = SumKernel(args.rff_dim, args.gamma_act_scale, args.seed).fit(fit_chunks)
    sig_kernel = SignatureKernel(
        args.sig_level, not args.no_sig_time_aug, args.sig_kernel, args.gamma_sig_scale
    ).fit(fit_chunks)
    print(f"[{time.time() - t0:5.1f}s] {task}: gamma_obs {obs_kernel.gamma:.4g}  gamma_act {sum_kernel.gamma:.4g}  theta {sig_kernel.theta:.4g}")
    del fit_chunks

    # -- evaluate on the sampled test steps --------------------------------------
    idx, n_success = sample_test_steps(ds, args.subset, args.n_steps, args.step_frac, rng)
    obs, chunks = gather(ds, idx, device)
    print(f"[{time.time() - t0:5.1f}s] {task}: {len(idx)} {args.subset} steps, {n_success} success / {len(idx) - n_success} failure")

    K_obs = obs_kernel(obs)
    K_sum = sum_kernel.gram(chunks)
    K_sig = sig_kernel.gram(chunks)

    head = (
        f"{task}, {len(idx)} test steps (one per episode, frozen at {args.step_frac:.0%} "
        f"episode completion), episode draw seed {args.seed}"
    )
    plot_panels(
        [
            ("observation kernel  RBF(obs)", K_obs),
            (f"sum kernel  time-aligned, {args.rff_dim} RFF", K_sum),
            ("product  RBF(obs) x sum", K_obs * K_sum),
        ],
        n_success,
        f"Observation x time-aligned sum kernel on action chunks — {head}",
        os.path.join(args.out_dir, f"kernel_matrix_sum_{task}_seed{args.seed}"),
    )
    plot_panels(
        [
            ("observation kernel  RBF(obs)", K_obs),
            (f"signature kernel  level {args.sig_level}, {args.sig_kernel}", K_sig),
            ("product  RBF(obs) x signature", K_obs * K_sig),
        ],
        n_success,
        f"Observation x signature kernel on action chunks — {head}",
        os.path.join(args.out_dir, f"kernel_matrix_sig_{task}_seed{args.seed}"),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=["push_t", "pretzel"], help="tasks to plot, two figures each")
    p.add_argument("--n-steps", type=int, default=100, help="rows/columns of the kernel matrix")
    p.add_argument("--seed", type=int, default=0, help="seed for step sampling and the RFF draw")
    p.add_argument(
        "--step-frac",
        type=float,
        default=0.8,
        help="episode completion the single step per episode is taken at (0 = first step, 1 = last)",
    )
    p.add_argument("--subset", default="test", help="subset the plotted steps are drawn from")
    p.add_argument("--fit-subset", default="calibration", help="subset the kernel hyperparameters are fit on")
    p.add_argument("--fit-max-steps", type=int, default=0, help="cap on fit steps (0 = all); bounds runtime on big tasks")
    p.add_argument("--rff-dim", type=int, default=256, help="random Fourier features for the sum kernel")
    p.add_argument("--gamma-obs-scale", type=float, default=1.0, help="multiplier on the observation median heuristic")
    p.add_argument("--gamma-act-scale", type=float, default=1.0, help="multiplier on the action-space median heuristic")
    p.add_argument("--gamma-sig-scale", type=float, default=1.0, help="multiplier on the signature median heuristic")
    p.add_argument("--sig-level", type=int, default=2, help="truncated signature level")
    p.add_argument("--sig-kernel", choices=["rbf", "linear"], default="rbf", help="kernel over the mean signatures")
    p.add_argument("--no-sig-time-aug", action="store_true", help="drop the signature time channel")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--data-dir", default=os.path.join(ROOT_DIR, "data"))
    p.add_argument("--config-dir", default=os.path.join(ROOT_DIR, "configs"))
    p.add_argument("--out-dir", default=str(HERE))
    args = p.parse_args()

    t0 = time.time()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    os.makedirs(args.out_dir, exist_ok=True)

    for task in args.tasks:
        run_task(task, args, device, t0)
    print(f"[{time.time() - t0:5.1f}s] done")


if __name__ == "__main__":
    main()
