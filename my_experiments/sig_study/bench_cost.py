"""Computational cost and memory footprint of the signature options.

Two regimes are measured separately because they answer different questions:

  latency    - one step at a time (B chunks), which is how FIPER's base loop
               and a real deployment call the detector.
  throughput - a whole episode batched, which is what an offline sweep does.

Also reports the analytic feature-tensor sizes, since for the wider action
spaces those, not the runtime, are the binding constraint.
"""
from __future__ import annotations

import sys, time

sys.path.insert(0, "/home/louis/fiper/my_experiments/sig_study")

import numpy as np
import pandas as pd
import torch

from sigtools import sig_dim, logsig_dim, augment, signature, sig_kernel_gram
from scores import SigConfig, sig_features
from episode_scores import sigvar_episode

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/home/louis/fiper/my_experiments/sig_study"

# (task, a_position, a_all, H, B)
SPECS = [
    ("sorting",    3,  6,  8,  32),
    ("stacking",   3, 21,  8,  32),
    ("push_t",     3,  3, 16, 256),
    ("pretzel",    3,  5, 16,  30),
    ("push_chair", 3,  3, 16, 256),
]
MEM_LIMIT = 12e9   # refuse to allocate more than this in the benchmark


def timeit(fn, n=20, warmup=3):
    for _ in range(warmup):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def bench_sizes():
    """Analytic tensor shapes - no computation, just the memory story."""
    rows = []
    for task, a_pos, a_all, H, B in SPECS:
        for name, a in (("position", a_pos), ("all", a_all)):
            for L in range(1, 7):
                d = a + 1                      # +1 for the time channel
                D = sig_dim(d, L)
                rows.append({
                    "task": task, "actions": name, "a": a, "d_time_aug": d,
                    "H": H, "B": B, "level": L,
                    "sig_dim": D, "logsig_dim": logsig_dim(d, L),
                    "bytes_per_step_fp32": D * B * 4,
                    "MB_per_step": D * B * 4 / 1e6,
                    "gram_BxB_MB": B * B * 4 / 1e6,
                })
    return pd.DataFrame(rows)


def bench_runtime():
    rows = []
    for task, a_pos, a_all, H, B in SPECS:
        for name, a in (("position", a_pos), ("all", a_all)):
            x1 = torch.randn(B, H, a, device=DEV) * 0.1              # one step
            xE = torch.randn(64, B, H, a, device=DEV) * 0.1          # 64 steps
            for L in range(1, 6):
                d = a + 1
                D = sig_dim(d, L)
                if D * B * 4 > MEM_LIMIT or D * 64 * B * 4 > MEM_LIMIT:
                    rows.append({"task": task, "actions": name, "level": L,
                                 "sig_dim": D, "infeasible": True})
                    print(f"  {task:10s} {name:8s} L{L} dim {D:9d} -- skipped "
                          f"({D*64*B*4/1e9:.0f} GB for a 64-step batch)", flush=True)
                    continue
                cfg = SigConfig(level=L, kernel="lin", time_aug=True)
                r = {"task": task, "actions": name, "a": a, "level": L,
                     "sig_dim": D, "B": B, "H": H}
                r["feat_1step_ms"] = 1e3 * timeit(lambda: sig_features(x1, cfg))
                r["sigvar_lin_1step_ms"] = 1e3 * timeit(
                    lambda: sigvar_episode(x1.unsqueeze(0), cfg))
                try:
                    r["sigvar_lin_64step_ms"] = 1e3 * timeit(
                        lambda: sigvar_episode(xE, cfg), n=5)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    r["sigvar_lin_64step_ms"] = float("nan")
                cfg_rbf = SigConfig(level=L, kernel="rbf", rbf_gamma=1.0, time_aug=True)
                r["sigvar_rbf_1step_ms"] = 1e3 * timeit(
                    lambda: sigvar_episode(x1.unsqueeze(0), cfg_rbf))
                rows.append(r)
                print(f"  {task:10s} {name:8s} L{L} dim {D:7d} "
                      f"feat {r['feat_1step_ms']:7.3f}ms  lin {r['sigvar_lin_1step_ms']:7.3f}ms  "
                      f"rbf {r['sigvar_rbf_1step_ms']:7.3f}ms", flush=True)
                torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def bench_pde():
    """Untruncated signature kernel: B x B Goursat solves per step."""
    rows = []
    for task, a_pos, a_all, H, B in SPECS:
        a = a_pos
        for Bsub in sorted({min(B, 32), min(B, 64), B}):
            x = torch.randn(Bsub, H, a, device=DEV) * 0.1
            for dy in (0, 1, 2):
                xa = augment(x, time=True)
                grid = (H - 1) * 2 ** dy
                mem = Bsub * Bsub * grid * grid * 4 * 2      # inc + u
                if mem > MEM_LIMIT:
                    rows.append({"task": task, "B": Bsub, "dyadic": dy,
                                 "infeasible": True, "est_GB": mem / 1e9})
                    print(f"  {task:10s} B={Bsub:3d} dy={dy} -- skipped "
                          f"({mem/1e9:.1f} GB)", flush=True)
                    continue
                try:
                    ms = 1e3 * timeit(lambda: sig_kernel_gram(xa, xa, dy), n=5)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    rows.append({"task": task, "B": Bsub, "dyadic": dy, "oom": True})
                    continue
                rows.append({"task": task, "B": Bsub, "dyadic": dy, "H": H,
                             "grid": grid, "gram_1step_ms": ms,
                             "peak_MB": mem / 1e6})
                print(f"  {task:10s} B={Bsub:3d} dy={dy} grid {grid:3d} "
                      f"{ms:9.3f} ms/step  ({mem/1e6:.0f} MB)", flush=True)
                torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def bench_cpu():
    """CPU cost via iisignature - the relevant number if the robot has no GPU."""
    import iisignature
    rows = []
    for task, a_pos, a_all, H, B in SPECS:
        for name, a in (("position", a_pos), ("all", a_all)):
            x = (np.random.randn(B, H, a + 1) * 0.1)
            for L in range(1, 6):
                if sig_dim(a + 1, L) > 3e5:
                    continue
                t0 = time.perf_counter()
                reps = 5
                for _ in range(reps):
                    for i in range(B):
                        iisignature.sig(x[i], L)
                ms = 1e3 * (time.perf_counter() - t0) / reps
                rows.append({"task": task, "actions": name, "a": a, "level": L,
                             "sig_dim": sig_dim(a + 1, L), "cpu_1step_ms": ms})
                print(f"  {task:10s} {name:8s} L{L} cpu {ms:8.2f} ms/step", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "sizes"):
        print("=== analytic sizes ===")
        df = bench_sizes(); df.to_csv(f"{OUT}/bench_sizes.csv", index=False)
    if which in ("all", "runtime"):
        print("=== GPU runtime ===")
        bench_runtime().to_csv(f"{OUT}/bench_runtime.csv", index=False)
    if which in ("all", "pde"):
        print("=== PDE signature kernel ===")
        bench_pde().to_csv(f"{OUT}/bench_pde.csv", index=False)
    if which in ("all", "cpu"):
        print("=== CPU (iisignature) ===")
        bench_cpu().to_csv(f"{OUT}/bench_cpu.csv", index=False)
