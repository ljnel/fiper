"""Per-step uncertainty scores built from action-chunk path signatures.

At step t a policy emits B sampled action chunks, i.e. B paths of length H in
R^a.  Three ways to turn that set of paths into one scalar:

  sigvar   : kernel variance of the chunk distribution in signature space -
             a signature analogue of FIPER's action-chunk entropy (ACE).
  sigtc    : MMD^2 between the *overlapping* parts of the chunk distributions
             at step t-k and step t - a signature analogue of FIPER's `tc`.
  sigmmd   : MMD^2 between the chunk distribution at step t and a fixed
             reference set of calibration chunks.

Each can be computed with a linear kernel on truncated signature features, an
RBF kernel on truncated signature features, or the untruncated signature
kernel (PDE solve).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from sigtools import augment, signature, signature_normalise, sig_kernel_gram, sig_dim


@dataclass
class SigConfig:
    level: int = 3
    time_aug: bool = True
    basepoint: bool = False
    lead_lag: bool = False
    normalise: bool = False           # Chevyrev-Oberhauser tensor normalisation
    level_scale: float | None = None  # theta; level k scaled by theta**k
    path_scale: float = 1.0           # dilation of the state channels
    kernel: Literal["lin", "rbf", "pde"] = "lin"
    rbf_gamma: float | str = "median"
    dyadic_order: int = 1             # PDE refinement (kernel == "pde")
    increments: bool = False          # signature of the *increment* path
    dtype: torch.dtype = torch.float32

    def tag(self) -> str:
        bits = [f"L{self.level}", self.kernel]
        if self.time_aug:
            bits.append("t")
        if self.basepoint:
            bits.append("bp")
        if self.lead_lag:
            bits.append("ll")
        if self.normalise:
            bits.append("nrm")
        if self.level_scale is not None:
            bits.append(f"ls{self.level_scale:g}")
        return "-".join(bits)

    def feat_dim(self, a: int) -> int:
        d = a * (2 if self.lead_lag else 1) + (1 if self.time_aug else 0)
        return sig_dim(d, self.level)


# --------------------------------------------------------------------------
def _prep(paths: torch.Tensor, cfg: SigConfig) -> torch.Tensor:
    """(N,H,a) raw chunks -> augmented paths ready for signature/PDE."""
    if cfg.increments:
        paths = torch.cat([torch.zeros_like(paths[:, :1]),
                           paths[:, 1:] - paths[:, :-1]], dim=1).cumsum(dim=1)
    return augment(paths, time=cfg.time_aug, basepoint=cfg.basepoint,
                   lead_lag=cfg.lead_lag, scale=cfg.path_scale)


def sig_features(paths: torch.Tensor, cfg: SigConfig) -> torch.Tensor:
    """(N,H,a) -> (N, D) signature features."""
    x = _prep(paths, cfg)
    levels = signature(x, cfg.level, flatten=False, level_scale=cfg.level_scale)
    if cfg.normalise:
        levels = signature_normalise(levels)
    return torch.cat(levels, dim=-1)


def _median_gamma(sq: torch.Tensor) -> float:
    v = sq[sq > 0]
    if v.numel() == 0:
        return 1.0
    med = torch.median(v)
    return float(1.0 / (2.0 * med)) if med > 0 else 1.0


def gram(paths_x: torch.Tensor, paths_y: torch.Tensor | None, cfg: SigConfig):
    """Return (Kxx, Kyy, Kxy) for the configured kernel.  paths_y may be None
    (then only Kxx is meaningful and the others are None)."""
    if cfg.kernel == "pde":
        X = _prep(paths_x, cfg)
        if paths_y is None:
            return sig_kernel_gram(X, X, cfg.dyadic_order), None, None
        Y = _prep(paths_y, cfg)
        return (sig_kernel_gram(X, X, cfg.dyadic_order),
                sig_kernel_gram(Y, Y, cfg.dyadic_order),
                sig_kernel_gram(X, Y, cfg.dyadic_order))

    SX = sig_features(paths_x, cfg)
    SY = sig_features(paths_y, cfg) if paths_y is not None else None
    if cfg.kernel == "lin":
        Kxx = SX @ SX.T
        if SY is None:
            return Kxx, None, None
        return Kxx, SY @ SY.T, SX @ SY.T

    # rbf on signature features
    Z = SX if SY is None else torch.cat([SX, SY], 0)
    sq = torch.cdist(Z, Z).pow(2)
    g = _median_gamma(sq) if isinstance(cfg.rbf_gamma, str) else float(cfg.rbf_gamma)
    K = torch.exp(-g * sq)
    n = SX.shape[0]
    if SY is None:
        return K, None, None
    return K[:n, :n], K[n:, n:], K[:n, n:]


# --------------------------------------------------------------------------
# score families
# --------------------------------------------------------------------------
def score_sigvar(chunks: torch.Tensor, cfg: SigConfig) -> float:
    """Kernel variance of the chunk distribution:  E k(X,X) - E k(X,X').
    Equals (1/2) E||phi(X)-phi(X')||^2, the natural spread of the distribution
    of paths in the signature RKHS."""
    K, _, _ = gram(chunks, None, cfg)
    return float(K.diagonal().mean() - K.mean())


def mmd2(Kxx, Kyy, Kxy, unbiased: bool = False) -> float:
    if not unbiased:
        return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())
    n, m = Kxx.shape[0], Kyy.shape[0]
    sx = (Kxx.sum() - Kxx.diagonal().sum()) / (n * (n - 1))
    sy = (Kyy.sum() - Kyy.diagonal().sum()) / (m * (m - 1))
    return float(sx + sy - 2.0 * Kxy.mean())


def score_sigtc(prev: torch.Tensor, curr: torch.Tensor, cfg: SigConfig,
                unbiased: bool = False) -> float:
    """MMD^2 between the overlapping tails/heads of two consecutive chunk
    distributions (already sliced by the caller)."""
    Kxx, Kyy, Kxy = gram(prev, curr, cfg)
    return mmd2(Kxx, Kyy, Kxy, unbiased)


def score_sigmmd(chunks: torch.Tensor, ref: torch.Tensor, cfg: SigConfig,
                 unbiased: bool = False) -> float:
    Kxx, Kyy, Kxy = gram(chunks, ref, cfg)
    return mmd2(Kxx, Kyy, Kxy, unbiased)
