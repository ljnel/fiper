"""Batched per-episode scoring.

FIPER's base loop calls the scorer once per timestep; that is fine for a
deployed detector but hopeless for a sweep, so everything here is batched over
the timesteps of an episode (and chunked to bound peak memory).  The values are
identical to the per-step formulas in `scores.py` - `check_batched_equals_perstep`
in run_study.py asserts that.
"""
from __future__ import annotations

import numpy as np
import torch

from scores import SigConfig, _prep, sig_features
from sigtools import sig_kernel_gram


def _feats(ap: torch.Tensor, cfg: SigConfig, block: int = 4096) -> torch.Tensor:
    """ap: (T,B,H,a) -> (T,B,D) signature features, computed in blocks."""
    T, B, H, a = ap.shape
    flat = ap.reshape(T * B, H, a)
    out = []
    for i in range(0, flat.shape[0], block):
        out.append(sig_features(flat[i:i + block], cfg))
    return torch.cat(out, 0).reshape(T, B, -1)


def _pde_gram(x: torch.Tensor, y: torch.Tensor, cfg: SigConfig,
              block: int = 8) -> torch.Tensor:
    """x: (T,n,H,a) y: (T,m,H,a) -> (T,n,m), blocked over T."""
    outs = []
    for i in range(0, x.shape[0], block):
        outs.append(sig_kernel_gram(_prep(x[i:i + block], cfg),
                                    _prep(y[i:i + block], cfg),
                                    cfg.dyadic_order))
    return torch.cat(outs, 0)


def _rbf_from_feats(S: torch.Tensor, gamma) -> torch.Tensor:
    """S: (T,B,D) -> (T,B,B) RBF Gram."""
    sq = torch.cdist(S, S).pow(2)
    if gamma == "median":
        med = sq.flatten(1).median(dim=1).values.clamp_min(1e-30)
        g = (1.0 / (2.0 * med)).view(-1, 1, 1)
    else:
        g = float(gamma)
    return torch.exp(-g * sq)


# --------------------------------------------------------------------------
def sigvar_episode(ap: torch.Tensor, cfg: SigConfig) -> np.ndarray:
    """Kernel variance of the chunk distribution at every step. ap: (T,B,H,a)."""
    if cfg.kernel == "pde":
        K = _pde_gram(ap, ap, cfg)
        return (K.diagonal(dim1=-2, dim2=-1).mean(-1) - K.mean((-2, -1))).cpu().numpy()
    S = _feats(ap, cfg)
    if cfg.kernel == "lin":
        # E||S||^2 - ||E S||^2, no Gram needed
        return (S.pow(2).sum(-1).mean(-1) - S.mean(1).pow(2).sum(-1)).cpu().numpy()
    K = _rbf_from_feats(S, cfg.rbf_gamma)
    return (K.diagonal(dim1=-2, dim2=-1).mean(-1) - K.mean((-2, -1))).cpu().numpy()


def _mmd2_from_grams(Kxx, Kyy, Kxy, unbiased=False):
    if not unbiased:
        return Kxx.mean((-2, -1)) + Kyy.mean((-2, -1)) - 2 * Kxy.mean((-2, -1))
    n, m = Kxx.shape[-1], Kyy.shape[-1]
    sx = (Kxx.sum((-2, -1)) - Kxx.diagonal(dim1=-2, dim2=-1).sum(-1)) / (n * (n - 1))
    sy = (Kyy.sum((-2, -1)) - Kyy.diagonal(dim1=-2, dim2=-1).sum(-1)) / (m * (m - 1))
    return sx + sy - 2 * Kxy.mean((-2, -1))


def sigtc_episode(ap: torch.Tensor, cfg: SigConfig, exec_h: int,
                  backtrack: int = 1, unbiased: bool = False) -> np.ndarray:
    """Signature-MMD between the overlapping parts of the chunk distributions
    predicted `backtrack` execution-horizons apart.  Mirrors TCEval's slicing:
    prev[:, exec_h*bt:] vs curr[:, :-exec_h*bt].  Step 0..bt-1 get score 0."""
    T, B, H, a = ap.shape
    off = exec_h * backtrack
    assert off < H, "backtrack exceeds prediction horizon"
    if T <= backtrack:
        return np.zeros(T)
    prev = ap[:-backtrack, :, off:, :]
    curr = ap[backtrack:, :, :H - off, :]

    if cfg.kernel == "pde":
        Kxx = _pde_gram(prev, prev, cfg)
        Kyy = _pde_gram(curr, curr, cfg)
        Kxy = _pde_gram(prev, curr, cfg)
    else:
        Sp, Sc = _feats(prev, cfg), _feats(curr, cfg)
        if cfg.kernel == "lin":
            Kxx = Sp @ Sp.transpose(-2, -1)
            Kyy = Sc @ Sc.transpose(-2, -1)
            Kxy = Sp @ Sc.transpose(-2, -1)
        else:
            Z = torch.cat([Sp, Sc], dim=1)
            K = _rbf_from_feats(Z, cfg.rbf_gamma)
            Kxx, Kyy, Kxy = K[:, :B, :B], K[:, B:, B:], K[:, :B, B:]
    vals = _mmd2_from_grams(Kxx, Kyy, Kxy, unbiased).cpu().numpy()
    return np.concatenate([np.zeros(backtrack), vals])


def sigmmd_episode(ap: torch.Tensor, ref: torch.Tensor, cfg: SigConfig,
                   ref_feats: torch.Tensor | None = None,
                   unbiased: bool = False) -> np.ndarray:
    """Signature-MMD between each step's chunk distribution and a fixed
    reference set `ref` (R,H,a) pooled from calibration rollouts."""
    T, B, H, a = ap.shape
    if cfg.kernel == "pde":
        R = ref.shape[0]
        Kxx = _pde_gram(ap, ap, cfg)
        refb = ref.unsqueeze(0).expand(T, R, H, a)
        Kyy = _pde_gram(refb[:1], refb[:1], cfg).expand(T, R, R)
        Kxy = _pde_gram(ap, refb, cfg)
    else:
        S = _feats(ap, cfg)
        Sr = ref_feats if ref_feats is not None else sig_features(ref, cfg)
        if cfg.kernel == "lin":
            Kxx = S @ S.transpose(-2, -1)
            Kyy = (Sr @ Sr.T).unsqueeze(0)
            Kxy = S @ Sr.T
        else:
            g = cfg.rbf_gamma
            assert not isinstance(g, str), "sigmmd needs a fixed gamma"
            Kxx = torch.exp(-float(g) * torch.cdist(S, S).pow(2))
            Kyy = torch.exp(-float(g) * torch.cdist(Sr, Sr).pow(2)).unsqueeze(0)
            Kxy = torch.exp(-float(g) * torch.cdist(S, Sr.unsqueeze(0).expand(T, *Sr.shape)).pow(2))
    return _mmd2_from_grams(Kxx, Kyy, Kxy, unbiased).cpu().numpy()
