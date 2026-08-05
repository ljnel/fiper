"""Path-signature tooling for the FIPER action-chunk study.

Two families are provided:

* `signature()` - truncated tensor-algebra signature features, batched, torch,
  runs on GPU. Explicit level tensors so the memory footprint is visible.
* `sig_kernel_gram()` - the *untruncated* signature kernel of Salvi et al.
  (2021) via the Goursat PDE finite-difference solver, batched on GPU.

Design notes
------------
The signature of a piecewise-linear path is the ordered tensor product of the
exponentials of its increments (Chen's identity):  S = exp(d1) * exp(d2) * ...
Level-k of exp(d) is d^{otimes k}/k!.  Chunks here are short (H = 8 or 16
knots) so the increment loop is trivial; all cost is in the tensor products,
which are pure batched matmuls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch


# --------------------------------------------------------------------------
# dimension bookkeeping
# --------------------------------------------------------------------------
def sig_dim(d: int, level: int) -> int:
    """Length of the flattened truncated signature (levels 1..level)."""
    if d == 1:
        return level
    return int((d ** (level + 1) - d) // (d - 1))


def logsig_dim(d: int, level: int) -> int:
    """Dimension of the truncated log-signature = sum of free-Lie-algebra ranks
    (Witt's formula)."""
    def moebius(n):
        res, p = 1, 2
        m = n
        while p * p <= m:
            if m % p == 0:
                m //= p
                if m % p == 0:
                    return 0
                res = -res
            p += 1
        if m > 1:
            res = -res
        return res

    total = 0
    for k in range(1, level + 1):
        s = sum(moebius(k // e) * d ** e for e in range(1, k + 1) if k % e == 0)
        total += s // k
    return total


# --------------------------------------------------------------------------
# path augmentations
# --------------------------------------------------------------------------
def augment(paths: torch.Tensor, *, time: bool = True, basepoint: bool = False,
            lead_lag: bool = False, scale: float = 1.0) -> torch.Tensor:
    """paths: (..., T, d) -> augmented (..., T', d')

    time      : append a monotone time channel in [0,1] (makes the signature
                injective up to reparameterisation and kills tree-like cancellation)
    basepoint : prepend a copy of the origin so translations of the path matter
    lead_lag  : lead-lag transform, doubles d and gives the signature access to
                quadratic variation
    scale     : multiply the *state* channels (time channel excluded) by this
    """
    x = paths
    if scale != 1.0:
        x = x * scale
    if lead_lag:
        T = x.shape[-2]
        lead = x.repeat_interleave(2, dim=-2)[..., 1:, :]
        lag = x.repeat_interleave(2, dim=-2)[..., :-1, :]
        x = torch.cat([lead, lag], dim=-1)
    if time:
        T = x.shape[-2]
        t = torch.linspace(0.0, 1.0, T, dtype=x.dtype, device=x.device)
        t = t.expand(*x.shape[:-1])
        x = torch.cat([x, t.unsqueeze(-1)], dim=-1)
    if basepoint:
        zero = torch.zeros_like(x[..., :1, :])
        x = torch.cat([zero, x], dim=-2)
    return x


# --------------------------------------------------------------------------
# truncated signature
# --------------------------------------------------------------------------
def signature(paths: torch.Tensor, level: int, *, flatten: bool = True,
              level_scale: float | None = None) -> torch.Tensor:
    """Truncated signature of a batch of piecewise-linear paths.

    paths : (N, T, d)
    level : truncation level M
    level_scale : if given (theta), level k is multiplied by theta**k.  This is
        the standard dilation trick that stops level 1 from dominating (or
        being dominated by) the higher levels in a linear kernel.

    returns (N, sig_dim(d, level)) when flatten, else a list of level tensors
    with shapes (N, d), (N, d, d), ...
    """
    N, T, d = paths.shape
    inc = paths[:, 1:] - paths[:, :-1]                      # (N, T-1, d)

    # levels[k-1] holds level k, shape (N, d^k)
    levels = [torch.zeros(N, d ** k, dtype=paths.dtype, device=paths.device)
              for k in range(1, level + 1)]

    for s in range(inc.shape[1]):
        dx = inc[:, s]                                       # (N, d)
        # exp(dx): E[m] = dx^{otimes m} / m!, shape (N, d^m)
        E = []
        cur = torch.ones(N, 1, dtype=paths.dtype, device=paths.device)
        for m in range(1, level + 1):
            cur = (cur.unsqueeze(-1) * dx.unsqueeze(-2)).reshape(N, -1)
            E.append(cur / math.factorial(m))
        # Chen: new_k = E_k + sum_{j=1}^{k-1} S_j (x) E_{k-j} + S_k
        new = []
        for k in range(1, level + 1):
            acc = E[k - 1] + levels[k - 1]
            for j in range(1, k):
                a = levels[j - 1]                            # (N, d^j)
                b = E[k - j - 1]                             # (N, d^(k-j))
                acc = acc + (a.unsqueeze(-1) * b.unsqueeze(-2)).reshape(N, -1)
            new.append(acc)
        levels = new

    if level_scale is not None:
        levels = [lv * (level_scale ** (k + 1)) for k, lv in enumerate(levels)]
    if flatten:
        return torch.cat(levels, dim=-1)
    return levels


def signature_normalise(sig_levels: list[torch.Tensor]) -> list[torch.Tensor]:
    """Chevyrev-Oberhauser tensor normalisation (robust variant): rescale the
    whole signature by a single dilation lambda so that ||S||=1.  Applied
    level-wise as lambda^k.  Keeps the kernel bounded, which matters for MMD."""
    N = sig_levels[0].shape[0]
    norms = [lv.pow(2).sum(-1) for lv in sig_levels]         # per level
    tot = 1.0 + sum(norms)
    lam = tot.pow(-0.5).unsqueeze(-1)
    return [lv * lam.pow(k + 1) for k, lv in enumerate(sig_levels)]


# --------------------------------------------------------------------------
# untruncated signature kernel (Goursat PDE, Salvi et al. 2021)
# --------------------------------------------------------------------------
def sig_kernel_gram(X: torch.Tensor, Y: torch.Tensor, dyadic_order: int = 0,
                    static_scale: float = 1.0) -> torch.Tensor:
    """Signature kernel Gram matrix K[i,j] = <S(X_i), S(Y_j)> (untruncated),
    for the *linear* static kernel on increments.

    X: (..., n, T, d), Y: (..., m, T, d)  ->  (..., n, m)
    Leading dims (if any) are treated as an independent batch of Gram matrices.

    Solves  du/dsdt = <dX_s, dY_t> u  on the [0,1]^2 grid with the standard
    second-order finite-difference scheme, refined `dyadic_order` times.
    Vectorised over the antidiagonals, so the whole n*m Gram is one solve.
    """
    squeeze = X.ndim == 3
    if squeeze:
        X, Y = X.unsqueeze(0), Y.unsqueeze(0)
    P = X.shape[:-3]
    n, m = X.shape[-3], Y.shape[-3]
    dX = (X[..., 1:, :] - X[..., :-1, :]) * static_scale     # (..., n, T-1, d)
    dY = (Y[..., 1:, :] - Y[..., :-1, :]) * static_scale     # (..., m, T-1, d)

    # inner products of increments, refined by 2**dyadic_order in each axis
    inc = torch.einsum("...isd,...jtd->...ijst", dX, dY)     # (..., n, m, T-1, T-1)
    r = 2 ** dyadic_order
    if r > 1:
        inc = inc.repeat_interleave(r, dim=-2).repeat_interleave(r, dim=-1) / (r * r)

    S, Tg = inc.shape[-2], inc.shape[-1]
    u = torch.ones(*P, n, m, S + 1, Tg + 1, dtype=X.dtype, device=X.device)

    # antidiagonal sweep: u[i,j] depends on (i-1,j), (i,j-1), (i-1,j-1)
    for p in range(2, S + Tg + 1):
        i0 = max(1, p - Tg)
        i1 = min(S, p - 1)
        if i0 > i1:
            continue
        ii = torch.arange(i0, i1 + 1, device=X.device)
        jj = p - ii
        a = u[..., ii - 1, jj]
        b = u[..., ii, jj - 1]
        c = u[..., ii - 1, jj - 1]
        inc_ij = inc[..., ii - 1, jj - 1]
        u[..., ii, jj] = (a + b) * (1.0 + 0.5 * inc_ij + inc_ij * inc_ij / 12.0) \
                         - c * (1.0 - inc_ij * inc_ij / 12.0)
    out = u[..., S, Tg]
    return out.squeeze(0) if squeeze else out


# --------------------------------------------------------------------------
# validation helper
# --------------------------------------------------------------------------
def validate_against_iisignature(seed: int = 0) -> dict:
    """Cross-check `signature` against iisignature on random paths."""
    import iisignature

    out = {}
    rng = np.random.default_rng(seed)
    for d in (2, 3, 4):
        for level in (1, 2, 3, 4):
            p = rng.normal(size=(5, 9, d))
            mine = signature(torch.tensor(p, dtype=torch.float64), level).numpy()
            ref = np.stack([iisignature.sig(p[i], level) for i in range(5)])
            out[(d, level)] = float(np.abs(mine - ref).max())
    return out
