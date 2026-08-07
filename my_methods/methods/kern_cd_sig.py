"""Signature-kernel kern_cd (specs/methods.md, "Signature kernel").

The part is a whole chunk, represented by the **level-2 term** of its truncated path
signature -- levels 0 and 1 are discarded, so what survives is exactly the pairwise
iterated integrals:

    mu_A = 1/B sum_b phi(S^2(a_b)),      phi: R^16 -> R^128 (RFF).

Level 0 is the constant 1 and carries nothing. Level 1 is the increment ``a_bH - a_b1``,
i.e. the displacement, which `kern_cd_disp` already isolates; keeping it here would mean
this method could not lose to that one for reasons other than noise, and the point is to
measure what level 2 adds on its own. Level 2 is where path *shape* lives: its
antisymmetric part is the signed area the chunk sweeps out, which separates trajectories
that share endpoints but curve differently.

Time augmentation
-----------------
A signature is invariant to reparameterisation, so without a time channel a chunk that
dawdles and one that moves steadily embed identically -- and a chunk that retraces its own
path cancels to zero (tree-like equivalence). Appending a monotone channel breaks both.
Its scale relative to the position channels is a real choice: the channel spans 1 by
construction, so the positions are dilated by ``theta = 1/median chunk total variation``
(fit on the calibration set) to put a typical chunk on the same footing. Any *overall*
scale would wash out -- level 2 is quadratic in the path, so a global factor is a constant
that the median heuristic inside phi absorbs -- but the space-to-time ratio does not.
"""

import torch

from my_methods.kern_cd_core import ChunkPartsMixin, KernCDMethod


class KernCDSig(ChunkPartsMixin, KernCDMethod):
    name = "kern_cd_sig"

    def _fit_parts(self, chunks: torch.Tensor) -> torch.Tensor:
        # Path dilation, fit on calibration only and then frozen for scoring.
        tv = (chunks[:, :, 1:] - chunks[:, :, :-1]).norm(dim=-1).sum(-1)  # [N, B]
        med = float(tv.median())
        self._theta = 1.0 / med if med > 0 else 1.0
        return self._parts(chunks)

    def _chunk_feature(self, chunks: torch.Tensor) -> torch.Tensor:
        """``[N, B, H, 3]`` -> ``[N, B, 16]``, the level-2 signature of each chunk."""
        N, B, H, _ = chunks.shape
        paths = chunks.reshape(N * B, H, -1) * self._theta
        time = torch.linspace(0.0, 1.0, H, dtype=paths.dtype, device=paths.device)
        paths = torch.cat([paths, time.expand(N * B, H).unsqueeze(-1)], dim=-1)  # [NB, H, 4]

        # For a piecewise-linear path with increments d_s, the level-2 signature is
        #     S^2 = sum_{s<u} d_s (x) d_u  +  1/2 sum_s d_s (x) d_s,
        # where the first term telescopes: sum_{s<u} d_s is the path position just before
        # increment u, relative to the start. Both terms are one batched einsum.
        inc = paths[:, 1:] - paths[:, :-1]  # [NB, H-1, 4]
        pre = paths[:, :-1] - paths[:, :1]  # [NB, H-1, 4], partial sums of the increments
        S2 = torch.einsum("nsi,nsj->nij", pre, inc) + 0.5 * torch.einsum("nsi,nsj->nij", inc, inc)
        return S2.reshape(N, B, -1)  # [N, B, 16]
