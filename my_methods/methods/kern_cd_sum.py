"""Sum-kernel kern_cd (specs/methods.md, "Sum kernel").

The part is a single action rather than a whole chunk, so the mean runs over both the B
chunks and the H horizon steps:

    mu_A = 1/(BH) sum_b sum_h phi(a_bh),      phi: R^3 -> R^128 (RFF).

This embeds the *set* of actions the policy is proposing at this step. It sees the spatial
spread of the chunk batch, which displacement cannot, but it is blind to ordering within a
chunk: permuting the horizon leaves mu_A unchanged. Against `kern_cd_flat` (same actions,
strict time alignment) it isolates how much the ordering is worth.
"""

import torch

from my_methods.kern_cd_core import KernCDMethod


class KernCDSum(KernCDMethod):
    name = "kern_cd_sum"

    def _parts(self, chunks: torch.Tensor) -> torch.Tensor:
        # One part per (chunk, horizon step): P = B*H, so _embed's mean over the part
        # axis is the 1/(BH) double sum.
        N = chunks.shape[0]
        return chunks.reshape(N, -1, chunks.shape[-1])  # [N, B*H, 3]
