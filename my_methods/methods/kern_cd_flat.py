"""Flattened kern_cd (specs/methods.md, "Flattened").

The other simple baseline: keep the whole chunk, but as a raw vector.

    mu_A = 1/B sum_b phi(vec(a_b)),      phi: R^{3H} -> R^128 (RFF).

Everything about the chunk survives, including where in the horizon each action sits --
but only under exact time alignment, since the Euclidean distance behind phi compares
step h to step h and nothing else. That is the complement of the sum kernel, which keeps
the actions and throws the alignment away.
"""

import torch

from my_methods.kern_cd_core import ChunkPartsMixin, KernCDMethod


class KernCDFlat(ChunkPartsMixin, KernCDMethod):
    name = "kern_cd_flat"

    def _chunk_feature(self, chunks: torch.Tensor) -> torch.Tensor:
        N, B = chunks.shape[:2]
        return chunks.reshape(N, B, -1)  # [N, B, H*3]
