"""Displacement kern_cd (specs/methods.md, "Displacement").

The simplest action channel: a chunk is summarised by where it ends up relative to where
it started, discarding the route taken.

    mu_A = 1/B sum_b phi(a_bH - a_b1),      phi: R^3 -> R^128 (RFF).

It is the control for the other three action methods -- it carries direction and extent of
the commanded motion but no shape -- so a method that only matches this one has not
gained anything from its richer chunk representation.
"""

import torch

from my_methods.kern_cd_core import ChunkPartsMixin, KernCDMethod


class KernCDDisp(ChunkPartsMixin, KernCDMethod):
    name = "kern_cd_disp"

    def _chunk_feature(self, chunks: torch.Tensor) -> torch.Tensor:
        return chunks[:, :, -1] - chunks[:, :, 0]  # [N, B, 3]
