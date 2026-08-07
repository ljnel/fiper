"""Observation-only kern_cd (specs/methods.md, "Observation-only").

The baseline of the family: no action channel at all, so the kernel is the observation
factor alone,

    k((o,A), (o',A')) = exp(-||o - o'||^2 / 2 sigma_o^2),

with sigma_o the median pairwise distance between calibration observation embeddings.
Every action-channel method multiplies exactly this factor by a second one, so any gap
against this row is attributable to the action channel and nothing else.
"""

from my_methods.kern_cd_core import KernCDMethod


class KernCDObs(KernCDMethod):
    name = "kern_cd_obs"

    # Nothing is drawn at random: no RFF map, and the only subsample (the median
    # heuristic) runs on at most 3000 of at most 2760 calibration steps, i.e. on all of
    # them. Repeated seeds would reproduce identical numbers.
    deterministic = True

    tensors = ["obs_embeddings"]
    uses_actions = False

    params = dict(lam=1.0e-5)
