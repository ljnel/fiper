"""Worked example and reference for the method contract.

Run it with::

    python -m my_methods.run example --task push_chair

======================================================================
WHAT YOU DECLARE
======================================================================

``tensors``     which tensors to load. Everywhere available: "obs_embeddings",
                "action_preds".
``actions``     which action channels to keep in ``action_preds``. Declaring
                ["position"] normalises every task to 3 channels -- worth doing,
                because `sorting` stores 6 channels and maps position to [3,4,5]
                while every other task maps it to [0,1,2].
``normalize``   per-tensor overrides merged over configs/eval/base.yaml.
``params``      hyperparameters. These become ``hparams.model``, which is how the
                results store tells two configurations of a method apart -- so
                anything that changes the numbers must live here, or two variants
                will silently collapse onto the same entry.

======================================================================
WHAT YOU IMPLEMENT
======================================================================

fit(calib)      once, before scoring.

                calib["obs_embeddings"]  (N_cal, D)          all calibration steps
                calib["action_preds"]    (N_cal, B, H, A)    B chunks, H horizon
                calib["episode_lengths"] (n_eps,)            sums to N_cal
                calib["successful"]      (n_eps,) bool       per-episode outcome

                Use episode_lengths to recover per-rollout structure:
                    np.split(X, np.cumsum(lengths)[:-1])

                NOTE: the calibration split is NOT all-nominal. On push_t only 21 of
                50 calibration rollouts succeeded. FIPER's own methods fit on all of
                them; if your method estimates a *nominal* support region, filter on
                calib["successful"] yourself.

score(step)     per rollout step, returns a float. Higher = more anomalous.

                step["obs_embeddings"]   (D,)
                step["action_preds"]     (B, H, A)

TEMPORAL SMOOTHING IS ALREADY HANDLED. Do not aggregate over time yourself:
FIPER sums your raw scores over a trailing window and sweeps all 17 window sizes
[1,2,3,4,5,7,9,11,13,15,20,25,30,35,40,45,50] automatically, calibrating thresholds
on the same windowed scores. Write a single-step score and you get all of it free.

D, B and H differ per task (D is 64/640/128/512/96) -- keep methods dimension-agnostic.

======================================================================
OPTIONAL
======================================================================

score_rollout(rollout)  implement INSTEAD of score() to score a whole rollout at
                        once; tensors gain a leading step axis, (T, D) and
                        (T, B, H, A), and you return T floats. Use it when per-step
                        calls waste work -- for kernel methods that share a
                        factorisation across queries this is worth ~30x.

self.subset             "calibration" or "test", set before each pass, if the two
                        must be scored differently (e.g. leave-one-out on your own
                        fit set).

history = n             gives score() the last n steps as a leading axis, oldest
                        first, zero-padded at episode start. Almost never needed --
                        only for scores defined *across* steps that cannot be
                        decomposed into per-step values (FIPER's STAC is the one
                        built-in case: an MMD between consecutive chunk predictions).

deterministic = True    declare it when the method uses no randomness, and --full runs
                        one seed instead of five. Repeated seeds of a deterministic
                        method give identical numbers (std 0), so the extra passes are
                        wasted. Leave it False if anything samples -- FIPER's rnd_oe and
                        logpzo train networks and show real seed-to-seed spread (std up
                        to 0.42), which one seed would hide.

self.p                  params as attributes: self.p.alpha
self.dataset            the ProcessedRolloutDataset, for anything not covered above
"""

import numpy as np

from my_methods.base import Method


class Example(Method):
    """Mahalanobis distance of the observation embedding from the calibration mean.

    Deliberately trivial -- it exists to show the contract end to end, not to be a
    good detector.
    """

    name = "example"

    tensors = ["obs_embeddings"]
    normalize = {"obs_embeddings": False}
    deterministic = True

    params = dict(shrinkage=1e-3)

    def fit(self, calib):
        X = np.asarray(calib["obs_embeddings"], dtype=np.float64)

        self.mean = X.mean(axis=0)
        cov = np.cov(X - self.mean, rowvar=False)
        # Shrink toward the identity: D can exceed N_cal (stacking has D=640 against
        # 2760 calibration steps, push_chair D=96 against 49), so the raw covariance
        # is often singular.
        cov += self.p.shrinkage * np.trace(cov) / len(cov) * np.eye(len(cov))
        self.precision = np.linalg.pinv(cov)

    def score(self, step):
        delta = np.asarray(step["obs_embeddings"], dtype=np.float64) - self.mean
        return float(delta @ self.precision @ delta)
