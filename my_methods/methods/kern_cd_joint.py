"""Joint observation + action-chunk KernCD (docs/kern_cd_action_channel.md).

Ports evaluation/method_eval_classes/kerncd_joint_eval.py onto the my_methods contract.

Extends kern_cd's observation-embedding support estimate with a per-step action-chunk
*mean embedding* ``mu_t`` and fits a single RBF KernCD on the standardised,
dimension-balanced, block-weighted concatenation

    z_t = [ sqrt(g_o) * obs_std_t / sqrt(d_obs) ,
            sqrt(g_a) * mu_std_t  / sqrt(d_act) ] .

Because an RBF on a concatenation is the product of RBFs on the two blocks, this is the
*joint* kernel ``RBF(obs) x RBF(action)`` -- the only form that can flag an "ordinary
observation, ordinary action, impossible combination" step, which no score-level fusion
of the two channels can (docs/joint_vs_marginal_kernels.md).

Registers three action features (``action_feature``):

    kern_cd_rbf_disp   displacement    mean chunk end-minus-start position (3 dims).
                                       Rung 1: the control isolating "does *any* action
                                       channel help" from "does the signature help".
    kern_cd_rbf_sig    signature       mean level-``sig_level`` truncated signature of
                                       the time-augmented, path-dilated chunk (20 dims
                                       at level 2). Rung 2.
    kern_cd_rbf_sigk   signature_rbf   RBF kernel *mean embedding* of the B chunk
                                       signatures via random Fourier features. Only the
                                       pooling over the B chunks differs from
                                       kern_cd_rbf_sig: averaging a characteristic
                                       feature map keeps the whole chunk distribution,
                                       where averaging the signatures themselves keeps
                                       only its mean.

Everything else -- the exact Cholesky path, the LOO calibration closed form, and the
whole threshold/window machinery -- is inherited unchanged.
"""

import os
import sys

import numpy as np
import torch

from my_methods.methods.kern_cd import _KernCDBase

# Reuse the validated truncated-signature implementation from the signature study
# (docs/kern_cd_action_channel.md). Kept as the single source of truth rather than
# vendored, so this method scores exactly what the study measured. my_experiments is
# not a package, hence the path injection.
_SIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "my_experiments", "sig_study"))
if _SIG_DIR not in sys.path:
    sys.path.insert(0, _SIG_DIR)
from sigtools import augment, signature  # noqa: E402


class _KernCDJoint(_KernCDBase):
    """Shared joint-feature machinery. Not registered -- it sets no ``name``."""

    tensors = ["obs_embeddings", "action_preds"]
    # Position channels only. Override the base optional_actions ([rotation]) so the
    # action feature is not silently widened.
    actions = ["position"]
    optional_actions = []
    normalize = {"obs_embeddings": False, "action_preds": False}

    #: Cap on paths per signature() call, bounds peak memory.
    _SIG_BATCH = 65536
    _STD_EPS = 1e-8

    def _prep_obs(self, obs) -> np.ndarray:
        # The joint feature is standardised and block-balanced before the kernel sees it,
        # so it is built in float64 (unlike the obs-only path, which fits on the raw
        # float32 tensors).
        return super()._prep_obs(obs).astype(np.float64)

    def _fit_features(self, calib) -> np.ndarray:
        obs = self._prep_obs(calib["obs_embeddings"])  # [N, d_obs] float64
        chunks = self._prep_chunks(calib["action_preds"])  # torch [N, B, H, 3]

        # Path dilation theta (signature rungs only): scale the position channels so
        # the median chunk total variation is 1, otherwise the higher signature levels
        # are numerically swamped (docs, section 1).
        self._uses_signature = self.p.action_feature in ("signature", "signature_rbf")
        self._theta = self._fit_theta(chunks) if self._uses_signature else 1.0
        # Fit the RFF map (bandwidth + Fourier weights) on calibration BEFORE building
        # any mean embedding, so fit and scoring share the same random features.
        if self.p.action_feature == "signature_rbf":
            self._setup_rff(chunks)
        mu = self._action_features(chunks)  # [N, F] float64

        # Per-block standardisers, fit on calibration only.
        self._obs_mean, self._obs_std = obs.mean(0), obs.std(0) + self._STD_EPS
        self._mu_mean, self._mu_std = mu.mean(0), mu.std(0) + self._STD_EPS

        return self._assemble_Z(obs, mu)

    def _features(self, rollout) -> np.ndarray:
        obs = self._prep_obs(rollout["obs_embeddings"])
        mu = self._action_features(self._prep_chunks(rollout["action_preds"]))
        return self._assemble_Z(obs, mu)

    # -- action-chunk features -------------------------------------------------
    def _prep_chunks(self, ap) -> torch.Tensor:
        """action_preds -> torch [N, B, H, A_pos]. All FIPER benchmark tasks are
        single-robot; multi-robot joint scoring is not supported."""
        t = torch.as_tensor(np.asarray(ap), dtype=torch.float32, device=self.device)
        if t.ndim == 5:  # [N, robots, B, H, A]
            if t.shape[1] != 1:
                raise NotImplementedError("kern_cd joint action channel does not support multi-robot rollouts.")
            t = t[:, 0]
        assert t.ndim == 4, f"unexpected action_preds ndim {t.ndim} (want [N,B,H,A])"
        return t

    def _fit_theta(self, chunks: torch.Tensor) -> float:
        x = chunks.reshape(-1, chunks.shape[-2], chunks.shape[-1])
        tv = (x[:, 1:] - x[:, :-1]).norm(dim=-1).sum(-1)  # total variation / chunk
        med = float(tv.median())
        return 1.0 / med if med > 0 else 1.0

    def _sig_features(self, paths: torch.Tensor) -> torch.Tensor:
        """Level-`sig_level` truncated signature of each (time-augmented, dilated)
        path, computed in memory-bounded batches."""
        out = []
        for s in range(0, paths.shape[0], self._SIG_BATCH):
            x = augment(paths[s : s + self._SIG_BATCH], time=self.p.sig_time_aug, scale=self._theta)
            out.append(signature(x, self.p.sig_level, flatten=True))
        return torch.cat(out, dim=0) if len(out) > 1 else out[0]

    def _setup_rff(self, chunks: torch.Tensor) -> None:
        """Fit the RBF-on-signature-features map: a bandwidth ``gamma`` (median
        heuristic on a subsample of calibration chunk signatures) and the random
        Fourier feature weights (W, b)."""
        N, B = chunks.shape[0], chunks.shape[1]
        paths = chunks.reshape(N * B, chunks.shape[-2], chunks.shape[-1])
        # Seeded per run, so the draw varies across evaluation seeds (averaging over the
        # RFF randomness) yet is identical between fit and scoring within a run.
        rng = np.random.default_rng(int(getattr(self.p, "seed", 0) or 0))
        n_paths = paths.shape[0]
        s = min(n_paths, 4096)  # subsample for the median heuristic
        idx = rng.choice(n_paths, size=s, replace=False) if n_paths > s else np.arange(n_paths)
        phi = self._sig_features(paths[torch.as_tensor(idx, device=paths.device)])  # [s, F]

        if self.p.sig_rff_gamma == "median":
            d2 = torch.cdist(phi, phi).pow(2)
            med = torch.median(d2[d2 > 0])
            gamma = float(1.0 / med) if float(med) > 0 else 1.0
        else:
            gamma = float(self.p.sig_rff_gamma)
        self._rff_gamma = gamma

        F, D = phi.shape[1], self.p.sig_rff_dim
        # RFF for k(x,y)=exp(-gamma||x-y||^2): W ~ N(0, 2*gamma), b ~ U(0, 2*pi).
        W = rng.standard_normal((F, D)) * np.sqrt(2.0 * gamma)
        b = rng.uniform(0.0, 2.0 * np.pi, size=D)
        self._rff_W = torch.as_tensor(W, dtype=phi.dtype, device=paths.device)
        self._rff_b = torch.as_tensor(b, dtype=phi.dtype, device=paths.device)

    def _rff(self, phi: torch.Tensor) -> torch.Tensor:
        """Random Fourier feature map: sqrt(2/D) cos(phi W + b)."""
        return np.sqrt(2.0 / self.p.sig_rff_dim) * torch.cos(phi @ self._rff_W + self._rff_b)

    def _action_features(self, chunks: torch.Tensor) -> np.ndarray:
        """[N, B, H, 3] chunks -> [N, F] per-step mean action-chunk embedding."""
        N, B = chunks.shape[0], chunks.shape[1]
        feature = self.p.action_feature
        if feature == "displacement":
            mu = (chunks[:, :, -1] - chunks[:, :, 0]).mean(dim=1)  # [N, 3]
        elif feature == "signature":
            paths = chunks.reshape(N * B, chunks.shape[-2], chunks.shape[-1])
            mu = self._sig_features(paths).reshape(N, B, -1).mean(dim=1)  # [N, F]
        elif feature == "signature_rbf":
            paths = chunks.reshape(N * B, chunks.shape[-2], chunks.shape[-1])
            psi = self._rff(self._sig_features(paths))  # [N*B, D]
            mu = psi.reshape(N, B, -1).mean(dim=1)  # [N, D] RBF mean embedding
        else:
            raise ValueError(
                f"Unknown action_feature '{feature}' (expected 'displacement', "
                "'signature' or 'signature_rbf')."
            )
        return mu.detach().cpu().numpy().astype(np.float64)

    def _assemble_Z(self, obs: np.ndarray, mu: np.ndarray) -> np.ndarray:
        obs_std = (obs - self._obs_mean) / self._obs_std
        mu_std = (mu - self._mu_mean) / self._mu_std
        # Per-block dimension balancing (docs/kern_cd_action_channel.md, option A).
        # After standardisation each block's expected squared norm grows with its width,
        # so the ~128-D observation block would swamp the 3/20-D action block in the RBF
        # distance. Dividing each block by sqrt(dim) equalises the two contributions
        # regardless of width, so block_ratio=1 is EXACT equal contribution -- a
        # data-independent default (dims are known before any label is seen), frozen like
        # every other FIPER model hyperparameter.
        obs_bal = obs_std / np.sqrt(obs_std.shape[1])
        mu_bal = mu_std / np.sqrt(mu_std.shape[1])
        g_o, g_a = 1.0, float(self.p.block_ratio)
        return np.concatenate([np.sqrt(g_o) * obs_bal, np.sqrt(g_a) * mu_bal], axis=1)


# The KernCD support estimator is RBF on the standardised joint feature in all three
# variants; only the action channel differs.
_ESTIMATOR = dict(kernel="rbf", gamma="median", lam=1.0e-5, rank=None, pivot="rp")
# g_a / g_o, applied after per-block standardisation + 1/sqrt(dim) balancing. 1.0 = exact
# equal contribution of the obs and action blocks (frozen default, chosen a priori; not
# selected on data -- see docs/kern_cd_action_channel.md, A).
_BLOCK_RATIO = dict(block_ratio=1.0)


class KernCDDisp(_KernCDJoint):
    name = "kern_cd_rbf_disp"

    params = dict(**_ESTIMATOR, **_BLOCK_RATIO, action_feature="displacement")


class KernCDSig(_KernCDJoint):
    name = "kern_cd_rbf_sig"

    params = dict(
        **_ESTIMATOR,
        **_BLOCK_RATIO,
        action_feature="signature",
        sig_level=2,
        sig_time_aug=True,
    )


class KernCDSigK(_KernCDJoint):
    name = "kern_cd_rbf_sigk"

    # The random Fourier draw depends on the run seed, so this variant genuinely varies
    # across seeds (the reference run shows TPR_std 0.004). Every seed must be run.
    deterministic = False

    params = dict(
        **_ESTIMATOR,
        **_BLOCK_RATIO,
        action_feature="signature_rbf",
        sig_level=2,
        sig_time_aug=True,
        sig_rff_dim=128,  # random Fourier features approximating the RBF mean embedding
        sig_rff_gamma="median",  # bandwidth on signature features, fit on calibration
        seed=0,  # replaced with the run seed by EvaluationManager.evaluate
    )
