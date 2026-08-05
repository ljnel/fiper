"""Joint observation + action-chunk KernCD (docs/kern_cd_action_channel.md).

Extends `kern_cd`'s observation-embedding support estimate with a per-step
action-chunk *mean embedding* ``mu_t`` and fits a single RBF KernCD on the
standardised, dimension-balanced, block-weighted concatenation

    z_t = [ sqrt(g_o) * obs_std_t / sqrt(d_obs) ,
            sqrt(g_a) * mu_std_t  / sqrt(d_act) ] .

The per-block ``1/sqrt(dim)`` factor makes the two blocks contribute equally to
the RBF distance regardless of width, so ``block_ratio = g_a/g_o = 1`` is exact
equal contribution and is the frozen, data-independent default (option A of
docs/kern_cd_action_channel.md - a model hyperparameter chosen a priori, never
selected on labels, matching how FIPER treats every other model hyperparameter).

Because an RBF on a concatenation is the product of RBFs on the two blocks, this
is the *joint* kernel ``RBF(obs) x RBF(action)`` - the only form that can flag an
"ordinary observation, ordinary action, impossible combination" step, which no
score-level fusion of the two channels can (joint_vs_marginal_kernels.md).

The action feature comes in two "rungs" (cfg.action_feature):

* ``displacement`` (rung 1): mean over the B sampled chunks of the chunk
  end-minus-start position displacement (3 dims).  The control that isolates
  "does *any* action channel help" from "does the signature help".
* ``signature`` (rung 2): mean over the B chunks of the level-``sig_level``
  truncated path signature of the time-augmented, path-dilated position chunk
  (20 dims at level 2).

Everything else - the exact Cholesky path, the leave-one-out (LOO) calibration
closed form, and the whole threshold/window machinery - is inherited unchanged
from :class:`KERNCDEval`.  The LOO closed form ``1/B_ii - lam*m`` is kernel- and
feature-agnostic (it never references ``k(x,x)``), so it applies verbatim to the
joint feature.
"""
import os
import sys

import numpy as np
import torch

from .kerncd_eval import KERNCDEval
from cd.algs.kern_cd import KernCD
from cd.algs.kernels import RBF

# Reuse the validated truncated-signature implementation from the signature
# study (see docs/kern_cd_action_channel.md).  Kept as the single source of
# truth rather than vendored, so this method scores exactly what the study
# measured.
_SIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "my_experiments", "sig_study")
)
if _SIG_DIR not in sys.path:
    sys.path.insert(0, _SIG_DIR)
from sigtools import augment, signature  # noqa: E402


class KERNCDJOINTEval(KERNCDEval):
    _SIG_BATCH = 65536  # cap on paths per signature() call, bounds peak memory

    def _dev(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _execute_preprocessing(self):
        self.embedding_dim = self.dataset.get_tensor_shape("obs_embeddings")[-1]
        self.action_feature = self.cfg.get("action_feature", "signature")
        self.sig_level = int(self.cfg.get("sig_level", 2))
        self.sig_time_aug = bool(self.cfg.get("sig_time_aug", True))
        # Both signature variants need the path dilation theta.
        self._uses_signature = self.action_feature in ("signature", "signature_rbf")
        # RBF-on-signature-features settings (action_feature == "signature_rbf"):
        # number of random Fourier features and the bandwidth (median heuristic
        # on calibration chunk signatures, or an explicit float).
        self.sig_rff_dim = int(self.cfg.get("sig_rff_dim", 128))
        self.sig_rff_gamma = self.cfg.get("sig_rff_gamma", "median")
        # Block weighting g_a / g_o (g_o fixed to 1); applied AFTER per-block
        # standardisation, so it is the genuine action/observation trade-off.
        self.block_ratio = float(self.cfg.get("block_ratio", 1.0))
        self._std_eps = 1e-8

        sub = self.dataset.get_subset(
            subset="calibration",
            required_tensors=["obs_embeddings", "action_preds"],
            required_actions=self.required_actions,   # -> position channels
            optional_actions=self.optional_actions,   # config forces this to []
            normalize_tensors=self.normalize_tensors,
        )
        obs = self._prep_obs(sub["obs_embeddings"])          # [N, dim] float64
        chunks = self._prep_chunks(sub["action_preds"])      # torch [N, B, H, 3]

        # Path dilation theta (signature rung only): scale the position channels
        # so the median chunk total variation is 1, otherwise the higher
        # signature levels are numerically swamped (docs, section 1).
        self._theta = self._fit_theta(chunks) if self._uses_signature else 1.0
        # Fit the RBF-on-signature-features map (bandwidth + Fourier weights) on
        # calibration BEFORE building any mean embedding, so fit and scoring share
        # the same random features.
        if self.action_feature == "signature_rbf":
            self._setup_rff(chunks)
        mu = self._action_features(chunks)                   # [N, F] float64

        # Per-block standardisers, fit on calibration only.
        self._obs_mean, self._obs_std = obs.mean(0), obs.std(0) + self._std_eps
        self._mu_mean, self._mu_std = mu.mean(0), mu.std(0) + self._std_eps

        Z = self._assemble_Z(obs, mu)
        self.model = KernCD(
            kernel=RBF(gamma=self.cfg.get("gamma", "median")),
            lam=self.cfg.get("lam", 1e-5),
            rank=self.cfg.get("rank", None),   # None => exact O(m^3) path
            pivot=self.cfg.get("pivot", "rp"),
        ).fit(Z)

        # LOO calibration scores (inherited, kernel/feature-agnostic).
        self._loo_scores = self._compute_loo_scores()
        if len(Z) <= int(self.cfg.get("loo_validation_max_m", 300)):
            self._validate_loo(Z)

        # State for the per-step passes (see inherited calculate_uncertainty_score).
        self._subset = None
        self._loo_idx = 0
        self._test_scores = None
        self._score_idx = 0

    # ---- feature construction ------------------------------------------------
    def _prep_obs(self, obs) -> np.ndarray:
        """Match kern_cd's obs handling: drop any history dim, keep the last
        `embedding_dim` features."""
        X = np.asarray(obs, dtype=np.float64)
        if X.ndim > 2:            # history_length > 1
            X = X[:, 0]
        return X[:, -self.embedding_dim:]

    def _prep_chunks(self, ap) -> torch.Tensor:
        """action_preds -> torch [N, B, H, A_pos].  All FIPER benchmark tasks are
        single-robot; multi-robot joint scoring is not supported."""
        t = torch.as_tensor(np.asarray(ap), dtype=torch.float32, device=self._dev())
        if t.ndim == 5:                       # [N, robots, B, H, A]
            if t.shape[1] != 1:
                raise NotImplementedError(
                    "kern_cd joint action channel does not support multi-robot rollouts."
                )
            t = t[:, 0]
        assert t.ndim == 4, f"unexpected action_preds ndim {t.ndim} (want [N,B,H,A])"
        return t

    def _fit_theta(self, chunks: torch.Tensor) -> float:
        x = chunks.reshape(-1, chunks.shape[-2], chunks.shape[-1])
        tv = (x[:, 1:] - x[:, :-1]).norm(dim=-1).sum(-1)     # total variation / chunk
        med = float(tv.median())
        return 1.0 / med if med > 0 else 1.0

    def _sig_features(self, paths: torch.Tensor) -> torch.Tensor:
        """Level-`sig_level` truncated signature of each (time-augmented,
        dilated) path, computed in memory-bounded batches."""
        out = []
        for s in range(0, paths.shape[0], self._SIG_BATCH):
            x = augment(paths[s:s + self._SIG_BATCH], time=self.sig_time_aug, scale=self._theta)
            out.append(signature(x, self.sig_level, flatten=True))
        return torch.cat(out, dim=0) if len(out) > 1 else out[0]

    def _rff_seed(self) -> int:
        """Per-run seed for the random Fourier features, so the draw varies across
        evaluation seeds (averaging over the RFF randomness) yet is identical
        between fit and scoring within a run."""
        try:
            return int(self.cfg.hparams.model.get("seed", 0) or 0)
        except Exception:
            return 0

    def _setup_rff(self, chunks: torch.Tensor) -> None:
        """Fit the RBF-on-signature-features map used by action_feature
        ``signature_rbf``: a bandwidth ``gamma`` (median heuristic on a subsample
        of calibration chunk signatures) and the random Fourier feature weights
        (W, b).  The per-step action feature then becomes ``mean_b psi(phi(chunk_b))``
        with ``psi`` the RFF map of ``k(x,y)=exp(-gamma||x-y||^2)`` - a finite
        approximation of the RBF kernel *mean embedding*.  Unlike the linear
        embedding (plain mean of signatures), the RBF embedding is characteristic,
        so the action channel sees the whole chunk distribution, not just its mean."""
        N, B = chunks.shape[0], chunks.shape[1]
        paths = chunks.reshape(N * B, chunks.shape[-2], chunks.shape[-1])
        rng = np.random.default_rng(self._rff_seed())
        n_paths = paths.shape[0]
        s = min(n_paths, 4096)                                # subsample for median
        idx = (rng.choice(n_paths, size=s, replace=False) if n_paths > s
               else np.arange(n_paths))
        phi = self._sig_features(paths[torch.as_tensor(idx, device=paths.device)])  # [s, F]
        if self.sig_rff_gamma == "median":
            d2 = torch.cdist(phi, phi).pow(2)
            med = torch.median(d2[d2 > 0])
            gamma = float(1.0 / med) if float(med) > 0 else 1.0
        else:
            gamma = float(self.sig_rff_gamma)
        self._rff_gamma = gamma
        F, D = phi.shape[1], self.sig_rff_dim
        # RFF for k(x,y)=exp(-gamma||x-y||^2): W ~ N(0, 2*gamma), b ~ U(0, 2*pi).
        W = rng.standard_normal((F, D)) * np.sqrt(2.0 * gamma)
        b = rng.uniform(0.0, 2.0 * np.pi, size=D)
        self._rff_W = torch.as_tensor(W, dtype=phi.dtype, device=paths.device)
        self._rff_b = torch.as_tensor(b, dtype=phi.dtype, device=paths.device)

    def _rff(self, phi: torch.Tensor) -> torch.Tensor:
        """Random Fourier feature map: sqrt(2/D) cos(phi W + b)."""
        return np.sqrt(2.0 / self.sig_rff_dim) * torch.cos(phi @ self._rff_W + self._rff_b)

    def _action_features(self, chunks: torch.Tensor) -> np.ndarray:
        """[N, B, H, 3] chunks -> [N, F] per-step mean action-chunk embedding."""
        N, B = chunks.shape[0], chunks.shape[1]
        if self.action_feature == "displacement":
            mu = (chunks[:, :, -1] - chunks[:, :, 0]).mean(dim=1)          # [N, 3]
        elif self.action_feature == "signature":
            paths = chunks.reshape(N * B, chunks.shape[-2], chunks.shape[-1])
            feats = self._sig_features(paths)                             # [N*B, F]
            mu = feats.reshape(N, B, -1).mean(dim=1)                      # [N, F]
        elif self.action_feature == "signature_rbf":
            paths = chunks.reshape(N * B, chunks.shape[-2], chunks.shape[-1])
            psi = self._rff(self._sig_features(paths))                    # [N*B, D]
            mu = psi.reshape(N, B, -1).mean(dim=1)                        # [N, D] RBF mean emb.
        else:
            raise ValueError(
                f"Unknown action_feature '{self.action_feature}' (expected "
                "'displacement', 'signature' or 'signature_rbf')."
            )
        return mu.detach().cpu().numpy().astype(np.float64)

    def _assemble_Z(self, obs: np.ndarray, mu: np.ndarray) -> np.ndarray:
        obs_std = (obs - self._obs_mean) / self._obs_std
        mu_std = (mu - self._mu_mean) / self._mu_std
        # Per-block dimension balancing (docs/kern_cd_action_channel.md, option A).
        # After standardisation each block's expected squared norm grows with its
        # width, so the ~128-D observation block would swamp the 3/20-D action
        # block in the RBF distance (obs contributes d_obs, action only d_action).
        # Dividing each block by sqrt(dim) equalises the two blocks' contribution
        # regardless of width, so block_ratio=1 is EXACT equal contribution - a
        # data-independent default (dims are known before any label is seen),
        # frozen like every other FIPER model hyperparameter.  block_ratio then
        # scales the action block's contribution relative to the balanced obs
        # block (used only for the sensitivity report, never selected on data).
        obs_bal = obs_std / np.sqrt(obs_std.shape[1])
        mu_bal = mu_std / np.sqrt(mu_std.shape[1])
        g_o, g_a = 1.0, self.block_ratio
        return np.concatenate([np.sqrt(g_o) * obs_bal, np.sqrt(g_a) * mu_bal], axis=1)

    # ---- test scoring (overrides the obs-only batched precompute) ------------
    def _precompute_scores_batched(self, subset: str) -> np.ndarray:
        scores = []
        for rollout in self.dataset.iterate_episodes(
            subset=subset,
            required_tensors=self.required_tensors,
            optional_tensors=self.optional_tensors,
            required_actions=self.required_actions,
            optional_actions=self.optional_actions,
            with_success_labels=True,
            normalize_tensors=self.normalize_tensors,
            history=self.cfg.history_length,
        ):
            obs = self._prep_obs(rollout["obs_embeddings"])
            mu = self._action_features(self._prep_chunks(rollout["action_preds"]))
            Z = self._assemble_Z(obs, mu)
            scores.append(np.asarray(self.model.score(Z)).reshape(-1))
        return np.concatenate(scores) if scores else np.zeros(0)


# Method-name resolution (EvaluationManager._get_method_eval_class) strips
# underscores and upper-cases: kern_cd_disp -> KERNCDDISPEval, kern_cd_sig ->
# KERNCDSIGEval.  Both rungs share one implementation; the rung is chosen from
# config (`action_feature`).
KERNCDDISPEval = KERNCDJOINTEval
KERNCDSIGEval = KERNCDJOINTEval
KERNCDSIGRBFEval = KERNCDJOINTEval
