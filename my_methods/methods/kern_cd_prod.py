"""Observation x action-chunk product kernel for KernCD.

Extends ``kern_cd_rbf`` (an RBF support estimator on observation embeddings) with an
action channel, combined by an explicit **product kernel**

    k((o,A), (o',A')) = exp(-gamma_o ||o-o'||^2) * k_act(A, A') .

For the action factor, write a step's chunk batch as ``A`` of shape ``(B, H, D)``: B
sampled chunks, each a length-H path in D=3 position space. With an RFF map
``phi: R^D -> R^d`` approximating ``<phi(u),phi(v)> ~ exp(-gamma_a ||u-v||^2)``,

    mu_A     = 1/(B sqrt(H)) sum_b (+)_h phi(A_bh)   in R^{H x d}
    mu_hat_A = mu_A / ||mu_A||_F
    k_act    = <mu_hat_A, mu_hat_A'>_F               so k_act(A,A) = 1.

The direct sum ``(+)_h`` stacks the per-timestep RFF vectors, so the inner product only
ever pairs ``A_bh`` with ``A'_b'h`` at **equal h**. The implied per-chunk kernel is the
time-aligned sum kernel ``k_traj(x,y) = sum_h exp(-gamma_a ||x_h - y_h||^2)``, and k_act is
the normalised mean-embedding (cosine) kernel over the B chunks.

Two things distinguish this from the earlier joint variants in ``kern_cd_joint.py``:

* Time alignment survives. Displacement and signature features collapse a chunk to a
  single vector, discarding *when* within the horizon the motion happens.
* k_act is a normalised *linear* kernel, not ``exp(-gamma_a MMD^2)``. So the joint kernel
  is NOT an RBF on a concatenation and cannot be built by concatenating blocks -- see
  docs/joint_vs_marginal_kernels.md section 6, which flags exactly this case. Hence the
  explicit :class:`ObsActionProductKernel` below. Normalising k_act to 1 on the diagonal
  also removes the need for the ``block_ratio`` / 1/sqrt(dim) balancing those variants use:
  there is no longer a width mismatch to correct.

Because the observation block is passed through **raw**, exactly as ``kern_cd_rbf`` uses
it, gamma_o is fit on identical data and ``K = K_obs * K_act`` elementwise. Setting
``k_act == 1`` therefore recovers ``kern_cd_rbf`` exactly: this method is a strict
generalisation of the baseline.

Registers:
    kern_cd_prod        the method (time-aligned direct sum over the horizon)
    kern_cd_prod_pool   the control (mean-pooled over the horizon; order-blind)
"""

import numpy as np
import torch
from cd.algs.kernels import RBF
from cd.algs.kernels.base import Kernel
from cd.utils.misc import median_distance

from my_methods.methods.kern_cd import _KernCDBase

# Deliberately does NOT import kern_cd_joint: that module injects my_experiments/sig_study
# onto sys.path at import time to pull in the signature tools, and this method exists to
# shed that machinery. _prep_chunks below is reimplemented rather than reused.


class ObsActionProductKernel(Kernel):
    """``k = RBF(obs) * <mu_hat, mu_hat'>`` over a packed ``[obs | vec(mu_hat)]`` matrix.

    The action block arrives already normalised to unit Frobenius norm, so -- unlike the
    normalised kernels in the ``cd`` package -- no training-diagonal has to be cached
    between the symmetric and cross-kernel calls.
    """

    def __init__(self, d_obs: int, gamma_obs="median", gamma_obs_scale: float = 1.0):
        self.d_obs = int(d_obs)
        self.gamma_obs = gamma_obs
        self.gamma_obs_scale = float(gamma_obs_scale)
        self.obs_kernel = None

    def fit(self, X: np.ndarray) -> "ObsActionProductKernel":
        # Resolve gamma_o with RBF's own heuristic (seeded, subsampled to 3000 points),
        # read it back through the public property, then rebuild the kernel with the
        # scaled float. Going through RBF twice avoids duplicating its median formula.
        # scale=1.0 is bit-identical to RBF(gamma="median") fitted directly.
        obs = np.ascontiguousarray(X[:, : self.d_obs])
        base = RBF(gamma=self.gamma_obs).fit(obs)
        self.obs_kernel = RBF(gamma=base.gamma * self.gamma_obs_scale).fit(obs)
        return self

    def __call__(self, x: np.ndarray, y=None) -> np.ndarray:
        y = x if y is None else y
        k_obs = self.obs_kernel(x[:, : self.d_obs], y[:, : self.d_obs])
        k_act = x[:, self.d_obs :] @ y[:, self.d_obs :].T
        return k_obs * k_act

    def diag(self, x: np.ndarray) -> np.ndarray:
        # k(x,x) = 1 * ||mu_hat||^2, which is 1 by construction. Computed rather than
        # asserted so a degenerate all-zero mu cannot silently produce a wrong score.
        # Overriding matters: the base Kernel.diag would build the full b x b matrix.
        a = x[:, self.d_obs :]
        return np.einsum("ij,ij->i", a, a)


class _KernCDProd(_KernCDBase):
    """Shared product-kernel machinery. Not registered -- it sets no ``name``."""

    tensors = ["obs_embeddings", "action_preds"]
    # Position channels only. Override the base optional_actions ([rotation]), which would
    # otherwise silently widen the action feature to 5 dims on pretzel and 6 on stacking.
    actions = ["position"]
    optional_actions = []
    normalize = {"obs_embeddings": False, "action_preds": False}
    # The RFF draw is seeded from the run seed, so results genuinely vary across seeds.
    deterministic = False

    #: Cap on elements in one RFF intermediate, bounding peak memory. push_t calibration
    #: alone is 1452*256*16 = 5.9M points, so the feature build has to be batched.
    _RFF_ELEM_BUDGET = 32_000_000

    # -- features --------------------------------------------------------------
    def _prep_obs(self, obs) -> np.ndarray:
        # Unlike the obs-only path (which fits on the raw float32 tensors to reproduce the
        # built-in kern_cd), the packed matrix is built in float64: the action block is a
        # 2048-4096 dimensional inner product whose values sit near 1 after normalisation.
        return super()._prep_obs(obs).astype(np.float64)

    def _fit_features(self, calib) -> np.ndarray:
        obs = self._prep_obs(calib["obs_embeddings"])
        chunks = self._prep_chunks(calib["action_preds"])
        # Fit gamma_a and draw the Fourier features on calibration BEFORE building any
        # embedding, so fit and scoring share the same random feature map.
        self._setup_rff(chunks)
        self._d_obs = obs.shape[1]
        return np.concatenate([obs, self._action_features(chunks)], axis=1)

    def _features(self, rollout) -> np.ndarray:
        obs = self._prep_obs(rollout["obs_embeddings"])
        chunks = self._prep_chunks(rollout["action_preds"])
        return np.concatenate([obs, self._action_features(chunks)], axis=1)

    def _make_kernel(self):
        return ObsActionProductKernel(
            self._d_obs,
            gamma_obs=self.p.gamma_obs,
            gamma_obs_scale=self.p.gamma_obs_scale,
        )

    # -- action chunks ---------------------------------------------------------
    def _prep_chunks(self, ap) -> torch.Tensor:
        """action_preds -> torch [N, B, H, D]. All FIPER benchmark tasks are
        single-robot; multi-robot joint scoring is not supported."""
        t = torch.as_tensor(np.asarray(ap), dtype=torch.float32, device=self.device)
        if t.ndim == 5:  # [N, robots, B, H, D]
            if t.shape[1] != 1:
                raise NotImplementedError("kern_cd_prod does not support multi-robot rollouts.")
            t = t[:, 0]
        assert t.ndim == 4, f"unexpected action_preds ndim {t.ndim} (want [N,B,H,D])"
        return t

    def _setup_rff(self, chunks: torch.Tensor) -> None:
        """Fit the spatial RBF bandwidth and draw the random Fourier features.

        The map is ``sqrt(2/d) cos(u W + b)`` with ``W ~ N(0, 2 gamma_a)`` and
        ``b ~ U(0, 2 pi)`` -- identical to sklearn's RBFSampler, but applied in torch so
        the ~1e10 cosine evaluations per seed run on the GPU.
        """
        D = chunks.shape[-1]
        pts = chunks.reshape(-1, D)
        rng = np.random.default_rng(int(getattr(self.p, "seed", 0) or 0))

        # median_distance subsamples to 3000 on the host, so pull a bounded slice off the
        # device first rather than moving all N*B*H points. Resolving through RBF keeps the
        # action bandwidth on the same convention as the observation block,
        # gamma = 1/(2 median^2). (spatial_kernel.py and kern_cd_joint.py use 1/median^2 --
        # a factor 2 in the exponent.)
        n = pts.shape[0]
        s = min(n, 20000)
        idx = rng.choice(n, size=s, replace=False) if n > s else np.arange(n)
        sample = pts[torch.as_tensor(idx, device=pts.device)].cpu().numpy().astype(np.float64)
        base_gamma = RBF(gamma=self.p.gamma_act).fit(sample).gamma
        # gamma_act_scale multiplies the heuristic. It is the knob controlling how much the
        # action channel matters at all: as gamma_a -> 0 every chunk embeds to the same
        # constant, k_act -> 1, and the product kernel reduces exactly to kern_cd_rbf.
        self._gamma_act = base_gamma * float(self.p.gamma_act_scale)

        d = int(self.p.rff_dim)
        W = rng.standard_normal((D, d)) * np.sqrt(2.0 * self._gamma_act)
        b = rng.uniform(0.0, 2.0 * np.pi, size=d)
        self._rff_W = torch.as_tensor(W, dtype=chunks.dtype, device=chunks.device)
        self._rff_b = torch.as_tensor(b, dtype=chunks.dtype, device=chunks.device)

    def _rff(self, u: torch.Tensor) -> torch.Tensor:
        """Random Fourier feature map: sqrt(2/d) cos(u W + b)."""
        return np.sqrt(2.0 / int(self.p.rff_dim)) * torch.cos(u @ self._rff_W + self._rff_b)

    def _action_features(self, chunks: torch.Tensor) -> np.ndarray:
        """[N, B, H, D] chunks -> [N, F] unit-norm chunk-batch embeddings.

        Time-aligned (the method): stack the H per-timestep RFF vectors, F = H*d, so the
        inner product pairs timesteps only at equal h.
        Pooled (the control): average over h as well, F = d, which compares every timestep
        against every other and is blind to ordering.
        """
        N, B, H, D = chunks.shape
        d = int(self.p.rff_dim)
        time_aligned = bool(self.p.time_aligned)
        out = np.empty((N, H * d if time_aligned else d), dtype=np.float64)

        rows = max(1, self._RFF_ELEM_BUDGET // (B * H * d))
        for s in range(0, N, rows):
            blk = chunks[s : s + rows]
            n = blk.shape[0]
            phi = self._rff(blk.reshape(-1, D)).reshape(n, B, H, d)
            if time_aligned:
                mu = (phi.sum(dim=1) / (B * np.sqrt(H))).reshape(n, H * d)
            else:
                mu = phi.mean(dim=(1, 2))
            out[s : s + n] = mu.cpu().numpy().astype(np.float64)

        # The 1/(B sqrt(H)) prescale above is exactly cancelled by this normalisation; it
        # is kept for fidelity to the definition and to hold pre-normalisation magnitudes
        # at O(1). k(x,x) = 1 comes entirely from here.
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)


# Estimator settings, shared with kern_cd_rbf: exact O(m^3) Cholesky path.
_ESTIMATOR = dict(lam=1.0e-5, rank=None, pivot="rp")
# gamma_obs: bandwidth of the observation RBF. gamma_act: bandwidth of the spatial RBF the
# RFF map approximates. Set independently by the median heuristic, both fit on calibration.
_KERNELS = dict(
    gamma_obs="median",
    gamma_act="median",
    # Multipliers on the two heuristics. Absolute bandwidths are not portable across tasks
    # (the action median spans 40 on push_chair to 2.4e5 on sorting), so the relative
    # weighting of the two blocks is swept through these rather than through gamma itself.
    gamma_obs_scale=1.0,
    gamma_act_scale=1.0,
    rff_dim=256,
)


class KernCDProd(_KernCDProd):
    """Time-aligned product kernel: the method."""

    name = "kern_cd_prod"

    params = dict(
        **_ESTIMATOR,
        **_KERNELS,
        time_aligned=True,
        seed=0,  # replaced with the run seed by EvaluationManager.evaluate
    )


class KernCDProdPool(_KernCDProd):
    """Horizon-pooled control: identical but blind to the ordering within a chunk.

    Isolates whether a gain comes from time alignment specifically or merely from having
    an action channel at all -- the distinction that collapsed the signature claim in
    docs/kern_cd_action_channel.md.
    """

    name = "kern_cd_prod_pool"

    params = dict(
        **_ESTIMATOR,
        **_KERNELS,
        time_aligned=False,
        seed=0,
    )
