"""Shared machinery for the kern_cd family described in ``specs/methods.md``.

Every method in that spec is the same support estimator -- the GP posterior variance

    s(x) = k(x,x) - k_x^T (K + n lambda I)^{-1} k_x

fitted by ``cd.algs.kern_cd.KernCD`` on the exact (rank=None) path -- under a different
kernel. The kernel is always a product of an observation factor and (except for the
observation-only baseline) an action factor,

    k((o,A), (o',A')) = exp(-||o-o'||^2 / 2 sigma_o^2) * exp(-||mu_A-mu_A'||^2 / 2 sigma_A^2),

so a method is fully specified by how it turns a step's batch of action chunks
``A in R^{B x H x 3}`` into the vector kernel mean embedding ``mu_A``. Each of those is

    mu_A = mean_p phi(x_p),      x_p = the p-th "part" of A,

with ``phi`` a Random Fourier Feature map of an RBF kernel. A subclass therefore only has
to implement :meth:`KernCDMethod._parts`, returning the ``[N, P, F]`` tensor of parts;
``P = B`` when the part is a whole chunk (disp / sig / flat) and ``P = B*H`` when it is a
single action (sum). Everything else -- bandwidths, the RFF draw, the estimator, the
leave-one-out calibration scores, batching -- lives here.

Two facts keep this cheap:

* A product of two RBFs is one RBF on the concatenation of the rescaled blocks. Packing
  ``z = [o/sigma_o | mu_A/sigma_A]`` and using ``RBF(gamma=1/2)`` reproduces the product
  kernel exactly, so no custom ``Kernel`` subclass is needed and the obs-only baseline is
  literally the same code path with the action block removed.
* The feature build (up to ~4e7 RFF evaluations per pass) runs in torch on the GPU; only
  the ``[N, d_obs + 128]`` packed matrix ever reaches numpy.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch
from cd.algs.kern_cd import KernCD
from cd.algs.kernels import RBF
from cd.utils.misc import median_distance
from scipy.linalg import solve_triangular
from sklearn.kernel_approximation import RBFSampler

from my_methods.base import Method

#: Points fed to the median heuristic. Matches ``cd.utils.misc.median_distance``'s own
#: cap, which is what the observation bandwidth already uses.
_MEDIAN_MAX_POINTS = 3000

#: Cap on elements in one RFF intermediate ([rows, P, n_components]), bounding peak
#: memory. push_t alone has 9458 * 256 * 16 = 3.9e7 actions in the test pass.
_RFF_ELEM_BUDGET = 64_000_000

#: Query rows per ``KernCD.score`` call. The cost of a score is a triangular solve
#: against the m x m Cholesky factor, so the factor should be streamed over as many
#: queries at once as memory allows: at stacking's m=2760 and 64883 test steps, one
#: query block per rollout takes 32 s against 5 s for a handful of large blocks. 8192
#: rows holds the (block, m) intermediates to a few hundred MB.
_SCORE_BLOCK = 8192


@contextlib.contextmanager
def _skip_condition_number():
    """Suppress the diagnostic ``np.linalg.cond`` call inside ``KernCD._fit_exact``.

    That line exists only to put a condition number in an INFO log record, but it is
    evaluated unconditionally and it is a full SVD of the m x m regularized kernel
    matrix: at stacking's m=2760 it takes 17.4 s, against 0.8 s for the gram matrix and
    Cholesky factorisation that constitute the actual fit. Over 5 methods x 5 tasks that
    is minutes of pure diagnostics.

    Patching numpy for the duration of the fit is a workaround, not the fix -- the fix is
    to guard the diagnostic upstream in cd_project (e.g. behind
    ``logger.isEnabledFor(logging.INFO)``). Nothing else runs inside the ``with`` block,
    the original is restored in a ``finally``, and the return value is only ever
    formatted into a log string, so no result depends on it.
    """
    original = np.linalg.cond
    np.linalg.cond = lambda *args, **kwargs: float("nan")
    try:
        yield
    finally:
        np.linalg.cond = original


class KernCDMethod(Method):
    """kern_cd with an observation kernel and an optional action-chunk kernel.

    Subclasses set ``name`` and, unless they are the observation-only baseline, implement
    :meth:`_parts`. Set ``uses_actions = False`` to drop the action factor entirely.
    """

    tensors = ["obs_embeddings", "action_preds"]
    #: Position channels only: D = 3 everywhere. Overrides the base ``optional_actions``
    #: ([rotation]), which would otherwise widen the action feature on pretzel/stacking.
    actions = ["position"]
    optional_actions = []
    normalize = {"obs_embeddings": False, "action_preds": False}
    #: The RFF draw is seeded from the run seed, so scores genuinely vary across seeds.
    deterministic = False

    #: False for the obs-only baseline; the action block is then omitted entirely.
    uses_actions = True

    params = dict(
        lam=1.0e-5,  # ridge / GP noise: the matrix factored is K + lam*m*I
        n_components=128,  # Random Fourier Features in phi
        seed=0,  # replaced with the run seed by EvaluationManager.evaluate
    )

    #: Brute-force LOO cross-check is O(m^4); only run it where that is cheap.
    _loo_validation_max_m = 300

    # ------------------------------------------------------------------ subclass hook
    def _parts(self, chunks: torch.Tensor) -> torch.Tensor:
        """``[N, B, H, 3]`` chunk batches -> ``[N, P, F]`` parts, one per phi evaluation.

        The mean embedding is ``mu_A = mean over the P parts of phi(part)``, so the choice
        of part *is* the method: the chunk displacement, its level-2 signature, its
        flattening, or each individual action.
        """
        raise NotImplementedError

    def _fit_parts(self, chunks: torch.Tensor) -> torch.Tensor:
        """As ``_parts``, but on the fit set, where a subclass may also fit a statistic.

        The only user is the signature method, which fits its path dilation here.
        """
        return self._parts(chunks)

    # ------------------------------------------------------------------------- fitting
    def fit(self, calib):
        self._rng = np.random.default_rng(int(getattr(self.p, "seed", 0) or 0))

        # push_t's calibration split contains 29 failed rollouts out of 50. A support
        # estimator for *successful* behaviour must not be fitted on them, and FIPER's own
        # threshold calibration already ignores them (base_eval_class.py:_get_thresholds
        # keeps only episodes with successful=True), so dropping them here makes the fit
        # set and the calibration set agree. The other four tasks are all-successful and
        # the mask is a no-op there.
        lengths = np.asarray(calib["episode_lengths"], dtype=int)
        successful = np.asarray(calib["successful"], dtype=bool)
        keep = np.repeat(successful, lengths)
        self._fit_episode_lengths = lengths
        self._fit_episode_kept = successful

        obs = self._obs(calib["obs_embeddings"])[keep]
        # sigma_o: median heuristic on the fit set, taken first (it depends on nothing).
        self._sigma_o = self._median(obs)

        if self.uses_actions:
            chunks = self._chunks(calib["action_preds"])[torch.as_tensor(keep, device=self.device)]
            parts = self._fit_parts(chunks)
            # sigma_phi second: it is the bandwidth *inside* phi, so it must be fixed
            # before any mean embedding exists.
            self._fit_phi(parts)
            mu = self._embed(parts)
            # sigma_A last: it is a bandwidth *on* the mean embeddings, which only exist
            # once phi is drawn.
            self._sigma_A = self._median(mu)
            Z = self._pack(obs, mu)
        else:
            Z = self._pack(obs, None)

        # RBF(gamma=1/2) on the packed [o/sigma_o | mu/sigma_A] is exactly the product of
        # the two RBFs: ||z-z'||^2 = ||o-o'||^2/sigma_o^2 + ||mu-mu'||^2/sigma_A^2.
        with _skip_condition_number():
            self.model = KernCD(kernel=RBF(gamma=0.5), lam=self.p.lam, rank=None).fit(Z)

        self.loo_scores = self._loo_scores()
        if len(Z) <= self._loo_validation_max_m:
            self._validate_loo(Z)

    # ------------------------------------------------------------------------ features
    def _obs(self, obs) -> np.ndarray:
        """Observation embeddings as ``[N, d_obs]`` float64."""
        X = np.asarray(obs)
        if X.ndim > 2:  # history_length > 1: keep the current step
            X = X[:, 0]
        return X.astype(np.float64)

    def _chunks(self, action_preds) -> torch.Tensor:
        """``action_preds`` -> torch ``[N, B, H, 3]`` on the run device."""
        t = torch.as_tensor(np.asarray(action_preds), dtype=torch.float32, device=self.device)
        if t.ndim == 5:  # [N, robots, B, H, D]
            if t.shape[1] != 1:
                raise NotImplementedError(f"{type(self).__name__} does not support multi-robot rollouts.")
            t = t[:, 0]
        assert t.ndim == 4, f"unexpected action_preds ndim {t.ndim} (want [N,B,H,D])"
        return t

    def _median(self, X) -> float:
        """Median pairwise distance, on a seeded subsample. Never returns 0."""
        A = X if isinstance(X, np.ndarray) else self._subsample(X).cpu().numpy().astype(np.float64)
        if len(A) > _MEDIAN_MAX_POINTS:
            A = A[self._rng.choice(len(A), _MEDIAN_MAX_POINTS, replace=False)]
        med = median_distance(np.ascontiguousarray(A))
        return float(med) if med > 0 else 1.0

    def _subsample(self, parts: torch.Tensor) -> torch.Tensor:
        """Flatten ``[N, P, F]`` parts to ``[n, F]``, subsampled for the median heuristic.

        Drawn on the device: the full stack reaches 4e7 rows, which is not worth moving
        to the host just to take a median over 3000 of them.
        """
        flat = parts.reshape(-1, parts.shape[-1])
        n = flat.shape[0]
        if n > _MEDIAN_MAX_POINTS:
            idx = self._rng.choice(n, _MEDIAN_MAX_POINTS, replace=False)
            flat = flat[torch.as_tensor(idx, device=flat.device)]
        return flat

    def _fit_phi(self, parts: torch.Tensor) -> None:
        """Draw the RFF map approximating ``exp(-||x-y||^2 / 2 sigma_phi^2)``.

        sklearn's ``RBFSampler`` is the source of the draw (``W ~ N(0, 2 gamma)``,
        ``b ~ U[0, 2 pi)``, ``phi(x) = sqrt(2/d) cos(xW + b)``); its weights are then moved
        to torch so the ~1e10 cosine evaluations per run happen on the GPU rather than in
        ``RBFSampler.transform``.
        """
        self._sigma_phi = self._median(parts)
        sampler = RBFSampler(
            gamma=1.0 / (2.0 * self._sigma_phi**2),
            n_components=int(self.p.n_components),
            random_state=int(getattr(self.p, "seed", 0) or 0),
        )
        # fit() only reads the feature count off X; no statistics are learned.
        sampler.fit(np.zeros((1, parts.shape[-1]), dtype=np.float64))
        self._phi_W = torch.as_tensor(sampler.random_weights_, dtype=torch.float32, device=self.device)
        self._phi_b = torch.as_tensor(sampler.random_offset_, dtype=torch.float32, device=self.device)
        self._phi_scale = float(np.sqrt(2.0) / np.sqrt(int(self.p.n_components)))

    def _embed(self, parts: torch.Tensor) -> np.ndarray:
        """``[N, P, F]`` parts -> ``[N, n_components]`` mean embeddings ``mu_A``, batched."""
        N, P, _ = parts.shape
        d = int(self.p.n_components)
        out = np.empty((N, d), dtype=np.float64)
        rows = max(1, _RFF_ELEM_BUDGET // (P * d))
        for s in range(0, N, rows):
            blk = parts[s : s + rows]
            phi = self._phi_scale * torch.cos(blk @ self._phi_W + self._phi_b)
            out[s : s + blk.shape[0]] = phi.mean(dim=1).cpu().numpy().astype(np.float64)
        return out

    def _pack(self, obs: np.ndarray, mu: np.ndarray | None) -> np.ndarray:
        """``[o/sigma_o | mu/sigma_A]``, the argument of the single RBF(gamma=1/2)."""
        if mu is None:
            return obs / self._sigma_o
        return np.concatenate([obs / self._sigma_o, mu / self._sigma_A], axis=1)

    def _features(self, rollout) -> np.ndarray:
        obs = self._obs(rollout["obs_embeddings"])
        if not self.uses_actions:
            return self._pack(obs, None)
        mu = self._embed(self._parts(self._chunks(rollout["action_preds"])))
        return self._pack(obs, mu)

    # -------------------------------------------------------- leave-one-out calibration
    def _loo_scores(self) -> np.ndarray:
        """Support score of every fit point with itself held out: ``1/B_ii - lam*m``.

        An exact kernel estimator interpolates its own fit set, so in-sample calibration
        scores collapse toward zero and would deflate every conformal threshold. With
        ``B = (K + lam*m*I)^-1 = (L L^T)^-1``, ``B_ii = ||L^-1 e_i||^2`` is the column-i
        squared sum of ``L^-1``. The identity never references ``k(x,x)``, so it holds for
        every kernel in the family unchanged.
        """
        m = len(self.model.data)
        L_inv = solve_triangular(self.model.L, np.eye(m), lower=True)
        b_diag = np.einsum("ki,ki->i", L_inv, L_inv)
        return 1.0 / b_diag - self.model.lam_ * m

    def _validate_loo(self, Z: np.ndarray) -> None:
        """Refitting with each point left out (same lam*m ridge) must match the closed form."""
        m = len(Z)
        lam_m = self.model.lam_ * m
        K = self.model.kernel(Z)
        brute = np.empty(m)
        for i in range(m):
            idx = np.arange(m) != i
            M = K[np.ix_(idx, idx)] + lam_m * np.eye(m - 1)
            k_vec = K[idx, i]
            brute[i] = K[i, i] - k_vec @ np.linalg.solve(M, k_vec)
        max_diff = np.abs(brute - self.loo_scores).max()
        assert np.allclose(brute, self.loo_scores, atol=1e-6, rtol=1e-5), (
            f"LOO closed-form mismatch vs brute force: max abs diff {max_diff:.2e}"
        )

    # ------------------------------------------------------------------------- scoring
    def score_subset(self, data) -> np.ndarray:
        """One score per step of the subset, in dataset order."""
        Z = self._features(data)
        scores = np.concatenate(
            [np.asarray(self.model.score(Z[s : s + _SCORE_BLOCK])).reshape(-1) for s in range(0, len(Z), _SCORE_BLOCK)]
        )
        if self.subset == "calibration":
            self._splice_loo(scores, data)
        return scores

    def _splice_loo(self, scores: np.ndarray, data) -> None:
        """Replace the fit set's in-sample scores with their leave-one-out counterparts.

        Steps of *dropped* (failed) calibration episodes keep the ordinary score they
        already have: those episodes were never in the fit set, so nothing is in-sample
        about them. FIPER's thresholds ignore them either way -- this only keeps the
        calibration score array complete and consistently defined.
        """
        lengths = np.asarray(data["episode_lengths"], dtype=int)
        assert np.array_equal(lengths, self._fit_episode_lengths), (
            "the calibration pass is out of step with the data handed to fit(): "
            f"episode lengths {lengths} vs {self._fit_episode_lengths}"
        )
        bounds = np.concatenate([[0], np.cumsum(lengths)])
        cursor = 0
        for i, kept in enumerate(self._fit_episode_kept):
            if not kept:
                continue
            n = lengths[i]
            scores[bounds[i] : bounds[i + 1]] = self.loo_scores[cursor : cursor + n]
            cursor += n
        assert cursor == len(self.loo_scores)


class ChunkPartsMixin:
    """For methods whose part is a whole chunk: ``mu_A = 1/B sum_b phi(f(a_b))``."""

    def _chunk_feature(self, chunks: torch.Tensor) -> torch.Tensor:
        """``[N, B, H, 3]`` -> ``[N, B, F]``, one feature vector per chunk."""
        raise NotImplementedError

    def _parts(self, chunks: torch.Tensor) -> torch.Tensor:
        return self._chunk_feature(chunks)
