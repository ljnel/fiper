from .base_eval_class import BaseEvalClass
import numpy as np
from scipy.linalg import solve_triangular
from cd.algs.kern_cd import KernCD
from cd.algs.kernels import RBF


class KERNCDEval(BaseEvalClass):
    """Kernelized support-estimation OOD detector on observation embeddings.

    Untrained alternative to RND-OE: fits a KernCD support estimator on the
    calibration observation embeddings and scores each test step by its
    regularized squared distance to the support (higher => more anomalous).

    Calibration scores use a leave-one-out (LOO) closed form so that the
    conformal thresholds are not deflated by fit-set leakage: an exact kernel
    estimator interpolates its fit set, driving in-sample calibration scores
    toward zero. The LOO score for fit point i is ``1/B_ii - lam*m`` where
    ``B = (K + lam*m*I)^-1`` and ``L`` is the stored Cholesky factor of
    ``K + lam*m*I``. Test points are genuinely out-of-fit, so they use the
    standard in-sample score.
    """

    def __init__(self, cfg, method_name, device, task_data_path, dataset, **kwargs):
        super().__init__(cfg, method_name, device, task_data_path, dataset, **kwargs)

    def _execute_preprocessing(self):
        self.embedding_dim = self.dataset.get_tensor_shape("obs_embeddings")[-1]

        obs_embeddings = self.dataset.get_subset(
            subset="calibration",
            required_tensors="obs_embeddings",
            normalize_tensors=self.normalize_tensors,
        )["obs_embeddings"]
        X = np.asarray(obs_embeddings)

        self.model = KernCD(
            kernel=RBF(gamma=self.cfg.get("gamma", "median")),
            lam=self.cfg.get("lam", 1e-5),
            rank=self.cfg.get("rank", None),  # None => exact O(m^3) path
            pivot=self.cfg.get("pivot", "rp"),
        ).fit(X)

        # Precompute LOO scores for the (ordered) calibration fit set.
        self._loo_scores = self._compute_loo_scores()
        # Validate the closed form against brute force when it is cheap to do so.
        if len(X) <= int(self.cfg.get("loo_validation_max_m", 300)):
            self._validate_loo(X)

        # State for the per-step passes (see calculate_uncertainty_score).
        self._subset = None
        self._loo_idx = 0
        self._test_scores = None
        self._score_idx = 0

    def _compute_loo_scores(self) -> np.ndarray:
        """LOO support score for every calibration fit point: 1/B_ii - lam*m."""
        if self.model.rank is not None:
            raise NotImplementedError("LOO calibration is only implemented for the exact path (rank=None).")
        m = len(self.model.data)
        lam_m = self.model.lam_ * m
        # B = A^-1 with A = L L^T, so diag(B)_i = ||L^-1 e_i||^2 = column-i sq-sum of L^-1.
        L_inv = solve_triangular(self.model.L, np.eye(m), lower=True)
        b_diag = np.einsum("ki,ki->i", L_inv, L_inv)
        return 1.0 / b_diag - lam_m

    def _validate_loo(self, X: np.ndarray):
        """Brute-force check: refit-leaving-each-out (same lam*m ridge) must match the closed form."""
        m = len(X)
        lam_m = self.model.lam_ * m
        K = self.model.kernel(X)
        brute = np.empty(m)
        for i in range(m):
            idx = np.arange(m) != i
            M = K[np.ix_(idx, idx)] + lam_m * np.eye(m - 1)
            k_vec = K[idx, i]
            brute[i] = K[i, i] - k_vec @ np.linalg.solve(M, k_vec)
        max_diff = np.abs(brute - self._loo_scores).max()
        assert np.allclose(brute, self._loo_scores, atol=1e-6, rtol=1e-5), (
            f"LOO closed-form mismatch vs brute force: max abs diff {max_diff:.2e}"
        )

    def _process_rollouts(self, subset: str):
        # Reset per-step counters at the start of each subset pass.
        self._subset = subset
        self._loo_idx = 0
        self._score_idx = 0
        # Non-calibration subsets are scored in ONE batched pass. FIPER's base loop
        # calls calculate_uncertainty_score once per step; scoring each step alone
        # forces an O(m^2), memory-bound triangular solve that re-streams the whole
        # Cholesky factor per query. Batching the whole subset amortizes that read
        # (~30x faster) and is numerically identical to per-step scoring.
        if subset != "calibration":
            self._test_scores = self._precompute_scores_batched(subset)
        return super()._process_rollouts(subset)

    def _precompute_scores_batched(self, subset: str) -> np.ndarray:
        """Score every step of a subset in per-rollout batches, in the exact order
        the base per-step loop will consume them (same iterate_episodes args)."""
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
            X = np.asarray(rollout["obs_embeddings"])
            if X.ndim > 2:  # mirror the per-step obs_embeddings[0] reduction
                X = X[:, 0]
            X = X[:, -self.embedding_dim:]
            scores.append(np.asarray(self.model.score(X)).reshape(-1))
        return np.concatenate(scores) if scores else np.zeros(0)

    def calculate_uncertainty_score(self, rollout_tensor_dict, **kwargs):
        # Calibration steps are visited in fit order (same start/end indices as the
        # fit set), so return the precomputed LOO scores in sequence.
        if self._subset == "calibration":
            score = self._loo_scores[self._loo_idx]
            self._loo_idx += 1
            return float(score)

        # Test steps: return the precomputed batched score in sequence.
        if self._test_scores is not None:
            score = self._test_scores[self._score_idx]
            self._score_idx += 1
            return float(score)

        # Fallback: direct per-step scoring (unbatched).
        obs_embeddings = rollout_tensor_dict["obs_embeddings"]
        if len(obs_embeddings.shape) > 1:
            obs_embeddings = obs_embeddings[0]
        obs_embedding = np.asarray(obs_embeddings[-self.embedding_dim:]).reshape(1, -1)
        return float(self.model.score(obs_embedding)[0])
