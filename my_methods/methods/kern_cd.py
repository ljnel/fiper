"""KernCD support estimation on observation embeddings.

Ports evaluation/method_eval_classes/kerncd_eval.py onto the my_methods contract.

Fits a KernCD support estimator on the calibration observation embeddings and scores
each step by its regularized squared distance to the support (higher => more anomalous).

Calibration scores use a leave-one-out closed form: an exact kernel estimator
interpolates its own fit set, so in-sample calibration scores collapse toward zero and
would deflate the conformal thresholds. The LOO score for fit point i is
``1/B_ii - lam*m`` with ``B = (K + lam*m*I)^-1``; it never references ``k(x,x)``, so it
is kernel- and feature-agnostic and the joint variants inherit it unchanged.

Registers:
    kern_cd_rbf    RBF kernel
    kern_cd_poly   inhomogeneous polynomial kernel
"""

import numpy as np
from cd.algs.kern_cd import KernCD
from cd.algs.kernels import RBF, Polynomial
from scipy.linalg import solve_triangular

from my_methods.base import Method


class _KernCDBase(Method):
    """Shared KernCD machinery. Not registered -- it sets no ``name``.

    Subclasses supply the feature map by overriding ``_fit_features`` (calibration,
    which may also fit standardisers) and ``_features`` (a rollout, scoring only).
    """

    tensors = ["obs_embeddings"]
    normalize = {"obs_embeddings": False}
    # Exact path (rank=None), no sampling: repeated seeds reproduce identical metrics.
    deterministic = True

    #: Brute-force LOO check is O(m^4); only run it where that is cheap.
    loo_validation_max_m = 300

    def fit(self, calib):
        Z = self._fit_features(calib)

        self.model = KernCD(
            kernel=self._make_kernel(),
            lam=self.p.lam,
            rank=self.p.rank,  # None => exact O(m^3) path
            pivot=self.p.pivot,
        ).fit(Z)

        self.loo_scores = self._compute_loo_scores()
        if len(Z) <= self.loo_validation_max_m:
            self._validate_loo(Z)

        # Position counters for the per-pass scoring below.
        self._cursor = 0
        self._pass = None

    def _make_kernel(self):
        # The built-in kern_cd's config predates the kernel switch and carries no
        # `kernel` key, so
        # its absence must keep meaning RBF -- the hparams dict is what identifies a
        # configuration in the results store, and adding a key would fork the HID.
        kernel_type = getattr(self.p, "kernel", "rbf")
        if kernel_type == "rbf":
            return RBF(gamma=self.p.gamma)
        if kernel_type == "poly":
            return Polynomial(degree=self.p.degree, gamma=self.p.gamma, coef0=self.p.coef0)
        raise ValueError(f"Unknown kernel '{kernel_type}' (expected 'rbf' or 'poly').")

    # -- features --------------------------------------------------------------
    def _prep_obs(self, obs) -> np.ndarray:
        # No dtype cast: the stored tensors are float32 and the built-in kern_cd fits on
        # them as-is. Promoting to float64 here is *more* accurate but does not reproduce
        # -- with the degree-3 polynomial kernel the difference reaches 0.01 TWA, because
        # cubing amplifies the relative error and the poly Gram matrix is far worse
        # conditioned than the RBF one. The joint variants override this: their reference
        # implementation does cast (kerncd_joint_eval.py:_prep_obs).
        X = np.asarray(obs)
        if X.ndim > 2:  # history_length > 1: keep the current step
            X = X[:, 0]
        return X

    def _fit_features(self, calib) -> np.ndarray:
        return self._prep_obs(calib["obs_embeddings"])

    def _features(self, rollout) -> np.ndarray:
        return self._prep_obs(rollout["obs_embeddings"])

    # -- leave-one-out calibration scores --------------------------------------
    def _compute_loo_scores(self) -> np.ndarray:
        """LOO support score for every calibration fit point: 1/B_ii - lam*m.

        B = A^-1 with A = L L^T, so diag(B)_i = ||L^-1 e_i||^2, the column-i squared
        sum of L^-1.
        """
        if self.model.rank is not None:
            raise NotImplementedError("LOO calibration is only implemented for the exact path (rank=None).")
        m = len(self.model.data)
        L_inv = solve_triangular(self.model.L, np.eye(m), lower=True)
        b_diag = np.einsum("ki,ki->i", L_inv, L_inv)
        return 1.0 / b_diag - self.model.lam_ * m

    def _validate_loo(self, Z: np.ndarray) -> None:
        """Refit-leaving-each-out (same lam*m ridge) must match the closed form."""
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

    # -- scoring ---------------------------------------------------------------
    def score_rollout(self, rollout):
        # Reset the position counter whenever a new pass starts. The calibration pass
        # visits rollouts in the same order as the data handed to fit(), so the LOO
        # scores line up positionally with the steps being scored.
        if self._pass != self.subset:
            self._pass = self.subset
            self._cursor = 0

        n = len(rollout["obs_embeddings"])
        if self.subset == "calibration":
            scores = self.loo_scores[self._cursor : self._cursor + n]
        else:
            # One batched solve per rollout rather than one per step: the triangular
            # solve re-streams the whole Cholesky factor per call, so amortising it
            # over a rollout is ~30x faster and numerically identical.
            scores = np.asarray(self.model.score(self._features(rollout))).reshape(-1)

        self._cursor += n
        return scores


class KernCDRBF(_KernCDBase):
    name = "kern_cd_rbf"

    params = dict(
        gamma="median",  # RBF bandwidth heuristic (or a float)
        lam=1.0e-5,  # ridge / GP noise (sigma^2 = lam * m)
        rank=None,  # null => exact O(m^3) path (calibration sets are small)
        pivot="rp",  # only used on the low-rank path (rank set)
    )


class KernCDPoly(_KernCDBase):
    """Inhomogeneous polynomial kernel: k(x,y) = (gamma*<x,y> + coef0)^degree."""

    name = "kern_cd_poly"

    params = dict(
        kernel="poly",
        degree=3,
        gamma="dimension",  # gamma = 1/d, keeps gamma*<x,y> ~ O(1) on un-normalized embeddings
        coef0=1.0,
        lam=1.0e-5,
        rank=None,
        pivot="rp",
    )
