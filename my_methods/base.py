"""Method base class, registry, and the adapter onto FIPER's ``BaseEvalClass``.

A method subclasses :class:`Method`, declares the tensors it wants, and implements
``fit`` (once, on the calibration split) and ``score`` (per rollout step). Everything
else -- config composition, hyperparameter identity, moving windows, conformal
thresholds, metrics -- is FIPER's and is reused unchanged.

The two pieces of glue:

* :func:`build_cfg` composes the method's ``DictConfig`` from ``configs/eval/base.yaml``
  plus the class declarations, so no ``configs/eval/{name}.yaml`` file is needed.
  Inheriting base.yaml is what makes results comparable with the built-in methods:
  identical window sizes, quantiles and threshold styles.
* :func:`make_eval_class` generates a ``BaseEvalClass`` subclass that forwards
  ``_execute_preprocessing`` -> ``fit`` and ``calculate_uncertainty_score`` -> ``score``.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import SimpleNamespace
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from evaluation.method_eval_classes import BaseEvalClass
from shared_utils import load_config

# name -> Method subclass, populated by __init_subclass__ on import.
REGISTRY: dict[str, type["Method"]] = {}

# Entries of FIPER's eval_results holding one array per step, per window size, per
# quantile and per threshold style. Dropped unless a method sets ``keep_score_arrays``.
_BULK_RESULT_KEYS = (
    "calibration_uncertainty_scores",
    "calibration_scores_by_threshold",
    "test_uncertainty_scores",
    "test_scores_by_threshold",
)


class Method:
    """Base class for a failure-prediction method.

    Subclass it, set ``name``, and implement ``score`` (and usually ``fit``). See
    ``my_methods/methods/example.py`` for the full contract with shapes.
    """

    #: Registry key. Also the label used in results (``Method`` column).
    name: str = ""

    #: Tensors to load. Available everywhere: "obs_embeddings", "action_preds".
    tensors: list[str] = ["obs_embeddings"]
    #: Tensors to load if the task has them; missing ones are skipped silently.
    optional_tensors: list[str] = []

    #: Action channels kept in ``action_preds``. "position" normalises every task to 3.
    actions: list[str] = ["position"]
    #: Extra channels, included only where the task defines them (e.g. "rotation").
    optional_actions: list[str] = []

    #: Per-tensor normalisation overrides, merged over the defaults in eval/base.yaml.
    normalize: dict[str, bool] = {}

    #: Escape hatch: n > 1 gives ``score`` the last n steps (see example.py). Rarely
    #: needed -- temporal smoothing is already applied to your scores by ``window_sizes``.
    history: int = 0

    #: Set True if the method uses no randomness, and ``--full`` will run a single seed
    #: instead of every seed in eval/base.yaml. Repeated seeds of a deterministic method
    #: reproduce identical numbers (std 0), so the extra passes are pure waste. Left
    #: False by default: guessing wrong the other way would silently discard a real
    #: variance estimate.
    deterministic: bool = False

    #: Hyperparameters. Registered as ``hparams.model``, which is what distinguishes
    #: one configuration of a method from another in the results store, so anything
    #: that changes the numbers belongs here.
    params: dict[str, Any] = {}

    #: Keep FIPER's per-step score arrays in the results (see ``_BULK_RESULT_KEYS``).
    #: They are by far the largest thing a run produces -- 293 MB per (task, method) on
    #: stacking, where writing them is ~45 s of a ~55 s evaluation -- and nothing that
    #: builds the results table reads them: ``_create_complete_df`` uses only
    #: ``test_metrics`` / ``window_sizes`` / ``quantiles``. The consumers that do want
    #: them are the optional plots (all ``create_plots: False`` in configs/results) and
    #: ``extract_warning_frames``, which hardcodes the built-in ``rnd_oe_and_entropy``.
    #: Set True on a method whose scores you want to plot per step.
    keep_score_arrays: bool = False

    # -- Injected by the adapter before fit() is called ----------------------------
    #: ``params`` as attributes, e.g. ``self.p.alpha``.
    p: SimpleNamespace
    #: The ``ProcessedRolloutDataset``, for anything the contract does not cover.
    dataset: Any
    #: "cuda" or "cpu".
    device: str
    #: Which pass is being scored: "calibration" or "test". Set before each pass, so
    #: a method can score the two differently (e.g. leave-one-out on its own fit set).
    subset: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only register classes that name themselves; intermediate base classes
        # (which inherit `name` rather than setting it) are skipped.
        name = cls.__dict__.get("name")
        if not name:
            return
        previous = REGISTRY.get(name)
        if previous is not None and previous is not cls:
            raise ValueError(
                f"Two methods claim the name '{name}': {previous.__module__}.{previous.__qualname__} "
                f"and {cls.__module__}.{cls.__qualname__}. Method names must be unique -- they key "
                "the results store."
            )
        REGISTRY[name] = cls

    def fit(self, calib: dict) -> None:
        """Fit on the calibration split. Called once, before any scoring.

        Override when the method needs to estimate something from calibration data.
        The default does nothing, which is correct for methods that are pure
        functions of a single step.
        """

    def score(self, step: dict) -> float:
        """Uncertainty score for one rollout step. Higher = more anomalous."""
        raise NotImplementedError(f"{type(self).__name__} must implement score() or score_rollout().")

    def score_rollout(self, rollout: dict) -> Any:
        """Optional: score a whole rollout at once, returning one score per step.

        Override *instead of* ``score`` when per-step scoring wastes work. Kernel
        methods are the motivating case: scoring one query at a time forces a
        triangular solve that re-streams the whole Cholesky factor per step, which
        FIPER's built-in ``kern_cd`` works around the same way for a ~30x speedup.

        The rollout dict holds the declared tensors with a leading step axis --
        ``(T, D)`` and ``(T, B, H, A)`` -- plus ``"successful"``. Return a sequence of
        ``T`` floats. Rollouts arrive in dataset order, and the calibration pass visits
        them in the same order as the data handed to ``fit``, so a method may keep its
        own position counter across calls (see ``self.subset``).

        The default delegates to ``score``, so implementing ``score`` alone is enough.
        """
        return NotImplemented

    def score_subset(self, subset: dict) -> Any:
        """Optional: score a whole subset at once, returning one score per step.

        The last rung of the same ladder as ``score_rollout``, and it takes precedence
        over it. The dict is exactly what ``fit`` receives -- the declared tensors with a
        leading step axis over every step of the subset, plus ``"episode_lengths"`` and
        ``"successful"`` -- and the steps are in the same order as ``iterate_episodes``
        visits them, so returning ``sum(episode_lengths)`` scores is unambiguous.

        Worth implementing when the per-query work is dominated by a fixed cost that a
        rollout is too small to amortise. A kern_cd score is one triangular solve against
        the m x m Cholesky factor: at 800 stacking rollouts of ~81 steps each that factor
        is re-streamed 800 times and costs 32 s, against 5 s for the identical arithmetic
        done in a few large blocks.

        The default returns ``NotImplemented``, so ``score`` or ``score_rollout`` alone is
        enough.
        """
        return NotImplemented


def build_cfg(
    method_cls: type[Method],
    base_config_path: str,
    param_overrides: dict[str, Any] | None = None,
) -> DictConfig:
    """Compose the method's eval config from ``configs/eval/base.yaml`` + declarations.

    Args:
        method_cls: The registered method class.
        base_config_path: Path to the ``configs/`` directory.
        param_overrides: Values overriding ``method_cls.params``, e.g. from a sweep.

    Returns:
        A ``DictConfig`` shaped exactly like a hand-written ``configs/eval/*.yaml``.
    """
    cfg = load_config("eval", "base", return_only_subdict=True, base_config_dir=base_config_path)
    cfg = cfg.copy()
    OmegaConf.set_struct(cfg, False)

    cfg.required_tensors = list(method_cls.tensors)
    cfg.optional_tensors = list(method_cls.optional_tensors)
    cfg.required_actions = list(method_cls.actions)
    cfg.optional_actions = list(method_cls.optional_actions)
    cfg.history_length = int(method_cls.history)

    # Merge declared normalisation over base's defaults, keeping base's shared keys
    # (mode, range_eps, limits, fit_offset) that the dataset normaliser expects.
    normalize_tensors = OmegaConf.to_container(cfg.normalize_tensors, resolve=True) or {}
    normalize_tensors.update(dict(method_cls.normalize))
    cfg.normalize_tensors = normalize_tensors

    # Resolve hparams only after the fields above are set: base.yaml defines
    # hparams.model entries as ${eval.*} interpolations, so resolving earlier would
    # bake in base's values instead of the method's.
    hparams = OmegaConf.to_container(cfg.hparams, resolve=True)
    params = dict(method_cls.params)
    params.update(param_overrides or {})
    hparams.setdefault("model", {}).update(params)
    cfg.hparams = OmegaConf.create(hparams)

    return cfg


def make_eval_class(method_cls: type[Method], params: dict[str, Any]) -> type[BaseEvalClass]:
    """Generate the ``BaseEvalClass`` subclass that runs ``method_cls``.

    FIPER's ``EvaluationManager`` instantiates eval classes with a fixed signature; this
    wraps the user's method in one without either side knowing about the other.
    """
    # A method implements score(), score_rollout() or score_subset(); the latter two
    # switch the adapter to precomputing a whole subset per pass instead of calling back
    # per step. score_subset wins when both are defined -- it is the coarser batching.
    uses_subset = method_cls.score_subset is not Method.score_subset
    uses_batching = uses_subset or method_cls.score_rollout is not Method.score_rollout

    class _MethodAdapter(BaseEvalClass):
        __doc__ = f"Adapter running {method_cls.__module__}.{method_cls.__qualname__}."

        def __init__(self, cfg, method_name, device, task_data_path, dataset, **kwargs):
            # Built before super().__init__ because that calls load_model(), which a
            # method could conceivably want to reach through self.impl.
            self.impl = method_cls()
            self.impl.p = SimpleNamespace(**params)
            self.impl.params = dict(params)
            self.impl.dataset = dataset
            self.impl.device = device
            super().__init__(cfg, method_name, device, task_data_path, dataset, **kwargs)

        def evaluate(self):
            results = super().evaluate()
            if not method_cls.keep_score_arrays:
                for key in _BULK_RESULT_KEYS:
                    results.pop(key, None)
            return results

        def _save_pickle(self, save_dir, data, filename):
            # The only caller is evaluate(), saving eval_results.pkl. Trim before the
            # dump rather than after, so the cost is never paid at all.
            if not method_cls.keep_score_arrays:
                data = {k: v for k, v in data.items() if k not in _BULK_RESULT_KEYS}
            super()._save_pickle(save_dir, data, filename)

        def _get_subset(self, subset: str):
            return self.dataset.get_subset(
                subset=subset,
                required_tensors=self.required_tensors,
                optional_tensors=self.optional_tensors,
                required_actions=self.required_actions,
                optional_actions=self.optional_actions,
                normalize_tensors=self.normalize_tensors,
                with_success_labels=True,
                history=self.cfg.history_length,
                return_episode_lengths=True,
            )

        def _execute_preprocessing(self):
            self.impl.fit(self._get_subset("calibration"))

        def _process_rollouts(self, subset: str):
            self.impl.subset = subset
            if uses_batching:
                self._batched_scores = self._precompute_scores(subset)
                self._batched_idx = 0
            return super()._process_rollouts(subset)

        def _precompute_scores(self, subset: str):
            """Score the subset ahead of the per-step loop, in the order it consumes."""
            if uses_subset:
                # get_subset concatenates episodes in the same order iterate_episodes
                # yields them (verified on both subsets of push_t and sorting: tensors,
                # episode lengths and success labels all match element for element), so
                # the flat result indexes straight into the per-step loop below.
                data = self._get_subset(subset)
                expected = int(np.sum(data["episode_lengths"]))
                out = np.asarray(self.impl.score_subset(data), dtype=float).reshape(-1)
                if out.shape[0] != expected:
                    raise ValueError(
                        f"{method_cls.__name__}.score_subset returned {out.shape[0]} scores for a "
                        f"'{subset}' subset of {expected} steps; it must return exactly one per step."
                    )
                return out

            scores = []
            # These iteration arguments must match BaseEvalClass._process_rollouts
            # exactly, or the precomputed sequence desynchronises from the per-step loop.
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
                expected = rollout[self.required_tensors[0]].shape[0]
                out = np.asarray(self.impl.score_rollout(rollout), dtype=float).reshape(-1)
                if out.shape[0] != expected:
                    raise ValueError(
                        f"{method_cls.__name__}.score_rollout returned {out.shape[0]} scores for a "
                        f"rollout of {expected} steps; it must return exactly one score per step."
                    )
                scores.append(out)
            return np.concatenate(scores) if scores else np.zeros(0)

        def calculate_uncertainty_score(self, rollout_tensor_dict, **kwargs) -> float:
            if uses_batching:
                score = self._batched_scores[self._batched_idx]
                self._batched_idx += 1
                return float(score)
            return float(self.impl.score(rollout_tensor_dict))

    _MethodAdapter.__name__ = f"{method_cls.__name__}Adapter"
    _MethodAdapter.__qualname__ = _MethodAdapter.__name__
    return _MethodAdapter


def discover() -> dict[str, type[Method]]:
    """Import every module in ``my_methods.methods`` so its classes self-register."""
    from my_methods import methods as methods_pkg

    for module in pkgutil.iter_modules(methods_pkg.__path__):
        if module.name.startswith("_"):
            continue
        importlib.import_module(f"{methods_pkg.__name__}.{module.name}")
    return REGISTRY
