"""A one-file-per-method layer over FIPER.

Write a failure-prediction method as a single file in ``my_methods/methods/``
declaring what data it wants and implementing ``fit`` and ``score``; run it with
``python -m my_methods.run <name> --task <task>``.

Nothing under the original FIPER tree is modified. The adapter in
:mod:`my_methods.base` generates a ``BaseEvalClass`` subclass at runtime and the
runner in :mod:`my_methods.run` feeds it to FIPER's own ``EvaluationManager``, so
methods written here are evaluated by exactly the same threshold, window and
metric machinery as ``entropy``, ``rnd_oe`` and the rest.
"""

from .base import REGISTRY, Method, build_cfg, discover, make_eval_class

__all__ = ["Method", "REGISTRY", "build_cfg", "make_eval_class", "discover"]
