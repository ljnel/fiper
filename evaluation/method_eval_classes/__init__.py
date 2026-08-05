from .base_eval_class import *
from .entropy_eval import *
from .rnd_eval import *
from .tc_eval import *
from .similarity_eval import *
from .logpzo_eval import *

# kern_cd is backed by the `cd` package (cd_project, a private repo — see pixi.toml).
# Imported defensively: without the guard, a missing or unauthorized install raises here
# and breaks `import evaluation` outright, taking every other method down with it.
try:
    from .kerncd_eval import *
    from .kerncd_joint_eval import *
except ImportError as _err:
    import warnings

    # Bind the message to a plain name: Python unbinds the `as` target when the except
    # block exits, so the stub below could not otherwise report the cause.
    _kerncd_error = str(_err)

    warnings.warn(
        f"kern_cd is unavailable ({_kerncd_error}). Install the `cd_project` dependency "
        "(needs read access to github.com/ljnel/cd_project), or drop 'kern_cd' from "
        "`methods` in configs/default.yaml to run the remaining methods.",
        stacklevel=2,
    )

    class KERNCDEval:
        """Placeholder so that requesting kern_cd fails with an actionable message.

        EvaluationManager resolves eval classes by name via ``hydra.utils.get_class``;
        without this stub the lookup would fail with an opaque import traceback that
        says nothing about the real cause.
        """

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                f"The kern_cd method requires the `cd_project` package, which failed to "
                f"import: {_kerncd_error}. Install it (needs read access to "
                "github.com/ljnel/cd_project), or remove 'kern_cd' from `methods` in "
                "configs/default.yaml."
            )

    # The joint action-channel variants share kern_cd's dependency; stub them too.
    KERNCDJOINTEval = KERNCDDISPEval = KERNCDSIGEval = KERNCDSIGRBFEval = KERNCDEval


