"""Print the results summary table.

    python -m my_methods.summary          # TWA / Accuracy / Det. Time
    python -m my_methods.summary --all    # plus TPR, TNR and the std columns

Reads what is already in ``data/results/``; does not re-run the benchmark.

Which methods appear is controlled by ``filter_values.Method`` in
configs/results/base.yaml.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from evaluation import ResultsManager  # noqa: E402

# Columns dropped by default. create_summary matches names exactly, so each "_std"
# partner has to be listed alongside its metric.
DROP = [
    "ID S",
    "OOD S",
    "ID F",
    "OOD F",
    "TPR",
    "TPR_std",
    "TNR",
    "TNR_std",
    "TWA_std",
    "Accuracy_std",
    "Det. Time_std",
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="my_methods.summary", description=__doc__)
    parser.add_argument("--all", dest="show_all", action="store_true", help="also show TPR, TNR and std columns")
    args = parser.parse_args(argv)

    manager = ResultsManager(os.path.join(ROOT_DIR, "configs"), os.path.join(ROOT_DIR, "data"))
    # kwargs override the results config for this call only (results_manager.py:274),
    # so neither view needs an edit to configs/results/base.yaml.
    if args.show_all:
        manager.create_summary()
    else:
        manager.create_summary(filter_columns=DROP, drop_after_best=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
