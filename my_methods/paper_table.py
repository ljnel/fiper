"""Paper-style comparison table over every method in the results store.

    python -m my_methods.paper_table

Layout mirrors FIPER Table 1: methods as rows, tasks as column groups, each
group = TWA(up) / Acc(up) / DT(down), plus an Average block. Best-in-column bolded
(all ties, not just the first). Emits booktabs LaTeX and compiles it to PDF, all
into data/results/.

Regenerates data/results/summaries/summary_00.csv itself rather than reading whatever
is there: that file is overwritten in place (overwrite_summary: "all"), and
`python -m my_methods.summary` leaves it in the *task-averaged* shape with no Task
column, which this script needs. Passing average_columns=["Quantile"] keeps Task.
"""

import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.results_manager import ResultsManager  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data/results")
SUMMARY = os.path.join(OUT_DIR, "summaries/summary_00.csv")
OUT_TEX = os.path.join(OUT_DIR, "comparison_table.tex")             # bare tabular
OUT_STANDALONE = os.path.join(OUT_DIR, "comparison_table_standalone.tex")
OUT_PDF = os.path.join(OUT_DIR, "comparison_table.pdf")

# Display names match method_name_mapping in configs/results/base.yaml.
METHOD_NAME = {
    "entropy": "ACE",
    "tc": "STAC",
    "similarity": "PCA-kmeans",
    "logpzo": "logpZO",
    "rnd_a": "RND-A",
    "rnd_oe": "RND-OE",
    "rnd_oe_and_entropy": "FIPER",
    "kern_cd_obs": "KernCD-Obs",
    "kern_cd_disp": "KernCD-Disp",
    "kern_cd_sig": "KernCD-Sig",
    "kern_cd_flat": "KernCD-Flat",
    "kern_cd_sum": "KernCD-Sum",
}
# Published baselines first, then the KernCD family; a rule separates the two blocks.
BASELINES = ["entropy", "tc", "similarity", "logpzo", "rnd_a", "rnd_oe",
             "rnd_oe_and_entropy"]
# Order follows specs/methods.md: the observation-only control, then the four action
# channels (displacement, signature, flattened, sum).
KERNCD = ["kern_cd_obs", "kern_cd_disp", "kern_cd_sig", "kern_cd_flat", "kern_cd_sum"]
METHOD_ORDER = BASELINES + KERNCD

TASK_ORDER = ["push_t", "sorting", "stacking", "pretzel", "push_chair"]
TASK_NAME = {"push_t": "Push-T", "sorting": "Sorting", "stacking": "Stacking",
             "pretzel": "Pretzel", "push_chair": "Push-Chair"}
METRICS = ["TWA", "Accuracy", "Det. Time"]
METRIC_HDR = {"TWA": "TWA", "Accuracy": "Acc", "Det. Time": "DT"}
HIGHER_BETTER = {"TWA": True, "Accuracy": True, "Det. Time": False}

CAPTION = (r"Failure prediction on the FIPER benchmark. \textbf{Bold} is best in column, "
           r"\underline{underline} runner-up. Best window and threshold "
           r"selected by TWA, averaged over quantiles. Baselines are 5-seed runs; "
           r"the KernCD family is single-seed, so sub-0.01 gaps are not meaningful. "
           r"KernCD-Obs has no action channel, so every gap against it is what that "
           r"method's chunk representation buys.")

# ---------------------------------------------------------------- data
# FIPER-paper footing: quantile-averaged, best window+threshold+HID selected by TWA.
ResultsManager(os.path.join(ROOT, "configs"),
               os.path.join(ROOT, "data")).create_summary(average_columns=["Quantile"])

df = pd.read_csv(SUMMARY)
present = set(df["Method"])
missing = [m for m in METHOD_ORDER if m not in present]
if missing:
    print(f"note: absent from the results store, dropped: {missing}")
METHOD_ORDER = [m for m in METHOD_ORDER if m in present]
df = df[df["Method"].isin(METHOD_ORDER)]

val = {m: {meth: {} for meth in METHOD_ORDER} for m in METRICS}
for _, r in df.iterrows():
    for m in METRICS:
        val[m][r["Method"]][r["Task"]] = float(r[m])

avg = {m: {meth: (float(np.mean([val[m][meth][t] for t in TASK_ORDER if t in val[m][meth]]))
                  if val[m][meth] else np.nan)
           for meth in METHOD_ORDER} for m in METRICS}

blocks = [(TASK_NAME[t], t) for t in TASK_ORDER] + [("Average", "__avg__")]


def cell(metric, meth, task):
    return avg[metric].get(meth, np.nan) if task == "__avg__" else val[metric][meth].get(task, np.nan)


# Best and runner-up *values* per column, so ties are all marked rather than just the
# first seen. Ranking is on the printed 3-decimal value, not the raw float: two cells that
# read identically in the table must get the same mark, or the reader sees a bold 0.900
# beside an underlined 0.900 and no way to tell why.
DECIMALS = 3
rank = {}
for _, tkey in blocks:
    for m in METRICS:
        vals = {round(v, DECIMALS)
                for v in (cell(m, me, tkey) for me in METHOD_ORDER) if not np.isnan(v)}
        # Runner-up is the next *distinct* value, so a tie for first has no second place.
        ordered = sorted(vals, reverse=HIGHER_BETTER[m])
        rank[(tkey, m)] = ordered[:2]


def mark(metric, meth, tkey):
    """0 = best in column, 1 = runner-up, None = neither."""
    v = cell(metric, meth, tkey)
    if np.isnan(v):
        return None
    ordered = rank.get((tkey, metric), [])
    r = round(v, DECIMALS)
    return next((i for i, b in enumerate(ordered) if r == b), None)


# ---------------------------------------------------------------- LaTeX
def fmt(v, m):
    if np.isnan(v):
        return "--"
    s = f"{v:.{DECIMALS}f}"
    if m == 0:
        return rf"\textbf{{{s}}}"
    # \underline rather than ulem's \uline: plain LaTeX, so the bare tabular stays
    # \input-able without the paper adding a package.
    return rf"\underline{{{s}}}" if m == 1 else s


n_metric_cols = len(blocks) * len(METRICS)
lines = [r"\setlength{\tabcolsep}{4.2pt}", r"\renewcommand{\arraystretch}{1.15}",
         r"\begin{tabular}{l" + "c" * n_metric_cols + "}", r"\toprule"]

# Group header + cmidrules delimiting each task block.
lines.append(" & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{b}}}" for b, _ in blocks) + r" \\")
lines.append(" ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(blocks))))

sub = []
for _, tkey in blocks:
    for m in METRICS:
        arr = r"$\uparrow$" if HIGHER_BETTER[m] else r"$\downarrow$"
        sub.append(f"{METRIC_HDR[m]}{arr}")
lines.append("Method & " + " & ".join(sub) + r" \\")
lines.append(r"\midrule")

for ri, meth in enumerate(METHOD_ORDER):
    row = [METHOD_NAME[meth]]
    for _, tkey in blocks:
        for m in METRICS:
            row.append(fmt(cell(m, meth, tkey), mark(m, meth, tkey)))
    lines.append(" & ".join(row) + r" \\")
    if ri + 1 < len(METHOD_ORDER) and meth in BASELINES and METHOD_ORDER[ri + 1] in KERNCD:
        lines.append(r"\midrule")

lines += [r"\bottomrule", r"\end{tabular}"]

with open(OUT_TEX, "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", OUT_TEX)

# The table is ~10in wide. Wrapping it in a minipage sized to \linewidth would clip it,
# because standalone's \linewidth defaults to ~4.8in and the page box is sized to the
# minipage rather than its overflowing content. Box the table first, then size the caption
# parbox to the measured width (\wd\tblbox) so the two always line up.
standalone = [
    r"\documentclass[border=12pt]{standalone}",
    r"\usepackage{booktabs}",
    r"\usepackage[T1]{fontenc}",
    r"\newsavebox{\tblbox}",
    r"\begin{document}",
    r"\sbox{\tblbox}{\footnotesize",
    *lines,
    r"}",
    r"\begin{tabular}{@{}c@{}}",
    r"\usebox{\tblbox} \\[14pt]",
    rf"\parbox[t]{{\wd\tblbox}}{{\scriptsize {CAPTION}}}",
    r"\end{tabular}",
    r"\end{document}",
]
with open(OUT_STANDALONE, "w") as f:
    f.write("\n".join(standalone) + "\n")
print("wrote", OUT_STANDALONE)

# ---------------------------------------------------------------- compile
if shutil.which("pdflatex") is None:
    sys.exit("pdflatex not found; wrote .tex only")

proc = subprocess.run(
    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
     "-output-directory", OUT_DIR, OUT_STANDALONE],
    capture_output=True, text=True,
)
if proc.returncode != 0:
    tail = "\n".join(proc.stdout.splitlines()[-25:])
    sys.exit(f"pdflatex failed:\n{tail}")

built = os.path.join(OUT_DIR, "comparison_table_standalone.pdf")
shutil.copyfile(built, OUT_PDF)
for ext in (".aux", ".log"):
    stale = os.path.join(OUT_DIR, "comparison_table_standalone" + ext)
    if os.path.exists(stale):
        os.remove(stale)
print("wrote", OUT_PDF)
