# Profiling the kern_cd family

Specification for a time-and-memory profile of the five methods in
[my_methods/methods/](../my_methods/methods/). These methods are specified in [specs/methods.md](../specs/methods.md); this file says what gets measured, on what, and what the output looks like.

## The question

The five methods share every line of code except `_parts`, and `kern_cd_obs` has no
action path at all. So the profile is not really "how long does each method take" — it is
**how total cost splits into a shared floor and a per-method delta**, where only the delta
is under the control of the kernel design.

Three things follow, and they set the whole design below:

1. The `kern_cd_obs` bar *is* the shared floor, measured rather than estimated: it runs
   the identical code path with the action block removed. Every other method minus `obs`
   is the exact cost of its action channel.
2. Any stage that is not `_parts`/`_embed` is common to all five, so resolving it finely
   buys nothing until it is shown to dominate.
3. The interesting axes are the input shapes to the RFF map: how many points are embedded, and how wide each point is.


## Scope

All 5 kern_cd methods, one seed, on the following tasks:

| task | type | $d_o$ | $B$ | $H$ | calib / test episodes |
|---|---|---|---|---|---|
| `pretzel` | real-world | 512 | 30 | 16 | 10 / 20 |
| `push_t` | simulation | 64 | 256 | 16 | 50 / 300 |

## Stages

Four segments per bar. `fit` runs once on the calibration split; `score_subset` then runs
twice, once per subset.

| segment | covers | runs on | method-specific |
|---|---|---|---|
| **Features (fit)** | rollout tensors → $Z$: obs extraction, host→device transfer, `_parts`, RFF embed, the three bandwidth medians, pack | $m$ rows | **yes, entirely** |
| **Estimator fit** | gram $K$, Cholesky $L$, closed-form LOO scores | $m \times m$ | no |
| **Features (test)** | the same code as above, on the test split | $N$ rows | **yes, entirely** |
| **Scoring** | blocked triangular solves against $L$, both subsets | $N \times m$ | no |

The fit/test split inside Features is kept because it is a real 3–4× and the two halves
mean different things: fit-set features are a one-off, test-set features are a per-step
deployment cost. The calibration scoring pass is folded into Scoring.

**Measure fine, display coarse.** Instrumentation sits at the ten natural seams in
[kern_cd_core.py](../my_methods/kern_cd_core.py) — `_obs`, $\sigma_o$, `_chunks`,
`_parts`, `_fit_phi`, `_embed`, $\sigma_A$, `_pack`, `KernCD.fit`, `_loo_scores`,
`model.score` — and all of it is retained in the raw output. Only the four-segment roll-up
is plotted. If Features dominates a bar, the drill-down is already there and needs no
re-run.

## Deliverable 1 — where the time goes

One stacked horizontal bar per (method, task), seconds, segmented as above. Method-major:
five groups of two bars, with the `kern_cd_obs` total drawn as a reference line, since
the comparison that matters is each method against the shared floor.

Answers: does the choice of part cost anything at all? `kern_cd_sum` builds $P = BH =
4096$ parts per step on push_t against `kern_cd_disp`'s $P = B = 256$ — sixteen times the
RFF evaluations. If that is invisible next to the $m^3$ and $N \times m$ work every method
pays identically, the action channel is free and the question is closed.

Two rules for this chart:

- **Unattributed residual is shown, not hidden.** GPU work is asynchronous, so segment
  sums equal wall clock only if `torch.cuda.synchronize()` is placed correctly at every
  boundary. The difference between wall clock and the sum of segments is drawn as its own
  segment. A large residual means the segmentation is wrong, and that should be visible
  rather than smoothed away.
- **A harness context bar accompanies the five.** Dataset rebuild (~12 s per task), the
  threshold sweep over 17 window sizes × 10 quantiles × 3 threshold styles, and metric
  computation are FIPER's costs, not the method's, but if a method is 8 s inside a 60 s
  evaluation then optimising it is not worth doing. Without this bar the chart flatters
  the methods.

## Deliverable 2 — what is big

A table of every array above ~10 MB: name, where it is created, symbolic shape, concrete
shape per task, dtype, bytes, and **lifetime** (transient within a call, or held across
one).

Symbolic shape says what breaks as tasks grow; the concrete shape pins what is true today;
lifetime is what governs peak memory. Three m×m float64 matrices alive at once in
`_loo_scores` is 183 MB at stacking's $m = 2760$ and 9.6 GB at $m = 20000$ — a fact no
single peak-memory number conveys.

Predictions worth confirming or refuting, computed from the shapes above for push_t's test
pass ($N = 9458$):

- `_chunks` materialises the whole subset's action tensor at once: `[9458, 256, 16, 3]`
  float32 is 465 MB on the device, and if the numpy source is float64 there is a 930 MB
  host array before it.
- `sum` and `flat` return **views** — their `_parts` is a `reshape` of a contiguous
  tensor and allocates nothing. `disp` allocates 29 MB. `sig` is the outlier: it builds
  `paths` (465 MB), concatenates a time channel (620 MB), then `inc` and `pre` (581 MB
  each), peaking near 2.3 GB.
- `_parts` runs on all $N$ rows before `_embed` batches. The `_RFF_ELEM_BUDGET` cap
  therefore bounds the RFF intermediate but not the parts tensor that feeds it.

If those hold, the interesting design question is whether `sig`'s `_parts` should be
batched on the same budget as `_embed`, and whether `_chunks` should stream.

## Deliverable 3 — peak memory

Peak host RSS and peak CUDA allocated (plus reserved) per segment.

On its own this number is close to useless — a bare "peak 9 GB" supports no decision. Its
purpose is as a **cross-check on deliverable 2**: sum the bytes the table says are live at
that moment and compare against the measurement. The gap is memory that was allocated
without being named, i.e. copies. Named suspects are the float64 host array behind
`_chunks`, the `.cpu().numpy().astype(np.float64)` in `_embed`, and whatever `KernCD.fit`
allocates internally.

A small gap is a genuine clean bill of health. A large gap is reported as
"unaccounted-for", not explained away: without allocation-site tracking the gap can be
measured but not attributed.

## Deliverable 4 — the scaling knobs

A table of stage → complexity → dominant symbol → which task maxes it, roughly

$$
\text{Features} = O\big(N P F + N P \cdot 128\big),
\qquad
\text{Estimator fit} = O\big(m^2 d + m^3\big),
\qquad
\text{Scoring} = O\big(N m d + N m^2 / \text{block}\big),
$$

with $d = d_o + 128$ the packed width.

This is what lets two profiled tasks say anything about the other three. It also makes the
profile falsifiable: predict stacking from push_t, then check.

## Output

A single self-contained HTML page, plus the raw per-seam measurements as CSV so the
numbers can be re-plotted without re-running anything.
