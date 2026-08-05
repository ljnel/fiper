# Adding an action-chunk channel to `kern_cd`

Final recommendations from the signature study
([`my_experiments/sig_study/REPORT.md`](../my_experiments/sig_study/REPORT.md))
and the joint-vs-marginal analysis
([`joint_vs_marginal_kernels.md`](joint_vs_marginal_kernels.md)).

## What to build

Extend `kern_cd`'s input from the observation embedding to a concatenation with
a per-step action-chunk **mean embedding**:

```
phi_a(chunk)  = truncated signature features of one action chunk
mu_t          = (1/B) sum_b phi_a(chunk_b)          # one vector per timestep
z_t           = [ sqrt(g_o) * obs_embedding_t , sqrt(g_a) * mu_t ]
KernCD(RBF).fit(z_calibration)
```

Nothing else changes: an RBF on `z` *is* the product kernel
`RBF(obs) x exp(-g_a * MMD^2(chunks))`, because a product of RBFs is an
anisotropic RBF on the concatenation. The existing exact path, LOO calibration
and threshold machinery carry over unchanged.

Do this **instead of** OR/AND-ing a separate action detector into `kern_cd`.
Score-level fusion collapses each channel to a scalar first, and is provably
blind to observation-action dependence for *every* fusion rule — see
[`joint_vs_marginal_kernels.md`](joint_vs_marginal_kernels.md) Lemma 2 and
Prop. 2.2. The joint kernel is the only form that can flag "ordinary action,
ordinary observation, impossible combination".

## Run it as a ladder, not a single config

| rung | action features | dims | question it settles |
|---|---|---:|---|
| 0 | none — `kern_cd` today | 0 | baseline (already measured) |
| 1 | mean chunk displacement | 3 | does *any* action channel help jointly? |
| 2 | level-2 signature mean embedding | 20 | does the signature beat rung 1? |
| ref | score-level `kern_cd or/and sigtc` | — | does joint beat fusion? (§5 of REPORT) |

Rung 1 is not optional. Across the study, level-2 signatures beat level-1
displacement in **0 of 15** cells, so without rung 1 you cannot tell whether a
gain comes from the signature or merely from having an action channel at all.

Reference numbers to beat (best TWA, full threshold grid):

| task | kern_cd | best score-level fusion |
|---|---:|---:|
| stacking | 0.725 | 0.741 (and) |
| sorting | 0.595 | 0.624 (or) |
| push_t | 0.565 | 0.618 (and) |

## Settings

- **Signature**: level 2, time augmentation on, position channels only, linear
  feature map. ~0.35 ms/step.
- **Path dilation** (mandatory): scale the state channels by
  `theta = 1 / median calibration chunk total variation`, fitted per task on
  calibration. Without it the signature is 99.8 % time channel on sorting, and
  overflows toward 1e9 on push_t. This fails silently — nothing errors, the
  score just stops carrying action information.
- **Block weighting** `g_a/g_o`: the only genuinely new hyperparameter. 128
  observation dims against 20 signature dims will not balance themselves.
  Standardise each block, then grid the ratio over a decade or two, selecting on
  **calibration only**. Report sensitivity rather than the test-set argmax.
- **`lam`, `rank`, `pivot`**: unchanged from `configs/eval/kern_cd.yaml`.

## What not to do

- **The untruncated (PDE) signature kernel.** 18 ms/step at dyadic order 1 and
  89-104 ms at order 2 for B=256, peaking at 1.9 GB, for a median **+0.000** TWA
  over a linear kernel on truncated features.
- **Truncation above level 2.** Median gain over level 1 is +0.000; level 1 is
  the argmax in 18 of 30 cells.
- **RBF on signature features for the *score*.** Consistently worse (-0.024
  median), and with a globally fitted bandwidth it collapses to a constant score
  on sorting. (This is distinct from the RBF *support* kernel on `z` above,
  which is a different role.)
- **The full action vector on stacking.** Level 4 is 245 410 dims/chunk, level 5
  is 5.4 M and will not allocate. Position channels only.

## Per-task notes

- **stacking, sorting, push_t** (m = 2760 / 2076 / 1452 calibration steps) carry
  the conclusions. Run all rungs here.
- **pretzel** (517 steps, 20 test episodes) and **push_chair** (**49 steps**) are
  anecdote. Skip push_chair entirely — 49 steps cannot support a ~148-dimensional
  joint support estimate.

## What to measure

Report **TNR alongside TWA**. The joint kernel's failure mode is an
under-covered joint support inflating false positives, and TWA can mask it. If
TNR drops relative to rung 0, the joint support is undersampled — reduce the
action block's weight or its dimension before concluding the method fails.

Expect the honest framing of any positive result to be *"adding an
action-chunk-distribution channel helps `kern_cd`"*, not *"path signatures help"*,
unless rung 2 separates from rung 1.
