# Path signatures for action-chunk failure prediction — viability study

**Verdict: use truncated signature *features* at level 2, linear kernel, with a
per-task path dilation. Do not use the signature kernel. Do not truncate above
level 2 — nothing above level 1 paid for itself anywhere in this study.**

The scores are competitive with FIPER's published methods and do add something
on top of `kern_cd`. But the gain comes from having an action-chunk-distribution
channel at all, not from the signature's higher-order terms — a plain chunk
displacement vector matches or beats level-2 signatures in 15 of 15 cells.

All numbers are best-TWA over FIPER's full threshold grid (3 styles × 10
quantiles × 17 windows), computed with the repo's own `compute_thresholds` /
`calculate_metrics`, so they are directly comparable to
`data/results/complete_results.csv`.

---

## 0. Validation of the harness and the signature code

| check | result |
|---|---|
| `signature()` vs `iisignature` | max abs err 2e-14 (d=2..4, level 1..4) |
| PDE signature kernel vs truncated inner product | converges; residual 9e-4 at dyadic order 2 = the finite-difference error |
| harness reproduces published `tc` TWA | sorting 0.505/0.505, stacking 0.647/0.649, push_t 0.656/0.656, pretzel 0.633/0.633, push_chair 0.750/0.750 |

The `tc` reproduction is the important one: it means a TWA produced here is on
the same footing as a TWA in the results CSV.

## 1. Scaling is the decision that matters, not the truncation level

Level *k* of a signature scales like ‖path‖ᵏ⁄k!. Raw action chunks and the
appended time channel are wildly mismatched:

| task | median chunk total variation (position) | action share of level energy, L1 → L5 |
|---|---:|---|
| sorting | 0.0020 | 1.9e-3 → 3.8e-14 |
| stacking | 0.0229 | 3.2e-2 → 4.5e-7 |
| pretzel | 0.0916 | 9.0e-2 → 5.9e-5 |
| push_chair | 0.331 | 1.9e-1 → 3.0e-3 |
| push_t | 98.9 | ~1.0, but raw level-5 magnitude is **7.6e9** |

With a time channel sweeping a unit interval, the level-*k* term contributes
exactly 1/k! — so on sorting the signature is **99.8 % time channel at level 1
and effectively 100 % by level 2**. That alone explains why the first sweep
showed byte-identical TWA for levels 2–5. push_t fails the opposite way:
unnormalised level-5 features reach 7.6e9, which is asking for trouble in fp32.

Fixing this is one line — dilate the state channels by θ = 1/median-TV, fitted
on calibration — and it is a prerequisite for the level knob to mean anything.
**If you take one thing from this study, take this.** It is also the thing most
likely to bite silently: nothing errors, the scores just stop carrying action
information.

## 2. Truncation level: 1–2, and higher levels are not worth it

After fixing the scaling, sweeping level 1..5 × {raw, TV-normalised} × 3 score
families × 5 tasks (150 + 50 configs):

| comparison | median ΔTWA | wins |
|---|---:|---|
| best level vs level 1 (sigvar, sigtc) | **+0.000** | level 1 is argmax in 18/30 |
| best level vs level 1 (sigmmd) | **+0.000** | — |
| signature L2 vs chunk displacement (= level 1) | −0.006 | **0 / 15** |
| signature L2 vs flat chunk vector (no signature) | −0.002 | 5 / 15 |

The one apparent counter-example — sigvar on pretzel going 0.677 → 0.809 at
level 4 — is on the 20-episode test set and does not replicate on any task with
a real sample size.

> A correction worth recording: an earlier version of this sweep appeared to
> show sigmmd gaining up to +0.079 TWA from higher levels. That was a bug in the
> sweep (the reference chunk pool was dilated twice, so reference and test lived
> at different scales). Corrected, the gain is +0.000.

## 3. Kernel: linear. The signature kernel is not worth its cost.

Level 2, position channels, TV-normalised, all 5 tasks × 3 families:

| kernel | median ΔTWA vs linear | wins | cost vs linear |
|---|---:|---|---|
| PDE signature kernel, dyadic 0 | +0.000 | 2/15 | 2–7× |
| PDE signature kernel, dyadic 1 | +0.000 | 2/15 | 3–40× |
| RBF on signature features, per-step median γ | −0.024 | 2/10 | ~1× |
| RBF on signature features, global γ | −0.010 | 1/15 | ~1× |

The untruncated kernel buys nothing the truncated features don't already have.
RBF actively hurts, and with a globally-fitted bandwidth it **collapses** on
sorting (TWA 0.511, AUROC 0.500 — a constant score). A per-step median bandwidth
is also conceptually wrong for `sigvar`: it renormalises away the very spread
the score is trying to measure.

## 4. Cost and memory

Per-step latency on an RTX 5090, scoring one timestep's B chunks (the way the
FIPER loop and a deployment actually call it):

| | L1 | L2 | L3 | L4 | L5 |
|---|---:|---:|---:|---:|---:|
| linear sig features, B=32 | 0.19 ms | 0.35 ms | 0.55 ms | 0.83 ms | 1.14 ms |
| linear sig features, B=256 | 0.32 ms | 0.64 ms | 1.07 ms | 1.65 ms | 2.32 ms |

Signature kernel (B×B Goursat solves per step):

| B | dyadic 0 | dyadic 1 | dyadic 2 | peak memory (dy 2) |
|---:|---:|---:|---:|---:|
| 32 | 1.2 ms | 2.5 ms | 4.9 ms | 20 MB |
| 256 | 2.6 ms | 18.2 ms | **89–104 ms** | **1.9 GB** |

So on push_t/push_chair (B=256) the signature kernel costs 18–104 ms per
timestep for zero detection gain, versus 0.64 ms for level-2 features.

**Feature dimension is the memory constraint, and only for wide action spaces.**
Position-only (a=3, d=4 with time) is trivial at any level ≤5 — 1364 floats,
≤1.4 MB per step even at B=256. Stacking's full 21-dim action vector is not:

| stacking, all 21 dims (d=22) | L1 | L2 | L3 | L4 | L5 |
|---|---:|---:|---:|---:|---:|
| feature dim | 22 | 506 | 11 154 | 245 410 | 5 399 042 |
| MB per step (B=32, fp32) | 0.003 | 0.06 | 1.4 | 31 | 691 |

Level 5 on stacking's full action vector is 5.4 M dimensions per chunk — it
would not allocate, and a 64-step batch at level 5 needs 44 GB. That is the
memory wall you were worried about, and it is entirely a function of *action
dimension*, not batch size or horizon. It never binds if you stay at level ≤2 or
restrict to the position channels. (Log-signatures would cut level-4 stacking
from 245 410 to 63 869 dims, but given level >1 buys nothing, that is solving a
problem you shouldn't have.)

## 5. Does it add anything to kern_cd?

This is the question that actually matters for your plan. Combination follows
`EvaluationManager._combine_two_methods` exactly (elementwise max/min of the
threshold-normalised scores), using the saved `kern_cd` `eval_results.pkl`:

| task | kern_cd | sigtc (L2, lin) | OR | AND |
|---|---:|---:|---:|---:|
| stacking | 0.725 | 0.704 | 0.729 | **0.741** |
| sorting | 0.595 | 0.588 | **0.624** | 0.586 |
| push_t | 0.565 | 0.609 | 0.595 | **0.618** |

Combining helps on all three (+0.016, +0.029, +0.053). The two detectors read
genuinely different things — observation embeddings vs the action-chunk
distribution — so this is the expected and encouraging result.

Two caveats before you bank it. Both numbers are maxima over the full threshold
grid, and the combination selects over a *larger* grid than either component, so
some of the gain is selection optimism; a nested-CV or held-out quantile choice
would give an honest estimate. And whichever of OR/AND wins flips by task, which
is a sign the margin is not robust.

## 6. Recommendation

Use, as the action-chunk detector:

```
signature features, level 2
time augmentation on
path dilation theta = 1 / median calibration chunk total variation
linear kernel
position channels only
score = sigtc  (signature-MMD between the overlapping parts of the chunk
                distributions at t-1 and t, mirroring the existing `tc` slicing)
```

Cost: ~0.35–0.7 ms/step on GPU, ≤1 MB/step transient. Negligible next to the
policy forward pass.

Do not use the signature kernel — 18–104 ms/step at B=256 for +0.000 TWA.

**Before publishing any signature-specific claim, run the displacement control.**
`baselines.py --level 2` gives it in one command. On this data a plain
displacement vector matches level-2 signatures in 15/15 cells, so the honest
framing of the result is "adding an action-chunk-distribution score helps
kern_cd", not "path signatures help". If you want the signature to earn its
place you need a setting where chunk *shape* rather than chunk *endpoint*
carries the failure signal — longer horizons, or a task where the arm's path
curvature matters — and none of these five tasks is that setting.

## 7. Reliability caveats

- **pretzel and push_chair have 20 test episodes each.** Their TWAs swing by
  ±0.1 between adjacent configs. Treat them as anecdote; stacking (800), sorting
  (400) and push_t (300) carry the conclusions.
- **TWA and AUROC disagree in places** — e.g. pretzel `sigtc` has TWA 0.70 with
  AUROC 0.25 (anti-correlated with failure), and published `tc` shows the same
  pattern (AUROC 0.12). A score can score well on TWA through the CP band while
  being non-monotone in failure risk. Worth understanding before trusting a
  headline TWA on the small tasks.
- Single seed; the repo's pipeline averages over 5.
- All best-TWA figures are maxima over the threshold grid, i.e. optimistic in
  absolute terms — but equally so for the published baselines they are compared
  against.

## Files

| file | what |
|---|---|
| `sigtools.py` | truncated signatures (GPU, validated vs iisignature), Goursat PDE signature kernel |
| `scores.py`, `episode_scores.py` | the three score families, per-step and batched |
| `harness.py` | FIPER threshold/metric machinery, standalone |
| `control_tc.py` | reproduces published `tc` — the harness validation |
| `diag_levels.py` | the scaling diagnostic of §1 |
| `run_sweep.py` | `scale` and `kernel` stages |
| `baselines.py` | the displacement / flat-vector controls of §2 |
| `bench_cost.py` | §4 latency and memory |
| `combine_kerncd.py` | §5 |

Results CSVs sit next to each script. `iisignature` was installed into a
scratchpad directory, not the pixi environment; nothing under `evaluation/`,
`datasets/`, `rnd/` or `configs/` was modified.
