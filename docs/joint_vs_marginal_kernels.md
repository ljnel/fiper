# Joint kernels vs. score-level fusion for failure detection

Why OR-ing a kernel detector on observations with a kernel detector on action
chunks is *strictly* weaker than running one kernel detector on the joint
`(observation, action)` space — and under exactly what conditions the difference
disappears.

The short version: score-level fusion collapses each channel to a scalar before
combining, and that collapse is irreversible. Anything that is anomalous only in
the *relationship* between the channels is invisible to every fusion rule, not
just to `or` and `and`.

This is background for the `kern_cd` + action-chunk-signature combination
measured in [`my_experiments/sig_study/REPORT.md`](../my_experiments/sig_study/REPORT.md) §5.

---

## 0. Setup and notation

`O` is the observation space (in FIPER: the `obs_embeddings` vector), `A` the
action-chunk space (the `action_preds` batch, or any feature of it — flattened
chunk, path signature, whatever). Write `X = O × A`.

Calibration data is `{(oᵢ, aᵢ)}ᵢ₌₁..m` drawn from a joint law `P_OA`, with
marginals `P_O` and `P_A`. Note `m` counts **timesteps**, not episodes:

| task | calibration episodes | calibration steps `m` |
|---|---:|---:|
| stacking | 50 | 2760 |
| sorting | 50 | 2076 |
| push_t | 50 | 1452 |
| pretzel | 10 | 517 |
| push_chair | 10 | **49** |

`k_o` and `k_a` are PSD kernels on `O` and `A`, with RKHSs `H_o, H_a` and feature
maps `φ_o, φ_a`. `⟨·,·⟩` is the RKHS inner product.

### The detector

Both the marginal and the joint detector are the same object — `kern_cd`'s
regularised support score — instantiated on different spaces:

```
s_λ(x) = k(x,x) − k_xᵀ (K + λm I)⁻¹ k_x,     Kᵢⱼ = k(xᵢ,xⱼ),  (k_x)ᵢ = k(x,xᵢ)
```

Higher = more anomalous. The first lemma says what this quantity *is*, which is
what makes the rest geometric rather than algebraic.

---

## 1. The support score is a projection residual

**Lemma 1.** For every `x`,

```
s_λ(x) = min over c ∈ ℝᵐ of  { ‖φ(x) − Σᵢ cᵢ φ(xᵢ)‖² + λm‖c‖² }.
```

*Proof.* Expand the objective:
`f(c) = k(x,x) − 2cᵀk_x + cᵀKc + λm‖c‖²`.
It is strictly convex in `c`, and `∇f(c) = −2k_x + 2(K + λmI)c`, so the minimiser
is `c* = (K + λmI)⁻¹k_x`. Substituting and using `(K + λmI)c* = k_x`:

```
f(c*) = k(x,x) − 2c*ᵀk_x + c*ᵀ(K + λmI)c*
      = k(x,x) − 2c*ᵀk_x + c*ᵀk_x
      = k(x,x) − k_xᵀ(K + λmI)⁻¹k_x  =  s_λ(x).   ∎
```

**Corollary 1.1.** Let `V = span{φ(xᵢ)}`. Then `s_λ(x) ↓ dist²(φ(x), V)` as
`λ ↓ 0`, and for all `λ > 0`, `dist²(φ(x), V) ≤ s_λ(x) ≤ k(x,x)`.

*Proof.* The `λ‖c‖²` penalty is nonneg and decreasing to 0, so `s_λ` decreases to
the unpenalised minimum, which is by definition the squared distance from `φ(x)`
to the span. The upper bound is `c = 0`. ∎

So: **the detector measures how far the test point's feature vector lies from the
linear span of the calibration features.** Everything below is a statement about
that span.

---

## 2. Score-level fusion is blind to dependence

Let `s_o : O → ℝ` and `s_a : A → ℝ` be the two marginal scores, fitted
independently — `s_o` on `{oᵢ}` with kernel `k_o`, `s_a` on `{aᵢ}` with `k_a`.
Define the **marginal profile**

```
Φ : X → ℝ²,   Φ(o,a) = (s_o(o), s_a(a)).
```

**Definition.** A set `E ⊆ X` is *Φ-saturated* if `x ∈ E` and `Φ(x') = Φ(x)`
together imply `x' ∈ E`. Equivalently, `E` is a union of fibers `Φ⁻¹(t)`.

**Lemma 2 (fiber invariance).** Let `ψ : ℝ² → ℝ` be *any* fusion rule and
`D = ψ ∘ Φ` the fused detector. Then `D` is constant on every fiber of `Φ`, and
the realisable decision regions `{D > τ}` are exactly the Φ-saturated sets.

*Proof.* If `Φ(x) = Φ(x')` then `D(x) = ψ(Φ(x)) = ψ(Φ(x')) = D(x')`, giving
constancy, hence `{D > τ}` is a union of fibers. Conversely if `E` is
Φ-saturated, `1_E` is well defined as a function of `Φ(x)`, so `ψ := 1_{Φ(E)}`
realises it. ∎

This covers `max` (OR), `min` (AND), sums, products, and any learned combiner —
including ones fitted on labelled failures. The obstruction is not the choice of
`ψ`; it is that `Φ` has already been applied.

**Corollary 2.1 (accept regions are rectangles).** With OR at thresholds
`τ_o, τ_a`, the accept region is
`{s_o ≤ τ_o} × {s_a ≤ τ_a}` — a Cartesian product. With AND it is the complement
of `{s_o > τ_o} × {s_a > τ_a}`. Neither can be, say, a diagonal band.

Now the consequence that matters. Write `supp` for support.

**Proposition 2.2 (blindness to dependence).** Suppose the marginal supports are
strictly larger than the joint in the sense
`supp P_OA ⊊ supp P_O × supp P_A`, and pick

```
(o*, a*) ∈ (supp P_O × supp P_A) \ supp P_OA.
```

In the large-sample, small-`λ` limit, `s_o(o*) → 0` and `s_a(a*) → 0`. Hence
`Φ(o*,a*) → (0,0) = Φ(in-distribution point)`, and by Lemma 2 **no fusion rule
`ψ` assigns `(o*,a*)` a different score than a genuinely in-distribution pair.**

*Proof sketch.* `o* ∈ supp P_O` means every neighbourhood of `o*` has positive
`P_O`-mass, so as `m → ∞` the calibration set `{oᵢ}` accumulates at `o*`; by
continuity of `φ_o`, `φ_o(o*) ∈ closure(span{φ_o(oᵢ)})`, so
`dist²(φ_o(o*), V_o) → 0`, and Corollary 1.1 gives `s_o(o*) → 0`. Same for `a*`.
The pair `(o*, a*)` is therefore in the same fiber as any in-distribution pair,
whose marginal scores also vanish. ∎

The hypothesis `supp P_OA ⊊ supp P_O × supp P_A` is exactly the statement *"the
two channels are dependent"* at the level of supports. It is the generic
situation in robot rollouts: the action chunk the policy emits is strongly
constrained by what it is looking at.

---

## 3. Product kernels see the joint

**Lemma 3 (tensor feature map).** If `k_o, k_a` are PSD with feature maps
`φ_o, φ_a`, then `k := k_o · k_a` is PSD on `X`, with feature map
`φ = φ_o ⊗ φ_a` into the Hilbert tensor product `H_o ⊗ H_a`.

*Proof.* On elementary tensors the inner product satisfies
`⟨u⊗v, u'⊗v'⟩ = ⟨u,u'⟩⟨v,v'⟩`, so

```
⟨φ(o,a), φ(o',a')⟩ = ⟨φ_o(o),φ_o(o')⟩ · ⟨φ_a(a),φ_a(a')⟩ = k_o(o,o')·k_a(a,a')
```

which exhibits `k` as a Gram kernel, hence PSD. (This is the constructive proof
of the Schur product theorem.) ∎

Write `uᵢ := φ_o(oᵢ)`, `vᵢ := φ_a(aᵢ)`, so `φ(oᵢ,aᵢ) = uᵢ ⊗ vᵢ`. Three subspaces
matter:

```
V_joint := span{ uᵢ ⊗ vᵢ  :  i = 1..m }          what the joint data spans
V_o     := span{ uᵢ },   V_a := span{ vᵢ }        what each marginal spans
V_marg  := V_o ⊗ V_a = span{ uᵢ ⊗ vⱼ  :  i,j }    what the marginals jointly certify
```

**Lemma 4 (the gap).** `V_joint ⊆ V_marg`, with

```
dim V_joint ≤ m           dim V_marg = (dim V_o)(dim V_a) ≤ m².
```

In particular whenever `dim V_o · dim V_a > m` the inclusion is strict, and the
codimension of `V_joint` inside `V_marg` is up to `m² − m`.

*Proof.* Each generator `uᵢ ⊗ vᵢ` of `V_joint` is one of the generators
`uᵢ ⊗ vⱼ` of `V_marg` (take `j = i`), giving the inclusion. `V_joint` has `m`
generators. For `V_marg`, if `{ũ_p}` and `{ṽ_q}` are bases of `V_o, V_a` then
`{ũ_p ⊗ ṽ_q}` is a basis of the tensor product, of size `dim V_o · dim V_a`. The
strictness and codimension follow by counting. ∎

This lemma is the whole story, and it has a one-line reading:

> **The tensor product of two vectors, each lying in its own marginal span, need
> not lie in the span of the tensor products.**

`V_marg \ V_joint` is precisely the set of directions that are *marginally
certified but jointly unobserved* — i.e. the dependence structure. Marginal
detectors are, by construction, blind to it; the product-kernel detector measures
distance to `V_joint` and therefore sees it.

**Theorem 5 (separation).** Let `(o*, a*)` satisfy `φ_o(o*) ∈ V_o` and
`φ_a(a*) ∈ V_a`, so both marginal scores vanish at `λ → 0`. Let `wᵢ := uᵢ ⊗ vᵢ`,
let `G` be their Gram matrix, and suppose

```
|⟨φ(o*,a*), wᵢ⟩| ≤ ε   for all i,        λ_min(G) ≥ γ > 0.
```

Then the joint score obeys

```
s₀^joint(o*, a*) ≥ k_o(o*,o*)·k_a(a*,a*) − m ε² / γ,
```

while `s_o(o*) = s_a(a*) = 0`.

*Proof.* By Corollary 1.1, `s₀ = ‖φ‖² − ‖P_V φ‖²` with `V = V_joint`. Writing
`W = [w₁ … w_m]`, the projection satisfies
`‖P_V φ‖² = (Wᵀφ)ᵀ G⁻¹ (Wᵀφ) ≤ ‖Wᵀφ‖² / λ_min(G) ≤ mε²/γ`. And
`‖φ‖² = ⟨φ_o(o*)⊗φ_a(a*), φ_o(o*)⊗φ_a(a*)⟩ = k_o(o*,o*)k_a(a*,a*)`. ∎

For normalised kernels (`k(x,x) = 1`, e.g. RBF) the bound reads
`s₀^joint ≥ 1 − mε²/γ`: the joint detector saturates at its maximum possible
score while both marginals report a perfect zero.

---

## 4. A worked example

Two calibration clusters, perfectly coupled: `(o=A, a=1)` and `(o=B, a=2)`. Test
the pairing `(o=A, a=2)` — ordinary in each channel, never seen as a pair.

Idealise each cluster as a point. Let `u₁=φ_o(A), u₂=φ_o(B), v₁=φ_a(1),
v₂=φ_a(2)`, unit norm, with cross terms `⟨u₁,u₂⟩ = ⟨v₁,v₂⟩ = ε`.

**Marginal detectors.** `V_o = span{u₁,u₂} ∋ u₁`, so `s_o(A) = 0` exactly.
Likewise `v₂ ∈ V_a` gives `s_a(2) = 0`. By Lemma 2, *no* fusion rule flags it.

**Joint detector.** `V_joint = span{u₁⊗v₁, u₂⊗v₂}`, and

```
⟨u₁⊗v₂, u₁⊗v₁⟩ = ⟨u₁,u₁⟩⟨v₂,v₁⟩ = ε
⟨u₁⊗v₂, u₂⊗v₂⟩ = ⟨u₁,u₂⟩⟨v₂,v₂⟩ = ε
```

The Gram of the two generators is `G = [[1, ε²],[ε², 1]]` (because
`⟨u₁⊗v₁, u₂⊗v₂⟩ = ⟨u₁,u₂⟩⟨v₁,v₂⟩ = ε²`), so `G⁻¹ = (1−ε⁴)⁻¹[[1,−ε²],[−ε²,1]]`
and with `Wᵀφ = (ε, ε)ᵀ`:

```
‖P_V φ‖² = (ε,ε) G⁻¹ (ε,ε)ᵀ = 2ε²(1−ε²)/(1−ε⁴) = 2ε²/(1+ε²)

s₀^joint(A,2) = 1 − 2ε²/(1+ε²).
```

Note this is *exact*, not a bound. With cluster separation 3 and RBF
`γ = 0.5`, `ε = e^{−4.5} ≈ 0.0111`, giving `s₀ ≈ 0.99975`.

### Numerical check

`scratchpad/joint_demo.py` runs this with 200 noisy samples (`σ = 0.15`) and
`λ = 10⁻⁴`:

| test point | obs-only | act-only | OR (max) | product |
|---|---:|---:|---:|---:|
| `(A,1)` seen pair | 0.0004 | 0.0003 | 0.0004 | 0.0006 |
| `(B,2)` seen pair | 0.0003 | 0.0004 | 0.0004 | 0.0005 |
| **`(A,2)` unseen pairing** | **0.0004** | **0.0004** | **0.0004** | **0.9874** |
| `(far,1)` marginally OOD | 1.0000 | 0.0003 | 1.0000 | 1.0000 |

Row 3 is Proposition 2.2: both marginal scores are numerically indistinguishable
from row 1's, so they lie in the same fiber and no `ψ` can separate them.
Measured 0.9874 vs. the idealised 0.99975 — the shortfall is cluster noise and
the ridge term, both of which pull `φ` slightly into `V_joint`.

Row 4 is the converse and is just as important: when a point is *marginally*
anomalous, OR already catches it and the joint kernel adds nothing.

---

## 5. When the gap vanishes

**Proposition 6 (no gain under support independence).** If
`supp P_OA = supp P_O × supp P_A`, then in the large-sample limit
`closure(V_joint) = closure(V_marg)`, and the joint detector has no advantage.

*Proof sketch.* `V_joint`'s limit is the closed span of
`{φ_o(o)⊗φ_a(a) : (o,a) ∈ supp P_OA}`. Under the hypothesis this index set is all
of `supp P_O × supp P_A`, whose elementary tensors span exactly the closure of
`V_o ⊗ V_a`. ∎

So the entire benefit is proportional to how strictly `supp P_OA` sits inside the
product of its marginals. **No dependence, no gain** — the joint kernel then only
costs you the statistical price below.

### The statistical price

Lemma 4 cuts both ways. Approximating an in-distribution point requires
`φ(x)` to be near `V_joint`, and `V_joint` must cover a set of dimension
`d_O + d_A` rather than `d_O` and `d_A` separately. Covering to resolution `δ`:

```
joint:     N ≍ δ^{−(d_O + d_A)}          exponents ADD
marginals: N ≍ δ^{−d_O} + δ^{−d_A}       terms add
```

Under-cover `V_joint` and *genuine* in-distribution test points also sit far from
it, scoring high — the false-positive rate rises and TNR collapses. This is the
real risk in practice, and it is why the argument above is a statement about
*representable decision regions*, not a prediction about measured TWA.

For the FIPER tasks, `m` runs 1452–2760 steps on the three large tasks, which is
workable, but consecutive steps within an episode are strongly correlated, so the
effective independent sample size is far smaller than `m` — closer to the episode
count times a small factor. **push_chair has 49 calibration steps total**; a joint
kernel there is hopeless.

### The bandwidth window

With `ε` the typical cross-kernel value, Theorem 5 needs `mε²/γ ≪ 1`. Both limits
degenerate:

- `ε → 1` (bandwidth too wide): all features nearly collinear, `V_joint` fills the
  space, every score → 0. Nothing is detected, jointly or marginally.
- `ε → 0` (bandwidth too narrow): every test point is orthogonal to `V_joint`,
  so `s₀ → k(x,x)` for *everything*. Everything is anomalous.

The useful regime is bandwidth comparable to the within-cluster spread but well
below the between-cluster separation — and the joint kernel has two bandwidths to
get right rather than one, with the product coupling them.

---

## 6. Practical note: a product of RBFs is an RBF on the concatenation

If both blocks use RBF kernels, then

```
k((o,a),(o',a')) = exp(−γ_o‖o−o'‖²) · exp(−γ_a‖a−a'‖²)
                 = exp( −(γ_o‖o−o'‖² + γ_a‖a−a'‖²) )
```

which is an *anisotropic* RBF on the concatenated vector `z = (o, a)` with a
block-diagonal metric. So the joint detector needs no new kernel class: it is the
existing `kern_cd` run on `[√γ_o · o , √γ_a · a]` with a plain RBF. The two
bandwidths become one relative weighting `γ_a/γ_o` between the blocks.

(This holds only when both factors are RBF. Pairing an RBF on observations with a
*linear* kernel on signature features — the configuration
[`REPORT.md`](../my_experiments/sig_study/REPORT.md) §6 recommends for the action
side — is a genuine product kernel and not reducible this way.)

---

## 7. Summary

| | score-level fusion (`or` / `and`) | joint / product kernel |
|---|---|---|
| decision regions | Φ-saturated sets only (Lemma 2); rectangles for OR/AND | arbitrary subsets of `O × A` |
| sees `supp P_OA ⊊ supp P_O × supp P_A`? | **no**, for every fusion rule (Prop. 2.2) | yes (Thm. 5) |
| sample complexity | `δ^{−d_O} + δ^{−d_A}` | `δ^{−(d_O+d_A)}` |
| bandwidths to tune | one per channel, independently | coupled pair |
| cost | two `m×m` solves | one `m×m` solve on concatenated features |

The expressiveness argument is airtight; the statistical argument runs the other
way. Which dominates is an empirical question about how strongly observations and
action chunks are coupled in a given task, and about whether `m` covers the joint
support. FIPER's time-varying thresholds (`tvt_quantile`, `tvt_cp_band`) also
recover part of the joint structure for free, since both channels correlate with
task phase and those thresholds condition on timestep — which further narrows the
gap the joint kernel has to beat.
