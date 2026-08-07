# Methods

What [my_methods/methods/](../my_methods/methods/) implements, as mathematics. The
specification these are built from is [specs/methods.md](../specs/methods.md); this file
is the same content with the shared machinery written out once, and it omits
implementation detail (batching, dtypes, block sizes).

## Setup

A rollout step $t$ carries an observation embedding and a batch of $B$ sampled action
chunks, each a length-$H$ path in Cartesian position space:

$$
o \in \mathbb{R}^{d_o},
\qquad
A = \big(a_1, \dots, a_B\big),
\qquad
a_b = \big(a_{b1}, \dots, a_{bH}\big) \in \mathbb{R}^{H \times 3}.
$$

Every method is a **kernel** $k$ on the pair $(o, A)$, plus one shared support estimator.
Calibration supplies $m$ steps; kern_cd is fitted on them and scores a query by its GP
posterior variance,

$$
s(x) \;=\; k(x,x) \;-\; k_x^\top \big(K + m\lambda I\big)^{-1} k_x,
\qquad \lambda = 10^{-5},
$$

which FIPER then windows and thresholds downstream. Only per-step scores are produced
here; combining across time is not this code's business.

Two properties of the fit are worth stating explicitly.

**Leave-one-out calibration.** An exact kernel estimator interpolates its own fit set, so
in-sample calibration scores collapse toward zero and would deflate every conformal
threshold. Calibration steps are therefore scored with each point held out, in closed
form,

$$
s^{\mathrm{LOO}}_i \;=\; \frac{1}{\big[(K + m\lambda I)^{-1}\big]_{ii}} \;-\; m\lambda ,
$$

an identity that never references $k(x,x)$ and so holds for every kernel below unchanged.

**Successful episodes only.** push_t's calibration split contains 29 failed rollouts out
of 50; a support estimate for *successful* behaviour is fitted on the other 21 (350 of
1452 steps). FIPER's own threshold calibration already discards failed episodes, so this
makes the fit set and the calibration set agree. The other four tasks are all-successful
and the filter is a no-op.

## The kernel

The observation factor is common to all five:

$$
k_o(o, o') = \exp\!\left(-\frac{\|o - o'\|^2}{2\sigma_o^2}\right).
$$

`obs` is that factor alone. The other four multiply it by an action factor, an RBF on a
vector **kernel mean embedding** $\mu_A$ of the chunk batch:

$$
k\big((o,A),(o',A')\big)
= \exp\!\left(-\frac{\|o-o'\|^2}{2\sigma_o^2}\right)
\exp\!\left(-\frac{\|\mu_A-\mu_{A'}\|^2}{2\sigma_A^2}\right).
$$

Since a product of RBFs is one RBF on the concatenation of the rescaled blocks, all five
are fitted as a single $\mathrm{RBF}(\gamma = 1/2)$ on

$$
z = \big[\, o/\sigma_o \;\big|\; \mu_A/\sigma_A \,\big],
$$

with the action block simply absent for `obs`. No custom kernel object is involved, and
`obs` is exactly the $\mu_A \equiv \text{const}$ limit of the others.

## The mean embeddings

Each $\mu_A$ averages a Random Fourier Feature map $\varphi: \mathbb{R}^F \to
\mathbb{R}^{128}$ over the *parts* of $A$, where

$$
\varphi(x) = \sqrt{\tfrac{2}{128}}\,\cos(W^\top x + b),
\quad W \sim \mathcal{N}(0, \sigma_\varphi^{-2} I),\;\; b \sim \mathcal{U}[0, 2\pi),
\qquad
\langle \varphi(x), \varphi(y)\rangle \approx \exp\!\left(-\frac{\|x-y\|^2}{2\sigma_\varphi^2}\right).
$$

The draw is seeded from the run seed, so the four action methods are genuinely stochastic
across seeds; `obs` is deterministic. What differs between methods is only the choice of
part:

| method | part | $F$ | $\mu_A$ |
|---|---|---|---|
| `kern_cd_obs` | — | — | — (observation factor only) |
| `kern_cd_disp` | chunk displacement | $3$ | $\frac{1}{B}\sum_b \varphi(a_{bH}-a_{b1})$ |
| `kern_cd_sig` | chunk level-2 signature | $16$ | $\frac{1}{B}\sum_b \varphi\big(S^{2}(\hat a_b)\big)$ |
| `kern_cd_flat` | flattened chunk | $3H$ | $\frac{1}{B}\sum_b \varphi\big(\operatorname{vec}(a_b)\big)$ |
| `kern_cd_sum` | single action | $3$ | $\frac{1}{BH}\sum_b\sum_h \varphi(a_{bh})$ |

They separate cleanly along two axes. `disp` keeps only the endpoints; the other three
keep the whole chunk. Of those, `flat` is strictly time-aligned (the distance behind
$\varphi$ compares step $h$ to step $h$ and nothing else), `sum` is permutation-invariant
in $h$ and so blind to ordering, and `sig` keeps ordering through iterated integrals
rather than through alignment.

### The signature term

`sig` uses the **level-2 term only** of the truncated path signature; levels 0 and 1 are
discarded. Level 0 is the constant $1$. Level 1 is the increment $a_{bH}-a_{b1}$, which is
exactly what `kern_cd_disp` already isolates, so keeping it would prevent the two methods
from separating. Level 2 is where path shape lives: for the augmented path $\hat a$,

$$
S^{2}(\hat a) = \iint_{u<v} d\hat a_u \otimes d\hat a_v \in \mathbb{R}^{4 \times 4},
$$

whose antisymmetric part is the signed area the chunk sweeps out — which distinguishes
trajectories sharing endpoints but curving differently.

Signatures are invariant to reparameterisation, so without a time channel a chunk that
dawdles and one that moves steadily embed identically, and a chunk retracing its own path
cancels to zero. A monotone channel breaks both:

$$
\hat a_{bh} = \Big(\theta\, a_{bh},\; \tfrac{h-1}{H-1}\Big) \in \mathbb{R}^4,
\qquad
\theta = \Big(\operatorname{median}_{t,b} \textstyle\sum_h \|a_{b,h+1} - a_{bh}\|\Big)^{-1}.
$$

The channel spans $1$, so $\theta$ puts a typical chunk's spatial extent on the same
footing. An *overall* scale would wash out — level 2 is quadratic in the path, so a global
factor is a constant that the median heuristic inside $\varphi$ absorbs — but the
space-to-time ratio does not, which is what $\theta$ fixes.

## Hyperparameters

Everything is fixed a priori or set by the median heuristic; nothing is tuned on data.

- $\lambda = 10^{-5}$, exact (not low-rank) kern_cd.
- $128$ Random Fourier Features.
- At most three bandwidths, each the median pairwise distance over the fit set, taken
  **independently and in this order**: $\sigma_o$ on observation embeddings, then
  $\sigma_\varphi$ on the parts (it lives inside $\varphi$, so it must be fixed before any
  mean embedding exists), then $\sigma_A$ on the mean embeddings (which only exist once
  $\varphi$ is drawn).
- Tensors are used raw, at `configs/eval/base.yaml`'s normalisation defaults.
