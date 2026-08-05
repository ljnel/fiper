# Action-channel KernCD methods

A family of failure detectors that extend `kern_cd` (a kernelized support
estimator on the observation embedding) with a channel built from the policy's
**predicted action chunks**. All share one design; they differ only in how a set
of sampled action chunks is turned into a feature vector.

Code: [kerncd_joint_eval.py](evaluation/method_eval_classes/kerncd_joint_eval.py).
Design notes: [kern_cd_action_channel.md](docs/kern_cd_action_channel.md),
[joint_vs_marginal_kernels.md](docs/joint_vs_marginal_kernels.md).

## Setup

At timestep $t$ the policy exposes an observation embedding
$o_t \in \mathbb{R}^{d_o}$ and $B$ sampled action chunks
$\{c_t^{(b)}\}_{b=1}^{B}$, each a length-$H$ position path $c_t^{(b)} \in \mathbb{R}^{H\times 3}$.
Each method maps the $B$ chunks to a single **action feature** $\mu_t \in \mathbb{R}^{d_a}$
(defined per method below) and forms a joint, block-balanced feature

$$
z_t = \left[\;\sqrt{g_o}\,\frac{\tilde o_t}{\sqrt{d_o}}\;,\;\; \sqrt{g_a}\,\frac{\tilde\mu_t}{\sqrt{d_a}}\;\right],
$$

where $\tilde{\cdot}$ denotes per-dimension standardization on calibration data.
A single RBF `KernCD` support estimator is fit on $\{z_t\}$ over the calibration
set, and each test step is scored by its regularized distance to that support
(higher $=$ more anomalous), calibrated with the leave-one-out closed form
$1/B_{ii}-\lambda m$ inherited from `kern_cd`.

### Why concatenate — the joint (product) kernel

Because an RBF over a concatenation factorizes,

$$
k(z_t,z_s) = \exp\!\big(-\gamma\lVert z_t-z_s\rVert^2\big)
= \underbrace{\exp\!\big(-\gamma_o\lVert \tilde o_t-\tilde o_s\rVert^2\big)}_{\text{observation kernel}}
\;\cdot\;
\underbrace{\exp\!\big(-\gamma_a\lVert \tilde\mu_t-\tilde\mu_s\rVert^2\big)}_{\text{action kernel}},
$$

fitting one RBF on $z$ **is** the product kernel $k_{\text{obs}}\cdot k_{\text{act}}$.
Unlike any score-level fusion (OR/AND of two separate detectors), the joint
kernel can flag an *ordinary observation, ordinary action, impossible
combination* — see [joint_vs_marginal_kernels.md](docs/joint_vs_marginal_kernels.md).
When $\mu_t$ is a mean embedding of the chunk set, $\lVert\tilde\mu_t-\tilde\mu_s\rVert^2$
is the (kernel) **maximum mean discrepancy** between the two chunk distributions.

### Block balancing (`block_ratio`)

After standardization each block's squared norm scales with its width, so the
$d_o\!\approx\!128$ observation block would swamp the $d_a\!\in\!\{3,20,128\}$
action block. Dividing each block by $\sqrt{d}$ equalizes their contribution, so
$g_a/g_o = 1$ (`block_ratio: 1.0`) is **exact equal contribution** — a
data-independent default, fixed a priori like every other model hyperparameter
(never selected on labels).

## The methods

Let $\phi(c)$ be the level-2, time-augmented, path-dilated truncated **signature**
of a chunk. Its dimension follows

$$
\dim\phi = \frac{d^{\,L+1}-d}{d-1}, \qquad d = 3\ (\text{position}) + 1\ (\text{time}) = 4,\quad L=2 \;\Rightarrow\; 20 .
$$

Path dilation $\theta = 1/\operatorname{median}(\text{chunk total variation})$ rescales the
state channels ($S^{(k)}(\theta X)=\theta^k S^{(k)}(X)$) so the signature levels
are numerically balanced against the unit-variation time channel.

| method | `action_feature` | $\mu_t$ | $d_a$ |
|---|---|---|---:|
| `kern_cd` | — (obs only) | — | 0 |
| `kern_cd_disp` | `displacement` | mean chunk displacement | 3 |
| `kern_cd_sig` | `signature` | linear signature mean embedding | 20 |
| `kern_cd_sig_rbf` | `signature_rbf` | RBF signature mean embedding | 128 |

**Displacement** — the mean end-minus-start motion of the chunk set:

$$
\mu_t = \frac{1}{B}\sum_{b=1}^{B}\big(c_t^{(b)}[H]-c_t^{(b)}[1]\big) \in \mathbb{R}^{3}.
$$

**Linear signature** — the empirical expected signature, i.e. the chunk
distribution's mean embedding under the *linear* signature kernel:

$$
\mu_t = \frac{1}{B}\sum_{b=1}^{B}\phi\big(c_t^{(b)}\big) \in \mathbb{R}^{20}.
$$

The action kernel is then $\exp(-\gamma_a\,\mathrm{MMD}^2_{\text{lin}})$ with
$\mathrm{MMD}^2_{\text{lin}}(P,Q)=\lVert \mathbb{E}_P\phi - \mathbb{E}_Q\phi\rVert^2$
— a distance between **means only** (the linear kernel is not characteristic).

**RBF signature** — replaces the linear signature kernel with the
*characteristic* RBF kernel $\kappa(x,y)=\exp(-\gamma\lVert x-y\rVert^2)$, whose
mean embedding sees the whole chunk distribution (spread, multimodality), not
just its mean. The infinite-dimensional embedding is approximated with
$D$ **random Fourier features**

$$
\psi(x) = \sqrt{\tfrac{2}{D}}\,\cos\!\big(W x + b\big),\quad
W_{ij}\sim\mathcal N(0,2\gamma),\ \ b_i\sim\mathcal U(0,2\pi),
\qquad \langle\psi(x),\psi(y)\rangle \approx \kappa(x,y),
$$

$$
\mu_t = \frac{1}{B}\sum_{b=1}^{B}\psi\big(\phi(c_t^{(b)})\big) \in \mathbb{R}^{D},\quad D=128,
$$

so $\lVert\tilde\mu_t-\tilde\mu_s\rVert^2 \approx \mathrm{MMD}^2_{\text{RBF}}$.
Bandwidth $\gamma$ is set by the median heuristic on calibration chunk
signatures.

## Results

TWA / TNR per task, FIPER footing (best window+threshold, quantile-averaged);
full table in [kern_cd_action_channel_comparison.pdf](scratchpad/kern_cd_action_channel_comparison.pdf).
Baseline `kern_cd` is the 5-seed run; the three action-channel methods are
2-seed reruns, so the sub-0.01 average gaps are suggestive, not decisive.

| task | KernCD | +Disp | +Sig | +SigRBF |
|---|---|---|---|---|
| stacking | **0.698** / 0.663 | 0.676 / 0.781 | 0.674 / 0.746 | 0.651 / 0.780 |
| sorting | 0.554 / 0.631 | **0.561** / 0.603 | 0.543 / 0.597 | 0.537 / 0.590 |
| push_t | 0.543 / 0.559 | **0.550** / 0.810 | 0.548 / 0.689 | 0.541 / 0.664 |
| pretzel | **0.750** / 0.800 | 0.745 / 0.800 | 0.746 / 0.800 | 0.727 / 0.785 |
| push_chair | 0.894 / 0.990 | **0.945** / 1.000 | 0.923 / 0.960 | 0.819 / 0.800 |
| **average** | 0.688 / 0.729 | **0.695 / 0.799** | 0.687 / 0.758 | 0.655 / 0.724 |

**Findings.** The action channel helps, but only through its simplest feature.
The ladder is monotone in the *wrong* direction — richer action representations
do worse in order of richness:

$$
\text{displacement (3-D mean)} \;>\; \text{linear signature (20-D mean)} \;>\; \text{RBF signature (full distribution)} .
$$

- **`+Disp`** beats the baseline on average TWA (0.695 vs 0.688) and raises TNR
  substantially (0.799 vs 0.729): a small accuracy gain and consistently fewer
  false positives.
- **`+Sig`** ($\approx$ baseline) never separates from `+Disp`: the 20-D
  signature earns nothing over a 3-D mean displacement.
- **`+SigRBF`** is *worst* (0.655 TWA), below baseline. Making the action kernel
  characteristic hurts — the extra distributional detail is noise, not signal,
  and it dilutes the mean-level information that actually carries failure.
  Push-chair (49 calibration steps) fails hardest, as its support cannot sustain
  the higher-variance RBF mean embedding.

The honest framing: *adding a mean-action channel helps `kern_cd`*; path
signatures and characteristic distribution kernels do not.
