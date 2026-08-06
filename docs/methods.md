# Methods

Idealized descriptions of the methods in [my_methods/methods/](../my_methods/methods/).
Implementation details (batching, dtypes, standardisation epsilons) are omitted.

## Setup

A rollout step $t$ carries

$$
o_t \in \mathbb{R}^{d_o}, \qquad
A_t = \big(a^{(1)}_t, \dots, a^{(B)}_t\big), \qquad
a^{(b)}_t = \big(a^{(b)}_{t,1}, \dots, a^{(b)}_{t,H}\big) \in \mathbb{R}^{H \times 3},
$$

an observation embedding and a batch of $B$ sampled action chunks, each a length-$H$ path
in position space. Calibration gives $m$ such steps, $\{(o_i, A_i)\}_{i=1}^m$.

Every method below is: **a feature map $z = \Phi(o, A)$, plus a kernel $k$ on it**. kern_cd is
then fitted on $Z = \{z_i\}_{i=1}^m$ and its score

$$
s(z) \;=\; k(z,z) - k_Z(z)^\top (K + \lambda m I)^{-1} k_Z(z),
\qquad \lambda = 10^{-5},
$$

is the per-step anomaly score (FIPER windows and thresholds it downstream). Calibration steps
are scored leave-one-out, in closed form:

$$
s^{\mathrm{LOO}}_i \;=\; \frac{1}{\big[(K + \lambda m I)^{-1}\big]_{ii}} - \lambda m .
$$

Bandwidths written $\gamma_\bullet$ are set by the median heuristic on calibration data,
$\gamma = 1/(2\,\mathrm{med}^2)$ with $\mathrm{med}$ the median pairwise distance.

| name | $\Phi$ | $k$ |
|---|---|---|
| `example` | $o$ | — (Mahalanobis, no kern_cd) |
| `kern_cd_rbf` | $o$ | RBF |
| `kern_cd_poly` | $o$ | polynomial, degree 3 |
| `kern_cd_rbf_disp` | $[\,o \mid \text{chunk displacement}\,]$ | RBF |
| `kern_cd_rbf_sig` | $[\,o \mid \text{mean chunk signature}\,]$ | RBF |
| `kern_cd_rbf_sigk` | $[\,o \mid \text{mean embedding of chunk signatures}\,]$ | RBF |
| `kern_cd_prod` | $(o, A)$ | $\text{RBF}(o,o') \cdot \langle \hat\mu_A, \hat\mu_{A'}\rangle$ |
| `kern_cd_prod_pool` | $(o, A)$ | same, horizon-pooled |

---

## `example`

Reference implementation of the method contract; no kernel.

$$
\hat\mu = \frac{1}{m}\sum_i o_i, \qquad
\hat\Sigma_\epsilon = \hat\Sigma + \epsilon\,\frac{\operatorname{tr}\hat\Sigma}{d_o} I,
\qquad
s(o) = (o - \hat\mu)^\top \hat\Sigma_\epsilon^{-1} (o - \hat\mu).
$$

---

## `kern_cd_rbf`, `kern_cd_poly`

Observation embeddings only, $z = o$:

$$
k_{\mathrm{rbf}}(o, o') = \exp\!\big(-\gamma_o \|o - o'\|^2\big),
\qquad
k_{\mathrm{poly}}(o, o') = \Big(\tfrac{1}{d_o}\langle o, o'\rangle + 1\Big)^{3}.
$$

---

## `kern_cd_rbf_disp`, `kern_cd_rbf_sig`, `kern_cd_rbf_sigk`

Concatenate the observation with a per-step **action-chunk mean embedding**
$\mu_t \in \mathbb{R}^{d_a}$, then run one RBF on the concatenation. Writing
$\tilde{x} = (x - \hat{\mu}_x)/\hat\sigma_x$ for per-coordinate standardisation against
calibration statistics,

$$
z_t = \Big[\;\frac{\tilde{o}_t}{\sqrt{d_o}} \;\;\Big|\;\; \sqrt{g_a}\,\frac{\tilde\mu_t}{\sqrt{d_a}}\;\Big],
\qquad g_a = 1,
$$

so that after standardisation each block contributes equally to $\|z - z'\|^2$ regardless of its
width. Because an RBF on a concatenation factorises, this is the **joint** kernel

$$
k(z, z') = \exp\!\big(-\gamma\|\tilde{o} - \tilde{o}'\|^2 / d_o\big)\cdot
           \exp\!\big(-\gamma g_a \|\tilde\mu - \tilde\mu'\|^2 / d_a\big).
$$

The three variants differ only in $\mu_t$. Let $\hat{a}$ denote the time-augmented, dilated chunk

$$
\hat{a}_h = \Big(\theta\, a_h,\; \tfrac{h-1}{H-1}\Big) \in \mathbb{R}^{4},
\qquad
\theta = \Big(\operatorname{median}_{t,b} \textstyle\sum_h \|a_{h+1} - a_h\|\Big)^{-1},
$$

and $S^{\le 2}(\hat a) = \big(\int d\hat a,\; \iint d\hat a \otimes d\hat a\big) \in \mathbb{R}^{4 + 16}$
its level-2 truncated signature.

$$
\begin{aligned}
\textsf{disp:} \quad
&\mu_t = \frac{1}{B}\sum_b \big(a^{(b)}_{t,H} - a^{(b)}_{t,1}\big) &&\in \mathbb{R}^{3} \\[4pt]
\textsf{sig:} \quad
&\mu_t = \frac{1}{B}\sum_b S^{\le 2}\big(\hat{a}^{(b)}_t\big) &&\in \mathbb{R}^{20} \\[4pt]
\textsf{sigk:} \quad
&\mu_t = \frac{1}{B}\sum_b \varphi\Big(S^{\le 2}\big(\hat{a}^{(b)}_t\big)\Big) &&\in \mathbb{R}^{128}
\end{aligned}
$$

with $\varphi$ a random Fourier feature map (drawn once on calibration),

$$
\varphi(x) = \sqrt{\tfrac{2}{D}}\,\cos(W^\top x + b),
\quad W \sim \mathcal{N}(0, 2\gamma_s I),\;\; b \sim \mathcal{U}[0, 2\pi],
\quad \langle \varphi(x), \varphi(y)\rangle \approx e^{-\gamma_s\|x-y\|^2}.
$$

`sig` and `sigk` differ only in how the $B$ chunks are pooled: `sig` averages the signatures
(keeping their mean), `sigk` averages a characteristic feature map (keeping the whole
distribution of chunk signatures).

---

## `kern_cd_prod`, `kern_cd_prod_pool`

An explicit product kernel on $(o, A)$, with no standardisation or block balancing:

$$
k\big((o,A), (o',A')\big)
= \underbrace{\exp\!\big(-\gamma_o\|o - o'\|^2\big)}_{k_{\mathrm{obs}}}
\cdot \underbrace{\big\langle \hat\mu_A, \hat\mu_{A'} \big\rangle}_{k_{\mathrm{act}}} .
$$

With $\varphi: \mathbb{R}^3 \to \mathbb{R}^{d}$ an RFF map as above (bandwidth $\gamma_a$), the
action embedding stacks the per-timestep features and is normalised:

$$
\mu_A = \frac{1}{B\sqrt{H}}\sum_b \bigoplus_{h=1}^{H} \varphi\big(a^{(b)}_h\big) \in \mathbb{R}^{H d},
\qquad
\hat\mu_A = \frac{\mu_A}{\|\mu_A\|},
\qquad k_{\mathrm{act}}(A,A) = 1 .
$$

The direct sum $\bigoplus_h$ pairs timesteps only at **equal $h$**, so the implied per-chunk kernel
is the time-aligned sum kernel and $k_{\mathrm{act}}$ is the cosine kernel between chunk-batch mean
embeddings:

$$
k_{\mathrm{traj}}(x, y) = \sum_{h=1}^{H} \exp\!\big(-\gamma_a\|x_h - y_h\|^2\big),
\qquad
k_{\mathrm{act}}(A, A') \propto \frac{1}{B^2}\sum_{b,b'} k_{\mathrm{traj}}\big(a^{(b)}, a'^{(b')}\big).
$$

**`kern_cd_prod_pool`** (control) averages over $h$ as well,

$$
\mu_A = \frac{1}{BH}\sum_b \sum_h \varphi\big(a^{(b)}_h\big) \in \mathbb{R}^{d},
\qquad
k_{\mathrm{traj}}(x,y) = \sum_{h, h'} \exp\!\big(-\gamma_a\|x_h - y_{h'}\|^2\big),
$$

which compares every timestep against every other and is therefore blind to ordering within a
chunk.

Two remarks:

- $k_{\mathrm{act}}$ is a normalised *linear* kernel on mean embeddings, not
  $\exp(-\gamma\,\mathrm{MMD}^2)$, so the joint kernel is **not** an RBF on a concatenation and
  cannot be obtained by concatenating blocks as in the variants above.
- $\gamma_a \to 0 \Rightarrow k_{\mathrm{act}} \to 1$, recovering `kern_cd_rbf` exactly; the
  method is a strict generalisation of that baseline.
