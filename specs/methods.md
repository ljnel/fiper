# Notation

- Use length-scale ($\sigma$) notation for kernels (leads to consistent dimensions across kernels)
- Observation embedding vector: $o$
- Batch of action chunks: $A \in \mathbb{R}^{B\times H\times D}$
- Individual action chunks: $a \in \mathbb{R}^{H\times D}$
- All our methods use the $D=3$ Cartesian coordinates of the actions

# Methods

All our methods fit a support estimator using the GP posterior variance (called kern_cd here):
$$
s(x) = k(x,x) - k_x^T (K+n\lambda I)^{-1}k_x
$$
Thus, what differentiates the methods is the choice of kernel. We use the median heuristic to choose the kernel bandwidth. When there are multiple kernel bandwidths, these are tuned separately.

## Observation-only (_obs_)
This is a baseline that uses only a kernel on observation embeddings.
$$
k((o,A), (o',A')) = 
\exp\left(-\frac{||o-o'||^2}{2\sigma_o^2}\right)
$$

## Action channel methods

In addition to the observation kernel, these methods include a kernel on action chunk batches. We use a product kernel
$$
k((o,A), (o',A')) = 
\exp\left(-\frac{||o-o'||^2}{2\sigma_o^2}\right)
\exp\left(-\frac{||\mu_A-\mu_{A'}||^2}{2\sigma_A^2}\right)
$$
Here, $\mu_A$ is a vector kernel mean embedding (KME) of the action chunk. Depending on how this is defined, we obtain the methods explained below. In each of these methods, $\varphi$ is the feature map associated with a RBF kernel; we shall represent it by Random Fourier Features with $m=128$ dimensions.

### Displacement (_disp_)
A baseline that uses only the displacement of each action chunk.
$$
\mu_A = \frac{1}{B}\sum_b \varphi(a_{bH}-a_{b1})
$$

### Signature kernel (_sig_)
Uses the signature kernel features at level 2.
$$
\mu_A = \frac{1}{B}\sum_b \varphi(S^{2}(a_b))
$$
In practice, we add a time channel to $a_b$ before obtaining the signature features.

### Flattened (_flat_)
Another simple baseline that flattens each action chunk.
$$
\mu_A = \frac{1}{B}\sum_b \varphi(\operatorname{vec}(a_b))
$$

### Sum kernel (_sum_)
$$
\mu_A = \frac{1}{BH}\sum_b\sum_h \varphi(a_{bh})
$$

# Implementation details:
- The GP posterior variance should use my kern_cd implementation, imported
from cd_project; use the exact (not low-rank) version.
- Use leave-one-out scores with kern_cd to calibrate the classification threshold (i.e. do not calibrate on the train set)
- We only implement the score for invidual time steps; combining scores across time steps and thresholding happens downstream in the FIPER benchmark.

