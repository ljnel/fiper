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
Thus, what differentiates the methods is the choice of kernel.

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
Here, $\mu_A$ is a vector kernel mean embedding (KME) of the action chunk. Depending on how this is defined, we obtain the methods explained below. In each of these methods, $\varphi$ is the feature map associated with a RBF kernel; in order to obtain a finite-dimensional representation, we use Random Fourier Features. Note that this makes the methods stochastic (seed-dependent).

### Displacement (_disp_)
A baseline that uses only the displacement of each action chunk.
$$
\mu_A = \frac{1}{B}\sum_b \varphi(a_{bH}-a_{b1})
$$

### Signature kernel (_sig_)
Uses the level-2 term of the truncated path signature (levels 0 and 1 are discarded).
$$
\mu_A = \frac{1}{B}\sum_b \varphi(S^{2}(a_b))
$$
In practice, we add a time channel (scaled appropriately) to $a_b$ before obtaining the signature features.

### Flattened (_flat_)
Another simple baseline that flattens each action chunk.
$$
\mu_A = \frac{1}{B}\sum_b \varphi(\operatorname{vec}(a_b))
$$

### Sum kernel (_sum_)
$$
\mu_A = \frac{1}{BH}\sum_b\sum_h \varphi(a_{bh})
$$

## Hyperparameter settings
- In kern_cd, fix $\lambda = 1e^{-5}$
- Use n_components = 128 Random Fourier Features
- In total, there are at most 3 kernel bandwidths: $\sigma_o$ on observation embeddings, $\sigma_\varphi$ inside the feature map $\varphi$, and $\sigma_A$ on the mean embeddings $\mu_A$. Each is set by the median heuristic on the fit set, independently and in the order $\sigma_o$, $\sigma_\varphi$, $\sigma_A$.

# Implementation details:
- The GP posterior variance should use my kern_cd implementation, imported
from cd_project; use the exact (not low-rank) version.
- Use leave-one-out scores with kern_cd to calibrate the classification threshold (i.e. do not calibrate on the train set)
- We only implement the score for invidual time steps; combining scores across time steps and thresholding happens downstream in the FIPER benchmark.
- Push T's train/calibration dataset includes some failed episodes; these should
be filtered before fitting / calibrating kern_cd
- RFF via sklearn's RBFSampler
- Tensors are used raw: keep configs/eval/base.yaml's normalization defaults

