# Regularized log-determinant kernel objective

In [methods.md](../methods.md), we described tuning kernel hyperparameters via the median heuristic. Here, we consider an alternative criterion.

$$
\begin{aligned}
\max_{\theta,\ \tau} \quad & \frac{1}{n} \log \det (K_\theta + \lambda I) + \log(1 - \tau) \\
\text{subject to} \quad & q_i(\theta) \le \tau, \qquad i = 1, \dots, n, \\
& 0 \le \tau < 1.
\end{aligned}
$$

Where:

- $\theta$ — kernel hyperparameters.
- $\tau$ — slack variable upper-bounding every $q_i(\theta)$.
- $K_\theta \in \mathbb{R}^{n \times n}$ — the kernel Gram matrix at $\theta$.
- $\lambda > 0$ — ridge regularizer.
- $n$ — number of data points.
- $q_i(\theta)$ — the leave-one-out (LOO) score of kern_cd on data point $i$

This criterion could be especially interesting with the action-channel methods; these have multiple kernel bandwidths which might have different optimal values.

## Instructions

- Carry out a feasibility study to determine whether this hyperparam tuning method improves headline results (TWA etc.)
- Start with a small and cheap experiment (kern_cd_obs on the smallest task), gradually widening the scope only if the results so far are not interesting
- Do not run the experiment over multiple seeds (which takes too long)
- Do not modify existing files