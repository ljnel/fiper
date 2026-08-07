# Regularized log-determinant kernel objective

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
