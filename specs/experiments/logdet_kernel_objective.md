# Regularized log-determinant kernel objective

In [methods.md](../methods.md), we described tuning kernel hyperparameters via the median heuristic. Here, we consider an alternative criterion.

$$
\begin{aligned}
\max_{\theta,\ \tau} \quad & \frac{1}{n} \log \det (K_\theta + \lambda I)
+ \beta \log\left(1 - \tau - \frac{1}{\alpha n}\sum_{i=1}^{n}\big(q_i(\theta) - \tau\big)_+\right) \\
\text{subject to} \quad & 0 \le \tau < 1.
\end{aligned}
$$

Where:

- $\theta$ — kernel hyperparameters.
- $\tau$ — cutoff on the LOO scores; at the optimum it is their $(1-\alpha)$-quantile.
- $K_\theta \in \mathbb{R}^{n \times n}$ — the kernel Gram matrix at $\theta$.
- $\lambda > 0$ — ridge regularizer.
- $n$ — number of data points.
- $q_i(\theta)$ — the leave-one-out (LOO) score of kern_cd on data point $i$
- $\alpha \in (0, 1]$ — tail fraction; only the worst $\alpha n$ LOO scores drive the second term.
- $\beta > 0$ — weight between the two terms.

By the Rockafellar–Uryasev identity, minimizing the hinge sum over $\tau$ gives the conditional value-at-risk, so the second term is $\beta \log(1 - \mathrm{CVaR}_\alpha(q(\theta)))$. Since $\log$ is monotone the optimal $\tau$ is unchanged by it, so $\tau$ is the $(1-\alpha)$-quantile of the LOO scores and can be reused directly as the classification threshold. Both terms are needed: the log-determinant alone is maximized by a kernel so narrow that every point looks novel, while the CVaR term alone is minimized by one so wide that nothing does.

This criterion could be especially interesting with the action-channel methods; these have multiple kernel bandwidths which might have different optimal values.

## Instructions

- Carry out a feasibility study in my_experiments/ to determine whether this hyperparam tuning method improves headline results (TWA etc.)
- Deliverables:
    - some plots that show the hyperparams picked by this method vs those picked by the median heuristic
    - a single table summarizing all experiments, showing the effect of the method on the headline benchmark scores
    - feel free to produce other visualizations if they demonstrate something interesting
- Start with the kern_cd _flat_ method; the whole point is to see whether the criterion above can balance the bandwidths of the observation and action kernels better than just using the median heuristic separately on each
- Start with a small and cheap experiment (the smallest task), gradually widening the scope only if the results so far are not interesting
- Evaluate the criterion on a grid over $\theta$, obtaining $\tau$ and $\mathrm{CVaR}_\alpha$ by sorting the LOO scores rather than by running a solver
- Sweep $\alpha$ at fixed $\beta = 1$ before sweeping $\beta$, so that the two knobs are not confounded
- Do not run the experiment over multiple seeds (which takes too long)
- Do not modify existing files