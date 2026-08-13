# Hyperparameter sensitivity

We defined some kernel-based methods in [specs/methods.md](../methods.md). These can be run on the FIPER benchmark to measure headline metrics like TWA. It was found that the action channel is not at all effective for improving the score.

In this experiment, we want to answer the questions

**how stable are the headline results with respect to hyperparameter choices?**

**are there hyperparameter choices that make the action-channel methods significantly more competitive?**

- Conduct an experiment in a new script to determine whether this is the case.
- Keep the experiment cheap (it should run in a few minutes) - consider only the _sum_ kern_cd method, use only 1 task
- sweep the hyperparams in powers of 2
- hyperparams to sweep: kernel bandwidths (all 3 of them), lambda, number of RFF components
- measure the 3 headline metrics
- report numerically the bulk of results (or a summary statistic of them)
- for the most interesting axes, create a simple, informative plot (e.g. suppose that sigma_A and lambda move the results the most - in this case, create a heatmap showing results along these axes only). You may generate additional data for these interesting axes, but once again, keep the experiment cheap
- Use your judgement to select the grid size and bounds of each hyperparam
- Do not modify existing files