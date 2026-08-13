Look at @specs/methods.md  and its implementation in @my_methods/. There's a subtle issue with the LOO calibration in that time steps across trajectories are pooled, so LOO with one time step still calibrates on other steps from the same trajectory, which are correlated with this one.

See whether LOO over trajectories can improve my headline scores in data/results. (TWA etc.) By LOO over trajectories, I mean: to find the LOO score of a data point, leave out not just this point but all steps from this trajectory.

Please let me know how this impacts the shapes of the most important arrays (e.g. train array, and so on; do not report on arrays that are only relevant as implementation details).

Start your experiment simply and cheaply, gradually widening the scope only if the results so far are not interesting. That is, start with kern_cd_obs and the smallest task / dataset. Do not modify existing files and .md docs.