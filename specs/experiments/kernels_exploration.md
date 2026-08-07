# Kernels Exploration

This is a short experiment in which we'll try to get some visual insight into how well the kernels work.
In particular, whether the action kernels add meaninful information to the kernel on observation embeddings.

## Relevant files
The kernels are contained in my_methods/methods/

## Relevant kernels
- RBF kernel on observation embeddings (referred to as "observation kernel" hereafter)
- Sum-kernel (time-aligned) on action chunks, with mean embedding of action chunk batches
- Signature kernel on action chunks, with mean embeddings of action chunks batches (similar to kern_cd_rbf_sig)
Please tune hyperparams consistently with my_methods/methods.

## Deliverables
Two saved plots (pdf and png), consisting of 3 subplots each.
Each subplot shows a kernel matrix on train data; individual rows / columns are ~100 randomly selected steps from test episodes of Push T ordered by success  / failure.
Use roughly the same number of success / failure episodes, and randomly select steps from the last half of episodes.
Don't select two steps from the same episode.
Goal: see whether there's clear separation in the kernel matrix between success / failure blocks.

Plot 1:
- observation kernel, sum kernel, product kernel of these 2 kernels
Plot 2:
- observation kernel, signature kernel, product kernel of these two

On each subplot, show the following metrics: within-block mean similarity (for each of the 2 blocks), mean cross-similarity between success and failure

## Implementation
Implement as a new script in my_experiments called kernel_matrix_plots.py
Make the most relevant parameters to the script cli args.
I don't care about implementation details (as long as you implement the kernels correctly) - you are free to make low-level design
decision that don't impact the results significantly, *as long as* the script executes in < 30s.