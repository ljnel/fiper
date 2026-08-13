## Efficiency Experiment

In addition to the existing FIPER methods, in [method.md](../methods.md) we defined some kernel-based methods.

We now want to answer the questions

**(Q1)** How computationally expensive are the methods?
**(Q2)** How data-efficient are the methods?

Let's answer these questions, on all tasks, for the following methods:
- all methods from the FIPER paper
- among my methods, only _obs_ and _flat_

## Q1

Some kind of basic profiling experiment, perhaps broken down by train and inference costs.

## Q2

Perhaps some kind of score vs # train episodes curve.

## Deliverables

- One plot for each of the questions
- The plots should be appropriate for a single-column paper: right layout and information density


## Instructions
- Carry out this experiment in my_experiments/
- Don't modify existing files
- For now, try with 1 seed only (multiple seeds are too expensive)