# Separability Experiment

Our methods, defined in [methods.md](../methods.md), don't seem able to meaningfully improve scores on the FIPER benchmark, even after extensive tinkering. Let's try to determine how intrinsically hard this benchmark is.

To do so, let's see how well an "oracle" method does that has access to the success/failure labels. So, the oracle is a binary classifier / two-sample test that has access to
- FIPER's train data (which is the same as calibration data in FIPER): treat these as the "nominal" class (let's ignore the fact that Push T's train data is contaminated by failures)
- FIPER's test data: a mix of success and fail episodes, with labels

We want to answer the questions
- **(Q1) How similar is nominal data to test success episodes?**
- **(Q2) How similar is nominal data to test failure episodes?**

If there's no significant difference between these, especially near episode ends, then the corresponding one-class problem which is the FIPER benchmark is intrinsically hard.

We also want to answer the question
- **(Q3) Do the action channels of the data help to separate the nominal data from failure episodes?**

We will break down the answers along the following axes:
- task, so that we see whether the answers to our questions depend significantly on the task
- time step, since, presumably, success and failure episodes start off similarly and only diverge later on

## Experimental Details

### Similarity score

What is a good similarity / dissimilarity score for this experiment?
- Let's start with AUROC from a random forest model
- Use $k$-fold CV to obtain this AUROC (you can choose $k$ appropriate to the size of the data)
- CV should be done over episodes

### Which features / inputs should our separability model have access to?

We consider two cases.
- First, we answer Q1 and Q2 using only the observation embeddings as input
- Second, we answer Q3 by using the (task-space coordinates of) action chunks as inputs too. 
    - To this end, we compute the similarity scores using observation embeddings *and* action chunks.
    - For simplicity, combine these features by concatenation; to answer Q3, we restrict ourselves to the Stacking, Pretzel, and Sorting tasks, which have small ($B <= 32$) batches
    - Both feature sets are scored in the same run, on the same subsampled steps and the same CV folds, so the two AUROCs differ only in the features.
    - The quantity of interest is the change in the failure-minus-success gap, not the change in the failure AUROC alone — the success curve controls for calibration-vs-test shift that has nothing to do with failure. 
    
From these scores we subtract the observation-only scores obtained in the previous case, in order to estimate the benefit of the action channels.

### What exactly is our model trained on? 

At normalized time $t$, our separability model trains on a pool of all steps from the appropriate episodes up to this normalized time step
    - (episode length is a confound for success / failure, since failed episodes tend to be significantly longer - but this need not concern us here since our similarity model trains on pooled steps instead of whole episodes)

## Deliverable

Two plots, each saved as png and pdf, each with 5 subplots stacked vertically,
one per task, x-axis = normalized episode time. Tasks out of scope for a plot get
an empty annotated subplot, so both plots keep the same 5-panel layout.

**Plot 1 — observation-only (Q1, Q2).** y-axis = separability score. Two lines per
subplot: test success (green), test failure (red).

**Plot 2 — action channels (Q3).** y-axis = the difference-in-differences,
(failure - success | obs+act) - (failure - success | obs). ONE line per subplot:
Q3's estimand is a single number per time step, so do not plot the two comparisons
separately. y = 0 is the observation-only baseline.

Both plots show uncertainty. On plot 2 the uncertainty must be on the DiD itself:
form the DiD within each CV fold, then take a confidence interval over folds. Do
not build it from the spreads of the four AUROCs separately.

## Instructions

- Implement this experiment as a new script in my_experiments
- Do not modify existing files
- Prioritize making the script cheap to run

## Checks
- Visually confirm the layout of the plot (all labels should be visible)
- The script should print to the command line shapes and summary stats of all relevant arrays so that I can check
