# Separability Experiment

Our methods, defined in [methods.md](../methods.md), don't seem able to meaningfully improve scores on the FIPER benchmark, even after extensive tinkering. Let's try to determine how intrinsically hard this benchmark is.

To do so, let's see how well an "oracle" method does that has access to the success/failure labels. So, the oracle is a binary classifier / two-sample test that has access to
- FIPER's train data (which is the same as calibration data in FIPER): treat these as the "nominal" class (let's ignore the fact that Push T's train data is contaminated by failures)
- FIPER's test data: a mix of success and fail episodes, with labels

We want to answer the questions
- **how similar is nominal data to test success episodes?**
- **how similar is nominal data to test failure episodes?**

If there's no significant difference between these, especially near episode ends, then the corresponding one-class problem which is the FIPER benchmark is intrinsically hard.

We will break down the answers along the following axes:
- task, so that we see whether the answers to our questions depend significantly on the task
- time step, since, presumably, success and failure episodes start off similarly and only diverge later on

## Deliverable

- A plot (png and pdf) with 5 subplots stacked on top of one another
- One subplot per task
- Each subplot has x-axis = time (normalized per episodes), y-axis = our separability score (see below)
- Each subplots has two line plots: one for test success episodes (green), one for test failure episodes (red) [*NB*: train / test here means FIPER's own train / test terminology, not the train / test for our separability model]

## Experimental Details

- Which features / inputs should our separability model have access to? For now, observation embeddings only
- What exactly is our model trained on? At normalized time $t$, our separability model trains on a pool of all steps from the appropriate episodes up to this normalized time step
    - (episode length is a confound for success / failure, since failed episodes tend to be significantly longer - but this need not concern us here since our similarity model trains on pooled steps instead of whole episodes)

## Similarity score

What is a good similarity / dissimilarity score for this experiment?
- Let's start with AUROC from a random forest model
- Use k-fold CV to obtain this AUROC (you can choose k appropriate to the size of the data)
- CV should be done over episodes

## Instructions

- Implement this experiment as a new script in my_experiments
- Do not modify existing files
- Prioritize making the script cheap to run

## Checks
- Visually confirm the layout of the plot (all labels should be visible)
- The script should print to the command line shapes and summary stats of all relevant arrays so that I can check
