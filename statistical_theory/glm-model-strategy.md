# GLM Model Strategy

Source: CAS Exam 8, GLM Chapter 4 model strategy.

## Core idea

Build nested models of increasing complexity, compare them on holdout performance, choose the final candidate using both statistical fit and business judgment, then refit the chosen structure on all available data.

## Canonical sequence

1. Start with a simple baseline model.
2. Add predictors in nested steps so each candidate is a controlled extension of the previous one.
3. Compare candidate models on validation or holdout data, not just in-sample fit.
4. Pick the model that balances predictive performance with interpretability, stability, cost, and operational usefulness.
5. Refit that final specification on the full dataset for the production estimate.

## Simple schematic

```text
M1 -> M2 -> M3 -> M4 -> M5+
simple                    complex

holdout error often falls at first, then rises once extra complexity starts overfitting
```

## Why this matters here

This is a useful counterweight to purely opportunistic feature inclusion. It encourages:
- explicit complexity ladders instead of one-shot kitchen-sink fitting
- honest selection on out-of-sample performance
- a documented tradeoff between predictive lift and interpretability
- final refitting only after model structure has been chosen

## Relation to current `glm_ridge` work

We do not appear to have followed this exact textbook sequence when building `glm_ridge`.

What the repo already does that is adjacent:
- uses holdout and walk-forward evaluation extensively
- tracks significance, stability, non-linearity, and multicollinearity diagnostics
- researches feature subsets and width choices under `artifacts/research/`

What is still different from the Exam 8 framing:
- the current process is not organized as a clearly documented nested GLM ladder
- feature growth is driven more by research rankings and guardrails than by sequential nested candidate models
- the final `glm_ridge` selection logic is distributed across research artifacts rather than written down as one theory-first strategy

## Possible implementation path

If we want to adopt this more literally for `glm_ridge`, a practical version would be:

1. Define ordered feature blocks for each league.
   Example: baseline ratings -> recent form -> roster/injury context -> schedule/travel -> market/context features.
2. Fit nested `glm_ridge` candidates by block.
3. Score each candidate on the same holdout or walk-forward slices.
4. Choose the preferred block depth using both metrics and modeling judgment.
5. Refit the selected block structure on all training data.
6. Save the decision rationale next to the resulting artifact.

## Candidate block example

```text
M1: intercept + baseline strength
M2: M1 + recent form
M3: M2 + lineup / goalie / availability context
M4: M3 + rest / travel / scheduling
M5: M4 + interaction or specialty context features
```

## Working takeaway

For future GLM revisions, it would be reasonable to treat "nested block expansion + holdout comparison + final refit" as the default theory unless we have a strong reason to use a different search strategy.
