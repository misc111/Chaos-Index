# How To Validate Logistic Regression

Based on `09_GLM_Generalized_Linear_Models_for_Insurance_Rating.pdf`, especially Sections 4.3, 7.2, and 7.3.

## High-level idea

The text treats logistic-regression validation as an out-of-sample exercise. The model should be built on training data, refined with validation data if available, and judged on a holdout test set that is protected until the end. For event models, the key question is not just whether predicted probabilities look reasonable in sample, but whether they separate positives from negatives on unseen data in a way that is useful for decisions.

## 1. Start with the data split

The PDF recommends validating on holdout data, not on the same data used for fitting.

- `train/test` split:
  Use the training set for the full model-building process, then compare final candidate models on the test set.
- `train/validation/test` split:
  If data volume allows, use validation data during refinement and keep the test set untouched until the end.
- Out-of-time validation is preferred when time dependence matters:
  If future conditions can differ from the past, splitting by time is more honest than random splitting.

The text also makes two strong cautions:

- do not overuse the test set, or it stops being a real holdout
- if you choose a final model, refit that chosen structure on all available data afterward to get more credible parameter estimates

## 2. Use quantile plots on predicted probabilities

The book says that many of the lift-style diagnostics used for rating models can also be used for logistic regression.

For a logistic model:

1. Score the holdout set with predicted event probabilities.
2. Bucket records into quantiles by predicted probability.
3. For each bucket, compare:
   - the average predicted probability
   - the actual event rate

According to the text, a good validation plot has three desirable properties:

- accuracy:
  The average predicted probability should be close to the actual event rate in each bucket.
- monotonicity:
  Buckets with higher predicted probabilities should generally have higher actual event rates.
- vertical distance:
  The first and last buckets should be meaningfully separated, showing that the model distinguishes low-risk from high-risk cases.

This is essentially a ranking-plus-calibration check on holdout data.

## 3. Use Lorenz curves and the Gini index

The PDF also adapts lift measurement to logistic regression.

Procedure:

1. Sort holdout records by predicted probability.
2. Plot cumulative share of records on the x-axis.
3. Plot cumulative share of actual event occurrences on the y-axis.
4. Compute the Gini index from the area between the Lorenz curve and the line of equality.

Interpretation:

- a larger Gini means the model is better at separating likely events from unlikely events
- Gini is about ranking power, not about profitability or decision cost by itself

In the book’s framing, this is another way to assess how well the model differentiates the best and worst risks.

## 4. Use ROC curves for threshold-based decisions

This is the main logistic-specific validation tool emphasized by the text.

Logistic regression predicts a probability, but many real uses require a binary action. That means choosing a discrimination threshold, such as:

- investigate if fraud probability is above 50%
- do not investigate otherwise

Once a threshold is chosen, each holdout record falls into one of four groups:

- true positive
- false positive
- false negative
- true negative

From that confusion matrix, the text highlights:

- sensitivity or true positive rate:
  `TP / (TP + FN)`
- specificity:
  `TN / (TN + FP)`
- false positive rate:
  `1 - specificity`

The threshold creates a tradeoff:

- lower threshold:
  more true positives, fewer false negatives, but more false positives
- higher threshold:
  fewer false positives, but more missed events

The ROC curve summarizes that tradeoff by plotting:

- x-axis: false positive rate
- y-axis: true positive rate

for thresholds across the full range from 0 to 1.

Interpretation from the text:

- a useless model lies near the line of equality
- a better model bows upward, delivering higher hit rates for the same false positive cost
- the threshold choice is usually a business decision, not a purely statistical one

## 5. Summarize ROC performance with AUROC

The area under the ROC curve, AUROC, gives a one-number summary of discrimination quality.

- `0.500` means no predictive power beyond random ranking
- `1.000` means perfect separation
- higher AUROC means better discrimination

The text makes an important caveat:

- AUROC and Gini are directly related
- they should not be treated as independent pieces of evidence

So if both are reported, they are really telling the same ranking story in two different languages.

## 6. Be careful with cross-validation

The PDF is cautious about cross-validation for GLMs when variable selection is hand-guided.

Why:

- if variable choice or transformation decisions were influenced by the full dataset, then a later fold is not truly unseen
- proper cross-validation would need to repeat the entire model-building process inside each fold, not just refit fixed variables

The text therefore prefers a fixed holdout split for most insurance-style GLM work, especially when modeling decisions involve judgment. Cross-validation can still help tune limited design choices inside the training set, but the final validation should still use a distinct test set held back until the end.

## Practical summary

Based on this text, validating logistic regression means:

1. protect a real holdout set
2. compare predicted probabilities to actual event rates by quantile
3. evaluate ranking power with Lorenz/Gini
4. evaluate threshold tradeoffs with ROC and confusion matrices
5. use AUROC as a compact summary of discrimination
6. choose action thresholds with business costs in mind
7. refit the chosen model structure on all data only after validation is complete

## What the book emphasizes most

The central theme is that logistic regression should be judged on unseen data by how well it:

- ranks observations from low event risk to high event risk
- matches observed event frequencies across probability buckets
- supports operational decisions under a false-positive / false-negative tradeoff

That is the validation framework the PDF appears to recommend.
