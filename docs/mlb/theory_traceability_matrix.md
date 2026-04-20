# MLB Theory Traceability Matrix

## Purpose

This matrix maps the MLB rebuild to the two governing CAS monographs:

1. `09_GLM_Generalized_Linear_Models_for_Insurance_Rating.pdf`
2. `10_Holmes_Casotto_Penalized_Regression_and_Lasso_Credibility_2025_Revision.pdf`

Every row is tagged as one of:

- `Direct theory implementation`
- `Theory-compatible engineering support`
- `Betting overlay`
- `Dashboard/reporting layer`

## Matrix

| Area | MLB implementation target | Source trace | Classification |
| --- | --- | --- | --- |
| GLM core structure | Model outcome targets with exponential-family random component plus linear predictor on the proper link scale | GLM monograph pp. 2-7 | Direct theory implementation |
| Binary moneyline / runline / totals side models | Bernoulli/binomial with logit link for home win, runline cover, totals over | GLM monograph pp. 25-26; lasso monograph pp. 3-6 | Direct theory implementation |
| Team-runs models | Poisson GLM for home runs and away runs | GLM monograph pp. 21-22 | Direct theory implementation |
| Overdispersed count lane | Overdispersed Poisson / negative binomial lane for team runs when Poisson variance is inadequate | GLM monograph pp. 21-22 | Direct theory implementation |
| Positive continuous target lane | Gamma and inverse Gaussian for positive conditional targets such as positive-run components or severity-style overlays | GLM monograph pp. 19-21 | Direct theory implementation |
| Zero-heavy nonnegative lane | Tweedie for nonnegative zero-heavy aggregates where a compound Poisson-gamma process is reasonable | GLM monograph pp. 22-24 | Direct theory implementation |
| Normal benchmark lane | Normal GLM for run-differential benchmark comparisons | GLM monograph pp. 7, 19 | Direct theory implementation |
| Frequency/severity decomposition | Two-stage construction where appropriate instead of forcing a single target | GLM monograph pp. 43-45 | Direct theory implementation |
| Weights | Exposure-aware or aggregation-aware weighting through variance, not ad hoc sample duplication | GLM monograph pp. 16-18 | Direct theory implementation |
| Offsets | Market or prior complements transformed onto the linear-predictor scale | GLM monograph pp. 17-18; lasso monograph pp. 38-45 | Direct theory implementation |
| Market complement lane | Vig-free implied market probabilities converted to logit-scale offsets for credibility-aware models | Lasso monograph pp. 38-45 | Direct theory implementation |
| Prior-model complement lane | Prior-model offsets for complement-of-credibility modeling | Lasso monograph pp. 41-45, 99-104 | Direct theory implementation |
| Continuous transforms | Raw, log, polynomial, bin, hinge, and spline transforms driven by partial-residual and validation evidence | GLM monograph pp. 48-55 | Direct theory implementation |
| Categorical handling | One-hot/base-level treatment, grouped levels, populous base-level selection | GLM monograph pp. 12-15, 55-56 | Direct theory implementation |
| Ordinal treatment | Prefer ordinal encodings for lasso credibility where sensible so complements and shrinkage remain reviewable | Lasso monograph pp. 24-29, 39-41, 49-52 | Direct theory implementation |
| Interactions | Categorical x categorical, categorical x continuous, continuous x continuous | GLM monograph pp. 55-61 | Direct theory implementation |
| Multicollinearity review | Pairwise correlation, VIF, condition indices, aliasing review | GLM monograph pp. 27-28 | Direct theory implementation |
| Vanilla GLM inferential outputs | Standard errors, p-values, and confidence intervals for unpenalized GLMs only | GLM monograph pp. 8-9 | Direct theory implementation |
| Penalized regression family | Ridge, lasso, elastic net, lambda path, active-parameter summaries | GLM monograph pp. 101-103; lasso monograph pp. 14-24 | Direct theory implementation |
| Lasso variable policy | Sparse coefficient policy, implicit materiality threshold, conservative lambda selection | Lasso monograph pp. 21-24, 42-43 | Direct theory implementation |
| Lasso credibility | Complement via offset, shrinkage toward complement instead of zero, complement/indicated/observed review | Lasso monograph pp. 38-52 | Direct theory implementation |
| Lasso credibility diagnostics | Lambda review, complement review, relativity plots, exposure overlays, ordinal reversal checks | Lasso monograph pp. 46-52 | Direct theory implementation |
| Model fit measures | Log-likelihood, deviance, scaled deviance, nested-model F-tests where appropriate, AIC, BIC | GLM monograph pp. 62-67 | Direct theory implementation |
| Residual diagnostics | Raw, deviance, and working residuals; residual-vs-fitted; residual-vs-predictor; grouped residual views | GLM monograph pp. 67-74 | Direct theory implementation |
| Influence review | Cook's distance and influence-oriented outlier review | GLM monograph pp. 73-74 | Direct theory implementation |
| Lift and validation plots | Actual-vs-predicted, quantile plots, double lift, loss-ratio-style charts, Lorenz, Gini, ROC, AUROC | GLM monograph pp. 75-82 | Direct theory implementation |
| Split governance | Train/test, train/validation/test, cross-validation, holdout discipline | GLM monograph pp. 38-41 | Direct theory implementation |
| Stability governance | Bootstrap/refit stability, coefficient stability over time and across folds, rebuild governance | GLM monograph pp. 73-74, 86-88; lasso monograph pp. 21-24, 83-85 | Direct theory implementation |
| GLMM lane | Random-effects shrinkage for sparse grouping structures | GLM monograph pp. 93-96 | Direct theory implementation |
| DGLM lane | Separate dispersion model when fixed-dispersion GLMs are inadequate | GLM monograph pp. 96-98 | Direct theory implementation |
| GAM lane | Smooth nonlinearity with controlled flexibility | GLM monograph pp. 98-100 | Direct theory implementation |
| MARS lane | Hinge-term discovery and interaction mining, reusable into GLM where justified | GLM monograph pp. 100-101 | Direct theory implementation |
| Ensemble guidance | Compare simple and weighted ensembles only after model-level validation; preserve interpretability and governance | GLM monograph pp. 91-92 | Direct theory implementation |
| Immutable pregame ledger | Freeze pregame records by as-of timestamp and never overwrite after first pitch | Supports holdout integrity and model-governance discipline; not a monograph statistical primitive | Theory-compatible engineering support |
| `available_as_of` feature gating | Enforce pregame legality for every feature at build time | Supports monograph data-prep and validation discipline | Theory-compatible engineering support |
| MLB schema and manifests | Normalize games, odds, feature sets, model runs, predictions, validation outputs, theory trace | Supports reproducibility and documentation expectations | Theory-compatible engineering support |
| Fair odds, edge, EV, stake sizing | Betting decision layer on top of theory-derived probabilities | Not prescribed by the monographs | Betting overlay |
| Bet/pass reason codes | Operator-facing narrative of why a play was taken or skipped | Not prescribed by the monographs | Betting overlay |
| Dashboard pages, staging payloads, model cards | Render and ship operator-facing views of model and betting outputs | Not prescribed by the monographs | Dashboard/reporting layer |

## Key Warnings From The Monographs

- Do not report p-values for lasso, elastic net, or lasso-credibility models.
- Do not let holdout data leak into lambda selection or feature selection.
- Do not use continuous-variable lasso credibility casually; prefer categorical or ordinal treatment where possible.
- Do not let flexibility from hinges, splines, GAMs, or MARS outrun holdout governance.
- Do not treat a weak complement as harmless. Lasso credibility only works when the complement is meaningful.
