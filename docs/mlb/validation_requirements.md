# MLB Validation Requirements

## Purpose

This file defines the minimum validation factory required before the MLB rebuild can claim a production-grade statistical program.

## 1. Data Quality

Required:

- schema integrity
- uniqueness and duplicate prevention
- orphan prevention
- odds normalization checks
- vig-removal correctness
- timestamp integrity
- missingness summaries
- range checks
- categorical allowed-values checks
- raw snapshot reproducibility
- `available_as_of` enforcement
- no postgame fields in model matrices

Classification: `Theory-compatible engineering support`

Current implementation status:

- landed a first MLB-only data-quality validation task
- current checks cover:
  - required MLB table presence
  - duplicate MLB odds-line keys
  - orphan odds-line references to missing games/snapshots
  - snapshot timing after first pitch
  - implied-probability bounds
  - conservative two-way vig-sum sanity checks
- feature leakage checks now explicitly reject `available_as_of_utc` timestamps later than first pitch
- feature leakage checks now explicitly flag MLB postgame outcome tokens such as runs, run differential, and winner labels when they appear in feature columns

The remainder of this section is still required before Sprint 6 acceptance.

## 2. Split Integrity

Required:

- chronological train/validation/test discipline
- no split overlap
- true out-of-time holdout
- rolling or blocked CV
- prequential ordering checks
- first-pitch cutoff enforcement
- odds snapshot precedes prediction checks
- immutable prediction-ledger checks

Classification: `Direct theory implementation` plus `Theory-compatible engineering support`

## 3. GLM Output Requirements

Required for vanilla GLMs:

- coefficients
- standard errors
- p-values
- confidence intervals
- dispersion where applicable
- explicit weights and offsets
- explicit base levels

Required prohibition:

- no p-values for lasso, ridge, elastic net, or lasso credibility

Classification: `Direct theory implementation`

## 4. Multicollinearity And Aliasing

Required:

- pairwise correlation review
- near-duplicate feature detection
- VIF
- condition indices
- aliasing detection
- correlated-feature warnings
- documented fallback to elastic net or reduced feature blocks when justified

Classification: `Direct theory implementation`

## 5. Fit And Comparison

Required:

- log-likelihood
- deviance and scaled deviance where relevant
- nested-model F-tests where valid
- AIC
- BIC
- parameter count
- active parameter count
- lambda path reports
- CV metric reports
- comparative scorecards

Classification: `Direct theory implementation`

## 6. Residual Diagnostics

Required:

- raw residuals where relevant
- deviance residuals
- working residuals
- residual histograms
- Q-Q plots where meaningful
- residual-vs-fitted
- residual-vs-predictor
- grouped residual views by team, pitcher, park, month, market bucket
- influence diagnostics, leverage, Cook's distance, dfbetas where supported

Classification: `Direct theory implementation`

## 7. Lift, Calibration, And Logistic Validation

Required:

- actual-vs-predicted plots
- quantile plots
- simple lift
- double lift
- loss-ratio-style charts adapted to betting buckets
- Lorenz curve
- Gini
- ROC / AUROC
- confusion matrices by threshold
- threshold sensitivity
- calibration by quantile or decile
- Brier score and decomposition where applicable
- per-candidate curve artifacts for tournament candidates, not only the resolved primary model
- target-scoped family champion diagnostics for moneyline, runline, and totals

Classification: `Direct theory implementation`

Current betting-overlay addition:

- validation emits a `market_truth` task when MLB holdout games have persisted moneyline odds snapshots
- the task compares the selected primary model to prediction-time and closing no-vig market probabilities
- emitted outputs include summary, per-model rows, CLV, flat unit ROI, log loss/Brier deltas versus market baselines, and calibration tables
- this is not a CAS theory-core diagnostic; it is the scoreboard for whether a model is gaining betting edge against the market

Classification: `Betting overlay`

## 8. Lasso Credibility Diagnostics

Required:

- lambda review
- complement review
- relativity plots
- complement / indicated / observed overlays
- exposure bars
- ordinal reversal review
- control-variable review
- explicit complement documentation
- explicit lambda-choice documentation, including actuarial override when used

Classification: `Direct theory implementation`

## 9. Stability And Governance

Required:

- coefficient stability across folds
- coefficient stability across time windows
- season-to-season stability
- bootstrap intervals
- refit reproducibility
- feature contract drift checks
- artifact completeness checks
- model card generation
- theory traceability completeness

Classification: `Direct theory implementation` plus `Theory-compatible engineering support`

## 10. Ensemble Validation

Required:

- component error-correlation analysis
- diversity analysis
- weighted vs unweighted comparison
- true-holdout champion selection
- written promotion/rejection report

Classification: `Direct theory implementation` for comparison logic, `Theory-compatible engineering support` for reporting/governance
