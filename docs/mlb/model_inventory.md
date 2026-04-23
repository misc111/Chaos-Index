# MLB Model Inventory

Current-state note: the active default MLB lane is `glm_ridge`, `glm_elastic_net`,
`glm_lasso`, and `glm_vanilla`. Other lanes below are opt-in comparison lanes
or target-program inventory.

## Retired Baseline References

- `intercept_only`
- `market_only`
- `home_field_only`
- `simple_prior_only`

Classification: `Experimental` (retired from the active MLB lane)

## Vanilla GLM Family

- logistic GLM for `home_win`
- logistic GLM for `runline_cover`
- logistic GLM for `totals_over`
- Poisson GLM for `home_runs`
- Poisson GLM for `away_runs`
- overdispersed Poisson GLM for team runs
- negative binomial GLM for team runs
- normal GLM for run differential benchmark
- gamma GLM where strictly positive conditional targets are justified
- inverse Gaussian GLM where tail shape makes it more appropriate
- Tweedie GLM for nonnegative zero-heavy aggregates
- frequency/severity two-stage constructions where superior to a forced single target

Classification: `Direct theory implementation`

## Penalized GLM Family

- ridge GLM
- lasso GLM
- elastic net GLM
- lambda grid and path reporting
- active-parameter summaries

Classification: `Direct theory implementation`

## Lasso Credibility Family

- market-offset lasso credibility
- prior-model-offset lasso credibility
- selected complement-of-credibility variants
- complement / indicated / observed output tables
- relativity plots with exposure overlays

Classification: `Direct theory implementation`

## GLM Extensions

- GLMM
- DGLM
- GAM
- canonical MARS hinge discovery (target-state)
- MARS-discovered hinge reuse back into GLM candidates when validation supports it

Classification: `Theory-compatible extension`

Current-state caveat: the shipped `mars_hinge` implementation is a proxy
challenger and must be labeled `experimental`, not a direct canonical MARS lane.

## Ensemble Lane

- simple average ensemble
- weighted ensemble chosen on holdout evidence
- geometric average where log-link components justify it
- final champion committee with explicit promotion package

Classification: `Theory-compatible engineering support` / betting overlay governance

## Experimental Challenger Lane

These are explicit opt-in comparison models and must remain non-core:

- `mars_hinge` (proxy hinge search)
- `two_stage`
- random forest
- gradient-boosted trees
- neural net MLP
- Bayesian goals/state-space challengers

Classification: `Experimental` (comparison-only, non-champion)

## Candidate Tracking Requirements

Every candidate must record:

- model family
- target
- distribution
- link
- offsets/complements
- feature set identity
- validation scores
- calibration findings
- stability flags
- betting-overlay outcomes
- promotion or rejection reason

Classification: `Theory-compatible engineering support`

## Generated Feature/Beta Matrix

- See [model_feature_beta_matrix.md](</Users/davidiruegas/Library/Application Support/SportsModeling/docs/mlb/model_feature_beta_matrix.md:1>) for the maintained Markdown snapshot of current model feature selection and coefficient output across the production and candidate-comparison lanes.
