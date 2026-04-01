# NBA Profit Tournament Design

## Context

This design covers a promotion-ready NBA model search intended to increase betting profits beyond the current strategy results through March 29, 2026:

- `capitalPreservation`: `+$31.65` on `$1,550` risked, `21-10`, `2.04% ROI`
- `riskAdjusted`: `+$126.71` on `$5,100` risked, `23-11`, `2.48% ROI`
- `aggressive`: `+$126.71` on `$5,100` risked, `23-11`, `2.48% ROI`

The user explicitly wants a very broad exploration, including nonlinear candidate families, with betting profit as the primary optimization target. The result must be promotion-ready, not just a research report.

## Goal

Build a profit-first NBA challenger tournament that explores broad linear and nonlinear model families, tunes regularization much more thoroughly than the current shallow grids, and can promote a winning challenger into the real training and reporting pipeline if it clears profit, stability, and operational gates.

## Non-Goals

- This design does not change NHL or NCAAM production behavior.
- This design does not expand the user-facing product to leagues outside NHL, NBA, and NCAAM.
- This design does not require immediate hard-refresh publication. It focuses on the NBA research and promotion path first.
- This design does not optimize primarily for log loss, Brier, or calibration. Those remain guardrails, not the north-star objective.

## User-Approved Constraints

- League: NBA
- Objective: maximize betting profit first
- Search breadth: very broad, including nonlinear candidates
- Candidate families must include `gam_spline`, `mars_hinge`, `glmm_logit`, and `dglm_margin`
- Outcome must be promotion-ready
- Search budget: half-day serious, not full-day exhaustive

## Problem Statement

The repo already has:

- a research-only candidate model comparison lane
- a research backtest lane
- structured GLM feature-slate support
- an NBA-first research desk flow with auto-promotion

But the current search is still limited in two important ways:

1. Penalized GLM tuning is shallow. The production and research paths both rely on small hand-picked `C` grids, which means the effective lambda search is sparse and easy to miss profitable shrinkage levels.
2. Nonlinear challengers are not truly promotion-ready. They exist in research, but the trainable model registry, fit runner, predict runner, and research-desk materialization gate still favor current production families.

The net effect is that the repo can discover interesting nonlinear or differently regularized models, but it cannot yet run a broad, disciplined, profit-first tournament and then carry the winner cleanly into production.

## Design Overview

The system will become a staged challenger tournament:

1. Generate a broad candidate field across linear and nonlinear families.
2. Tune each candidate inside the training window.
3. Score each candidate on out-of-sample walk-forward betting results.
4. Rank candidates primarily by profit and bankroll growth.
5. Apply stability and operational gates.
6. Promote only a candidate that both beats the incumbent and can materialize in the real train/validate/reporting path.

In short:

`broad candidate generation -> profit-first walk-forward tournament -> incumbent challenge gate -> promotion-ready winner`

## Candidate Field

### Linear Lane

The linear lane will include:

- `glm_ridge`
- `glm_lasso`
- `glm_elastic_net`
- `glm_vanilla`

These models will be tested across multiple structured feature slates and width variants rather than a single production feature map. The design should support:

- the existing structured NBA GLM spec mechanism
- multiple named slates
- multiple widths per slate
- production-map and screened-feature fallback modes when the requested search mode calls for them

The purpose of the linear lane is not just to tune coefficients. It is to answer whether current regularization and feature-pruning choices are leaving profit on the table.

### Nonlinear Lane

The nonlinear lane will include:

- `gam_spline`
- `mars_hinge`
- `glmm_logit`
- `dglm_margin`

These families are already implemented in the research layer and should enter the same tournament rather than living in a separate comparison flow. Each one will use a controlled complexity grid so the search is broad without becoming a random kitchen sink.

### Incumbent Benchmark

The incumbent benchmark should be the currently active NBA champion from `active_champions` when available, falling back to the current default NBA champion behavior if needed. The tournament must compare challengers against this incumbent on the same windows and the same scoring rules.

## Hyperparameter Strategy

### Lambda-First Tuning

Regularization search should be expressed in true lambda terms and only converted to sklearn's `C = 1 / lambda` at fit time. This keeps tuning interpretable and makes the search dense where it matters.

The artifacts must store:

- searched `lambda`
- derived `C`
- searched `l1_ratio` where relevant
- selected best parameters

### Penalized GLM Search

For the half-day serious budget, the penalized GLM search should move from shallow lists to dense log-scale lambda grids:

- ridge: dense log-scale lambda sweep from very light to very strong shrinkage
- lasso: dense log-scale lambda sweep with more coverage on the strong-shrinkage side
- elastic net: the same lambda sweep plus a denser `l1_ratio` grid than today

The winning parameter set should be chosen by profit-first walk-forward results, not only mean log loss.

### Nonlinear Complexity Search

Each nonlinear family gets a bounded complexity grid:

- `gam_spline`: feature caps, knot counts, and regularization strength
- `mars_hinge`: feature caps, hinge density, interaction degree, and regularization strength
- `glmm_logit`: fixed-effect feature caps, with runtime and convergence captured explicitly
- `dglm_margin`: feature caps and iteration settings

This design intentionally keeps the search broad but disciplined. Families can differ in complexity, but all of them are evaluated under the same downstream profit criterion.

## Search Orchestration

### Tournament Stages

The tournament should run in stages:

1. Build the NBA research feature set from the configured historical seasons.
2. Resolve candidate feature views:
   - production model map
   - screened research pool
   - structured GLM slates and widths
3. Fit and tune candidate configurations on training windows.
4. Evaluate candidates on walk-forward out-of-sample betting results.
5. Build a scorecard with profit-first ordering.
6. Challenge the incumbent.
7. Materialize and promote the winner only if it clears all gates.

### Half-Day Budget Guardrails

Because the user chose the half-day serious budget, the orchestration should:

- cover all agreed model families
- use denser lambda search than current behavior
- avoid unbounded exploratory feature generation
- include enough repeatability checks to reject obvious one-window luck
- avoid a full exhaustive combinatorial explosion across every family and every width

## Promotion Policy

### Primary Ranking

The tournament ranks challengers primarily by betting outcomes:

- net profit
- ending bankroll
- ROI
- profitable fold count

### Secondary Guardrails

Secondary metrics remain guardrails, not primary objectives:

- max drawdown
- minimum bet count
- fold-level consistency
- calibration degradation versus incumbent
- catastrophic log-loss or ECE regressions

These metrics should stop a brittle winner, but they should not outrank clear profit improvement by default.

### Incumbent Challenge Rule

A challenger must beat the incumbent on the same walk-forward windows. A candidate that looks good in isolation but cannot beat the incumbent under matched evaluation should not promote.

### Materialization Rule

A research winner is not sufficient. The candidate must also:

- exist in the trainable model registry
- fit and predict through the real train pipeline
- serialize and reload cleanly
- generate OOF and reporting artifacts without breaking downstream consumers
- satisfy runtime sanity for daily use

If a model wins in research but fails the real training path or is operationally too slow, the promotion decision must record that explicitly and leave the incumbent active.

## Promotion-Ready Production Changes

To make nonlinear winners real candidates rather than research-only curiosities, the production path must expand in a controlled way.

### Registry

The canonical model registry should gain explicit entries for any challenger families allowed to promote. At minimum, this design assumes support for:

- `glm_vanilla`
- `gam_spline`
- `mars_hinge`
- `glmm_logit`
- `dglm_margin`

### Training And Prediction

The fit and predict runners should support these families the same way they already support the current trainable models:

- canonical model construction
- feature-column resolution
- fit
- predict
- model save/load compatibility

### Research Desk Policy

The research desk should stop blocking nonlinear winners solely because they are outside the current `MATERIALIZABLE_MODEL_NAMES` set. Instead, that gate should be driven by actual production support and runtime policy.

### Reporting And Artifacts

The run payload, validation outputs, and reporting surfaces should recognize newly trainable challengers as normal models, not special cases.

## Data Flow

1. Historical NBA source data is imported or refreshed.
2. Research features are built from the historical dataset.
3. Candidate feature sets are resolved.
4. Candidate models are tuned on training windows.
5. Walk-forward predictions generate model-level betting decisions and fold metrics.
6. A scorecard and promotion decision artifact are written.
7. If promoted, the new champion is persisted through the existing champion tables and the model becomes available to the real train/validate/report flow.

## Artifacts

The search should produce explicit artifacts for auditability:

- tournament configuration summary
- candidate-level parameter grid results
- fold-level profit and metric tables
- incumbent-versus-challenger comparison
- promotion decision summary with explicit gate outcomes
- winning model descriptor, including model family, feature slate, width, lambda, and any nonlinear complexity settings

The promotion summary should explain not only who won, but why promotion succeeded or failed.

## Error Handling

- If a candidate family fails to converge or fit on a fold, record the failure and continue unless the entire family is unusable.
- If a candidate cannot materialize in the production path, mark it as research-successful but promotion-ineligible.
- If the incumbent benchmark cannot be resolved from `active_champions`, fall back to the configured NBA default champion behavior.
- If structured GLM slate resolution fails, the error must remain explicit and must not silently fall back to unrelated features unless the requested mode allows fallback.

## Testing Strategy

Testing should cover both search correctness and promotion readiness.

### Unit And Contract Coverage

- lambda-to-`C` conversion and dense-grid generation
- profit-first ranking and tie-break logic
- new registry entries and alias resolution
- materialization-gate behavior for newly supported candidate families
- research-desk promotion decisions for nonlinear winners

### Service-Level Coverage

- research backtest scorecard generation with expanded candidate families
- production training path with newly supported challenger families
- OOF prediction generation for new trainable candidates
- reporting payload compatibility when new models appear

### Regression Coverage

- existing GLM research flows must still work
- current incumbent promotion flow must still work when the winner is a penalized GLM
- no silent breakage in downstream web/report consumers from new model identifiers

## Risks

### Runtime Risk

`glmm_logit` may be too slow or unstable for routine production retraining. This design accepts that risk but treats runtime as an explicit promotion gate.

### Overfitting Risk

A broad tournament can easily produce lucky winners. That is why the promotion gate emphasizes incumbent comparison, fold consistency, drawdown, and minimum bet count rather than blindly trusting top-line bankroll.

### Product Risk

New model families may change the shape of reporting and model metadata. The design avoids that by extending the existing registry and artifact contracts rather than creating a parallel challenger-only reporting path.

## Recommended Implementation Shape

Implementation should happen in three logical slices:

1. Expand the research tournament:
   - denser lambda search
   - broader structured feature exploration
   - profit-first ranking logic
2. Expand production materialization:
   - registry
   - fit/predict runners
   - artifact compatibility
3. Expand promotion logic:
   - nonlinear challenger eligibility
   - incumbent challenge reporting
   - explicit promotion gate outcomes

## Success Criteria

This design is successful when:

- the NBA research flow can run a broad tournament across both linear and nonlinear families
- penalized GLMs are tuned with materially denser lambda coverage
- the winner is chosen by profit-first walk-forward evidence
- a nonlinear winner can promote if it is truly better and operationally supported
- the incumbent remains active when a challenger wins on paper but fails materialization or stability gates
- the resulting artifacts explain the outcome clearly enough for future review
