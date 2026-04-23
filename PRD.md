# Product Requirements Document

## SportsModeling MLB Rebuild

Status: Active rebuild  
Audience: Product, modeling, and engineering  
Last updated: 2026-04-20

## 1. Purpose

This repository is an MLB actuarial betting platform centered on statistically principled GLM-family modeling, penalized regression, lasso credibility, and evidence-backed ensemble promotion.

The two governing statistical sources are:

1. `statistical_theory/09_GLM_Generalized_Linear_Models_for_Insurance_Rating.pdf`
2. `statistical_theory/10_Holmes_Casotto_Penalized_Regression_and_Lasso_Credibility_2025_Revision.pdf`

Anything not directly supported by those texts must be labeled as an overlay.

## 2. Product Summary

The target product is a production-grade MLB statistical ensemble betting platform for:

- moneyline
- runline
- totals

It must combine:

- deterministic ingest and model execution
- pregame-legal feature engineering with explicit availability controls
- CAS-monograph-driven GLM and lasso-credibility modeling
- formal validation and diagnostic artifact generation
- immutable historical prediction ledgers
- operator-facing betting, performance, research, and governance surfaces
- committed staging payloads for shipped dashboard views

## 3. Core Product Question

Which MLB bets are worth taking, why, how stable is that conclusion, how does it compare to the market, and what evidence supports promoting the responsible model or ensemble?

## 4. Target User

The primary user is the operator of a local-first actuarial betting system: a technical model owner who ingests MLB data, reviews forecasts, inspects validation evidence, compares candidate models, and promotes or rejects ensembles based on long-run profitability and statistical credibility.

## 5. Product Principles

- MLB-first contract: the repository contract, defaults, docs, and generated manifests should describe MLB as the primary system.
- Theory traceability: major modeling and validation behavior should map back to the governing monographs.
- Overlay clarity: betting, UX, and engineering conveniences must be labeled separately from direct statistical theory.
- Pregame integrity: predictions must reflect true pregame information only and remain immutable after first pitch.
- Deterministic workflows: refresh, validation, and staging generation should be reviewable and reproducible.
- Evidence before promotion: a champion model or ensemble must not be promoted without written comparative evidence.

## 6. In-Scope Capabilities

### 6.1 MLB data foundation

- MLB schedule, game, result, roster, pitcher, weather, and odds ingestion
- raw snapshot caching
- normalized persistence for MLB entities and market lines

### 6.2 Pregame MLB feature store

- explicit `available_as_of` logic
- leakage prevention
- feature registry and feature contract generation
- support for raw, logged, polynomial, binned, hinge, spline, grouped-category, and interaction transforms

### 6.3 CAS-governed model suite

- vanilla GLMs
- ridge, lasso, elastic net
- lasso credibility with market or prior complements
- GLMM, DGLM, GAM, and MARS-supported extension lane
- ensemble promotion lane

### 6.4 Validation and governance

- fit statistics
- residual diagnostics
- calibration and lift surfaces
- multicollinearity diagnostics
- stability diagnostics
- lasso credibility review outputs
- model cards and promotion/rejection reporting

### 6.5 Betting-product overlay

- fair odds conversion
- vig-free market probabilities
- edge and EV calculations
- bet/pass policies
- bankroll replay and stake sizing
- reason-code outputs for bet or pass decisions

### 6.6 Operator dashboard and staging

- live local dashboard
- committed MLB staging payloads
- research and admin views
- validation and diagnostics views

## 7. Direct Theory vs Overlay Boundary

### Direct theory implementation

- GLM distributions, links, offsets, weights, diagnostics, fit measures, and extensions
- penalized regression and lambda-governance logic
- lasso credibility complement and relativity diagnostics

### Theory-compatible engineering support

- deterministic orchestration
- immutable ledgers
- schema constraints
- artifact manifests
- registry/codegen plumbing

### Betting overlay

- market-vig removal for betting decisions
- edge thresholds
- bankroll replay
- stake sizing
- ROI reporting

### Dashboard/reporting layer

- route inventory
- payload formatting
- static staging snapshots
- model card presentation

## 8. Target Architecture

The rebuild should converge toward:

- `configs/mlb.yaml`
- `configs/feature_registry_mlb.yaml`
- `configs/model_feature_map_mlb.yaml`
- `configs/model_feature_guardrails_mlb.yaml`
- `data/raw/mlb/`
- `data/interim/mlb/`
- `data/processed/mlb/`
- `artifacts/validation/mlb/`
- `artifacts/reports/mlb/`
- `docs/mlb/`
- `web/public/staging-data/mlb/`

## 9. Delivery Program

The rebuild is structured in explicit sprints:

1. Sprint 0: theory lock and repo contract reset
2. Sprint 1: MLB data foundation
3. Sprint 2: pregame feature store
4. Sprint 3: vanilla GLM factory
5. Sprint 4: penalized regression and lasso credibility
6. Sprint 5: GLM extensions
7. Sprint 6: validation gold standard
8. Sprint 7: ensemble and betting engine
9. Sprint 8: dashboard and operator workflow
10. Sprint 9: release hardening

Current status note:

- Sprint 0 is complete.
- The implementation has moved into an integrated MLB modeling-core checkpoint spanning Sprint 1 MLB data foundation, Sprint 2 feature-store architecture, Sprint 4 penalized-regression and lasso-credibility execution, Sprint 6 validation/reporting contracts, and Sprint 8 staging/dashboard contract work.
- The repo should not be described as Sprint 0-only scaffolding anymore.

## 10. Release Readiness Standard

The rebuild is ready for release only when:

- the repo contract is MLB-first and internally consistent
- the theory traceability matrix covers the implemented statistical program
- the MLB data and feature contracts are executable
- the validation factory regenerates MLB artifacts deterministically
- the champion ensemble has a written promotion/rejection report
- dashboard and staging surfaces render MLB payloads coherently
- documentation matches the actual implementation
