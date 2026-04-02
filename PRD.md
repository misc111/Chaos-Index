# Product Requirements Document

## SportsModeling Current Product Baseline

Status: Baseline
Audience: Internal product and engineering
Last updated: 2026-04-02

## 1. Purpose

This document describes the product that already exists in this repository today. It is a baseline PRD, not a future-state roadmap. Its goal is to capture what the system is for, who it serves, what capabilities are already shipped, and which constraints shape product decisions.

The product is a local-first multi-league forecasting and betting-analysis workbench for NHL, NBA, and NCAA men's basketball. It combines:

- deterministic ingest and training pipelines
- league-aware feature engineering and model execution
- profitability-focused evaluation and replay
- a deterministic natural-language query interface
- a live local dashboard backed by SQLite
- a static GitHub Pages staging site backed by committed JSON snapshots

## 2. Product Summary

SportsModeling helps a single operator or a small internal team answer one core question:

Which games are worth betting, why does the model think that, and how has that approach performed over time?

The product is not just a model-training repository. It is an end-to-end forecasting product with persistent history, operator workflows, performance review, validation surfaces, and publishable dashboard outputs.

## 3. Problem Statement

Most consumer betting products show market prices but do not provide:

- an independent probability estimate
- a reproducible local record of pregame predictions
- a deterministic replay of historical bet decisions
- a unified way to inspect validation, calibration, and profitability
- a controlled research-to-promotion loop for model changes

SportsModeling exists to solve that gap for supported leagues by turning raw league and odds data into usable operator decisions and review surfaces.

## 4. Target Users

### Primary user

The primary user is the repo operator: a technical sports-model owner who runs data refreshes, trains models, inspects forecasts, reviews betting performance, and decides whether a model or strategy should be trusted.

### Secondary users

- Internal collaborator reviewing model outputs, validation artifacts, or dashboard pages
- Research-oriented user comparing candidate models and promotion outcomes
- External reviewer of the staged dashboard experience

This is not currently a public self-serve multi-tenant product.

## 5. User Jobs To Be Done

The product should enable a user to:

1. Refresh league data and odds in a deterministic way.
2. Generate forecasts for upcoming games across supported leagues.
3. Compare model probabilities against market-implied views.
4. Identify which games qualify as bets under a defined strategy.
5. Review historical profit/loss, ROI, bankroll path, and weekly bet breakdowns.
6. Inspect validation, calibration, leaderboard, slices, and diagnostics outputs.
7. Ask natural-language questions about forecasts, betting history, team outlooks, and model performance.
8. Compare candidate models and manage promotion decisions, especially in the NBA-first research workflow.
9. Publish a stable staging snapshot that mirrors shipped dashboard payloads without requiring live database access.

## 6. Product Principles

- Local-first: SQLite and local artifacts are the system of record for day-to-day operation.
- Deterministic: refresh, query, and publish flows should produce inspectable, repeatable results.
- Operator-centric: product surfaces are optimized for the model owner rather than a casual fan audience.
- Profitability-first: unless explicitly discussing statistical quality, model performance is interpreted through betting outcomes.
- Cross-league where practical: NHL, NBA, and NCAAM should share architecture while allowing league-specific adapters and policies.
- Historical integrity matters: frozen pregame predictions must remain distinguishable from synthetic replay or diagnostic outputs.

## 7. In-Scope Product Capabilities

### 7.1 Multi-league forecasting platform

The product supports NHL, NBA, and NCAAM forecasting under a shared architecture with league-specific data adapters, feature strategies, and configuration.

Expected outcome:

- each supported league can be fetched, trained, scored, queried, and displayed
- repo-wide refresh flows can execute all leagues in a fixed order

### 7.2 Deterministic pipeline execution

The product includes operational commands for:

- database initialization
- fetch
- odds refresh
- feature generation
- training
- validation
- backtesting
- daily runs
- all-league data refresh
- all-league hard refresh

Expected outcome:

- the operator can run atomic league-scoped commands or deterministic multi-league orchestration
- hard refresh performs a full refresh/train/publish flow without rebuilding features

### 7.3 Persistent forecasting and evaluation layer

The product stores forecasts, results, model runs, scores, validation outputs, odds snapshots, and related metadata in SQLite plus file-based artifacts.

Expected outcome:

- the system can answer live and historical product questions from persisted state
- training and replay do not depend on fragile in-memory workflows

### 7.4 Deterministic query interface

The product includes a natural-language local query surface for:

- betting history questions
- team forecast questions
- model leaderboard and performance questions
- championship probability heuristics
- report-style team summaries

Expected outcome:

- the operator can ask product questions in plain language and receive deterministic answers over local persisted data

### 7.5 Live dashboard

The Next.js dashboard is a core product surface, not just a demo. It provides league-aware pages for:

- overview
- predictions
- games today
- market board
- performance
- bet history
- bet sizing
- validation
- calibration
- diagnostics
- slices
- leaderboard
- actual vs expected
- research desk
- research admin

Expected outcome:

- the operator can inspect both current slate decisions and historical quality/performance from a single UI

### 7.6 Static staging publish

The product ships a second delivery target: a static GitHub Pages staging site generated from committed JSON snapshots in `web/public/staging-data/`.

Expected outcome:

- shipped staging views mirror committed dashboard payloads
- the staging site does not require live SQLite access
- dashboard-affecting payload changes can be published through snapshot regeneration and git push

### 7.7 Research and promotion workflow

The product includes a research layer for comparing candidate models, tracking runs, and exposing promotion outcomes. The current research desk experience is intentionally NBA-first, while research admin is a live local control room.

Expected outcome:

- candidate models can be evaluated and compared against current champions
- promotion history and gating outcomes can be inspected
- research surfaces stay separate from public staging when appropriate

## 8. Core User Experience

### 8.1 Daily operator flow

1. Refresh data and odds.
2. Train or update models.
3. Open the local dashboard.
4. Review `Games Today`, `Predictions`, and `Market Board`.
5. Decide which plays qualify under the active strategy.
6. Revisit `Bet History` and `Performance` to judge whether the system is improving.

### 8.2 Analysis flow

1. Ask a question through `make query Q="..."`.
2. Inspect answer text and payload derived from local state.
3. Use dashboard pages for deeper visual review when needed.

### 8.3 Research flow

1. Run candidate comparison or research workflow.
2. Inspect research desk and research admin surfaces.
3. Review whether a candidate should be promoted or rejected.
4. Preserve the decision trail in persistent outputs.

### 8.4 Publish flow

1. Regenerate staging snapshots from local dashboard data.
2. Commit the updated staged JSON.
3. Push `main`.
4. Let GitHub Pages publish the committed snapshot-based site.

## 9. Functional Requirements

### 9.1 League support

- The system must support NHL, NBA, and NCAAM.
- Ambiguous product questions should default to NBA when no stronger context exists.
- Cross-league failure modes should be investigated across all supported leagues before work is considered complete.

### 9.2 Forecast integrity

- The system must preserve an immutable pregame prediction ledger.
- Diagnostic, replay, and backtest outputs must remain separable from true historical pregame records.

### 9.3 Profitability review

- The product must support net profit/loss, ROI, bankroll path, amount risked, record, and game-by-game replay analysis.
- Betting-history questions should use the deterministic query path first.

### 9.4 Market comparison

- The dashboard must expose model probability vs. market-implied views.
- Users must be able to see whether a game is a bet or a pass under the active strategy and why.

### 9.5 Validation and diagnostics

- The system must surface leaderboard, validation, calibration, diagnostics, slices, and related evaluation artifacts.
- The operator must be able to inspect both statistical quality and betting-performance outcomes.

### 9.6 Research governance

- Candidate model comparison and promotion decisions must be represented as product surfaces rather than hidden ad hoc scripts.
- Research admin must remain a live local surface and not be exposed in static staging.

### 9.7 Publish parity

- If a dashboard or payload change affects the shipped staging experience, staging JSON must also be regenerated and committed.
- Static staging should be treated as a separate delivery target from the live local dashboard.

## 10. Non-Functional Requirements

- Determinism: refresh and query workflows should favor fixed ordering and reproducible outputs.
- Fail-fast behavior: composite refresh flows should stop on required-step failures rather than silently skipping work.
- Local operability: the full product should remain useful from a local machine without cloud infrastructure.
- Clear source of truth: code-first registries and SQLite-backed persisted outputs should remain authoritative.
- Maintainability: league-specific behavior should live in adapters, strategies, and policies rather than spreading through shared modules.
- Reviewability: generated docs, manifests, staging snapshots, and validation artifacts should make the system easier to inspect.

## 11. Success Criteria

Because this is a baseline PRD, these are product-health measures rather than new launch goals:

- The operator can reliably produce fresh forecasts for all supported leagues.
- The operator can answer betting-history and forecast questions from local persisted data without manual SQL spelunking.
- The dashboard provides a coherent path from current slate review to historical performance review.
- The staging site reflects committed product outputs rather than diverging from local dashboard payloads.
- Research and promotion workflows are inspectable enough to explain why a model is active.
- The system remains aligned with the repo's north star: long-run betting profitability through mispricing detection and disciplined bet selection.

## 12. Non-Goals

The current product baseline does not aim to be:

- a public sportsbook
- a real-time live-betting engine
- a generalized sports platform beyond NHL, NBA, and NCAAM
- a cloud-native multi-user SaaS application
- a fully automated no-operator system
- a product where GitHub Pages staging is the live source of truth

## 13. Constraints And Assumptions

- SQLite is the canonical local persistence layer.
- The product is operated from this repository and its make/CLI workflows.
- Staging is snapshot-based and must be regenerated explicitly.
- Hard refreshes intentionally reuse current processed feature snapshots instead of rebuilding features.
- Betting performance is the default lens for model-performance questions unless the user asks for statistical metrics.
- NBA currently receives the most advanced research-desk product treatment.

## 14. Known Product Risks

- The live dashboard and static staging site can drift if staging snapshots are not regenerated alongside payload changes.
- Cross-league consistency can erode if shared fixes are not checked against all supported leagues.
- The product mixes production, validation, and research surfaces closely enough that boundaries must stay intentional.
- Betting-performance interpretation can be misleading if odds coverage, replay coverage, or frozen-prediction integrity are incomplete.
- NBA-first research surfaces create an uneven product experience across leagues.

## 15. Open Questions

These questions are intentionally left open because they are product-direction questions, not baseline facts:

- Should research desk capabilities expand beyond NBA into NHL and NCAAM with equivalent promotion workflows?
- Should the deterministic query system become a first-class dashboard surface rather than staying CLI-first?
- Should the product continue optimizing primarily for a single expert operator, or begin supporting broader collaborator workflows?
- Which dashboard pages are core decision surfaces versus valuable but secondary review tools?

## 16. Source Anchors In This Repo

This PRD is grounded in the current repository structure and shipped surfaces, especially:

- `README.md`
- `CLAUDE.md`
- `docs/generated/architecture.md`
- `docs/generated/dashboard-routes.md`
- `docs/project-organization-deep-dive.md`
- `src/query/`
- `src/orchestration/`
- `web/app/`
- `web/public/staging-data/`

