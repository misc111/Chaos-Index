# MLB Data Contract

## Scope

The MLB rebuild requires normalized persistence for schedule, results, odds, features, model runs, predictions, and validation outputs.

## Core Storage Concepts

Minimum MLB entities:

- `mlb_games`
- `mlb_results`
- `mlb_odds_snapshots`
- `mlb_odds_market_lines`
- `mlb_feature_sets`
- `mlb_model_runs`
- `mlb_model_predictions`
- `mlb_validation_results`
- `mlb_backtest_bets`
- `mlb_performance_aggregates`
- `mlb_theory_trace`
- `mlb_prediction_components`

Classification: `Theory-compatible engineering support`

## Current Sprint Status

Implemented in the current Sprint 1 slice:

- physical MLB ingest tables:
  - `mlb_games`
  - `mlb_results`
  - `mlb_odds_snapshots`
  - `mlb_odds_market_lines`
- physical MLB governance tables:
  - `mlb_theory_trace`
  - `mlb_prediction_components`
- MLB contract views over the existing shared ledger for:
  - `mlb_feature_sets`
  - `mlb_model_runs`
  - `mlb_model_predictions`
  - `mlb_validation_results`
  - `mlb_backtest_bets`
  - `mlb_performance_aggregates`

This is an intentional migration step: MLB now has named contract surfaces in the schema, while the remaining feature/model/validation write paths can migrate onto them in later sprints without breaking the current repository machinery.

## Minimum Prediction Row Contract

Every prediction row must preserve:

- game identity
- as-of timestamp
- scheduled first-pitch time
- snapshot identity
- model version
- feature set identity
- market context
- per-model component predictions
- final ensemble prediction
- fair odds
- market vig-free implied probabilities
- edge
- bet/pass decision
- stake recommendation
- reason codes

Classification: `Theory-compatible engineering support` plus `Betting overlay`

## Pregame Immutability

- True pregame predictions must never be overwritten after first pitch.
- Replay, OOF, and diagnostic outputs must remain separate from frozen historical pregame records.

Classification: `Theory-compatible engineering support`

## Raw Snapshot Requirements

- cache raw source payloads
- preserve extraction time
- preserve source identity
- preserve market-line snapshot provenance
- support deterministic rebuilds and audit trails

Classification: `Theory-compatible engineering support`

## Historical Research Inputs

Historical manifests should support:

- manifest-backed bundle imports
- research-season scoping
- deterministic replay against historical odds and results

Classification: `Theory-compatible engineering support`
