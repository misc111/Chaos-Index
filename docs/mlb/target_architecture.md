# MLB Target Architecture

## Objective

Make MLB the only primary shipped lane while preserving legacy code and data for
audit/migration reference. Modeling governance follows the leak-repaired
tournament and the traceability matrix in
`docs/mlb/theory_traceability_matrix.md`.

## Enforced Boundaries

- `src.registry.leagues` marks MLB as `lifecycle=primary`; NBA and NHL are
  `lifecycle=legacy` and are not primary rebuild lanes.
- `make data_refresh` and `make hard_refresh` are the canonical repo-level
  orchestration targets. They call thin Python adapters in `src/orchestration/`
  and build steps only for primary rebuild leagues.
- `src.registry.models` separates model families into:
  - `core`: vanilla GLM, ridge, lasso, elastic net, and explicit lasso
    credibility lanes.
  - `extension`: GAM, GLMM, DGLM, and score/count GLM bridges.
  - `experimental`: MARS hinge proxy, two-stage proxy, tree, neural, and
    Bayesian challengers.
- Top-level `src/models/` is reserved for core implementations and
  compatibility shims. Extension implementations live under
  `src/models/extensions/`; non-CAS and proxy implementations live under
  `src/models/experimental/`.
- `web/public/staging-data/manifest.json` ships only MLB. Legacy NBA/NHL JSON
  snapshots are retained under `web/public/staging-data/legacy/`.
- Tournament and recommendation artifacts must carry `theory_classification`
  labels: `core-supported`, `theory-compatible extension`, or `experimental`.

## Preferred Directory Targets

- `configs/mlb.yaml`
- `configs/feature_registry_mlb.yaml`
- `configs/model_feature_map_mlb.yaml`
- `configs/model_feature_guardrails_mlb.yaml`
- `data/raw/mlb/`
- `data/interim/mlb/`
- `data/processed/mlb/`
- `artifacts/validation/mlb/`
- `artifacts/reports/mlb/`
- `artifacts/reports/mlb/tournament/research_director_20260422_second_round_leakage_repaired/`
- `docs/mlb/`
- `web/public/staging-data/mlb/`

## Layer Map

1. MLB ingest and raw cache
2. Normalized MLB persistence
3. Pregame feature store with leakage enforcement
4. Core GLM / penalized GLM / lasso credibility model lane
5. Opt-in extension and experimental challenger lanes
6. Validation, diagnostics, and tournament artifacts with theory labels
7. Ensemble and betting overlays, separated from theory-core model selection
8. MLB-only dashboard and staging delivery

## Residual Migration Debt

- NBA/NHL configs, feature builders, data sources, and tests still exist as
  legacy compatibility surfaces.
- Some legacy tests intentionally exercise non-MLB paths to prevent accidental
  breakage while the migration finishes.
- Full production MLB champion promotion is still blocked by bounded-data
  evidence, missing robust market complements, and incomplete long-horizon
  validation depth.
- Legacy top-level model import paths remain as compatibility shims; new code
  should import extensions and experimental challengers from their fenced
  namespaces.

## Acceptance Standard

The architecture is real only when the MLB lane can be refreshed, trained,
validated, tournament-compared, and staged with generated manifests and guard
tests proving these boundaries.
