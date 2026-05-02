# MLB Target Architecture

## Objective

Make MLB the only shipped lane. Retired league implementations are recoverable
from git history if needed, but are intentionally absent from the working tree.
Modeling governance follows the leak-repaired tournament and the traceability matrix in
`docs/mlb/theory_traceability_matrix.md`.

## Enforced Boundaries

- `src.registry.leagues` registers MLB only.
- `make data_refresh` and `make hard_refresh` are the canonical repo-level
  orchestration targets. They call thin Python adapters in `src/orchestration/`
  and build steps only for primary rebuild leagues.
- `src.registry.models` separates model families into:
  - `core`: vanilla GLM, ridge, lasso, elastic net, and explicit lasso
    credibility lanes.
  - `extension`: theory-compatible GLM extensions (GAM, GLMM, DGLM, and score/count GLM bridges).
  - `baseline`: retired reference lane for MLB (empty in the active registry contract).
  - `experimental`: explicit non-CAS/proxy challenger rows (`mars_hinge`, `two_stage`, RF/GBDT/NN/Bayes variants) that are opt-in only and never champion-eligible.
- Top-level `src/models/` is reserved for core implementations and
  compatibility shims. Extension implementations live under
  `src/models/extensions/`; non-CAS and proxy implementations live under
  `src/models/experimental/`.
- Current `mars_hinge` is treated as an experimental proxy challenger, not as
  a direct canonical MARS implementation.
- `web/public/staging-data/manifest.json` ships only MLB and the staging root
  contains no retired league snapshots.
- Tournament and recommendation artifacts must carry `theory_classification`
  labels: `core-supported`, `theory-compatible extension`, or `experimental`.
- Validation, tournament, current-best, and promotion artifacts carry a shared
  governance contract that separates direct theory implementation,
  theory-compatible engineering support, betting overlays, and dashboard/reporting
  surfaces.

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
5. Opt-in extension lane
6. Validation, diagnostics, and tournament artifacts with theory labels
7. Ensemble and betting overlays, governed separately from theory-core model selection
8. MLB-only dashboard and staging delivery

## Residual Migration Debt

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
