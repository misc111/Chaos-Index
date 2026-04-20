# MLB Target Architecture

## Objective

Preserve the strongest engineering patterns from the legacy system while resetting the statistical and product contract around MLB.

## Preserve

- code-first registries and generated manifests
- deterministic orchestration
- immutable pregame prediction ledger
- staging snapshot pipeline
- artifact and report generation discipline

## Rewrite

- league registry defaults and scope contract
- MLB ingest adapters
- MLB feature strategies and registries
- MLB model inventory and validation factory
- MLB dashboard payloads and staging data

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
- `docs/mlb/`
- `web/public/staging-data/mlb/`

## Layer Map

1. Ingest and raw cache
2. Normalized MLB persistence
3. Pregame feature store with availability enforcement
4. GLM / penalized / credibility / extension model factory
5. Validation artifact factory
6. Ensemble and betting overlay
7. Dashboard and staging delivery

## Acceptance Standard

The architecture is only considered real when the MLB lane can be refreshed, trained, validated, and surfaced through the dashboard/staging contract with documentation that matches the implementation.
