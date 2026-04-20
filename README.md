# MLB-First Actuarial Betting Platform

This repository is being rebuilt into an MLB-first statistical ensemble betting system for:

- moneyline
- runline
- totals

The governing statistical sources are:

- `statistical_theory/09_GLM_Generalized_Linear_Models_for_Insurance_Rating.pdf`
- `statistical_theory/10_Holmes_Casotto_Penalized_Regression_and_Lasso_Credibility_2025_Revision.pdf`

The old NHL/NBA system established useful infrastructure, but it is no longer the governing product contract. Legacy NHL/NBA code may still exist during migration; the active contract, docs, configs, and sprint plan now point at the MLB rebuild.

## Current Phase

Sprint 0 is complete. The repo is now in an integrated MLB modeling-core checkpoint spanning:

- Sprint 1 data foundation: MLB config, ingest, storage, and odds lanes are live
- Sprint 2 feature architecture: MLB-first pregame feature seams, leakage guards, and league-specific feature boundaries are active
- Sprint 4 penalized regression and lasso credibility: penalized GLM contracts, lambda search, and opt-in lasso-credibility training/reporting lanes are live
- Sprint 6 validation gold standard: the MLB validation pipeline now emits contract-backed validation outputs, calibration summaries, and honest penalized/credibility applicability handling
- Sprint 8 dashboard/staging plumbing: MLB staging payloads and shipped-site contract are wired into the web lane

Still incomplete:

- the full multi-target vanilla GLM / GLM-extension model factory
- the full residual / lift / stability / promotion evidence surface for every theory-core family
- champion ensemble promotion governance and release hardening

See `docs/mlb/` for the authoritative rebuild documents.

## Product Direction

The target product is a local-first actuarial betting platform with:

- deterministic ingest and orchestration
- pregame-only feature engineering with leakage protection
- GLM-family, penalized-GLM, and lasso-credibility modeling
- formal diagnostics and validation artifacts
- immutable historical prediction ledgers
- operator dashboard and committed staging payloads
- evidence-backed champion ensemble promotion

## Primary Commands

The rebuild is moving the default contract to MLB:

```bash
make init-db
make fetch
make fetch-odds
make features
make train
make validate
make dashboard
```

Equivalent explicit MLB lane:

```bash
make init-db CONFIG=configs/mlb.yaml
make fetch CONFIG=configs/mlb.yaml
make fetch-odds CONFIG=configs/mlb.yaml
make features CONFIG=configs/mlb.yaml
make train CONFIG=configs/mlb.yaml
make validate CONFIG=configs/mlb.yaml
```

Repo-level orchestration is also being rewritten around the MLB lane:

```bash
make data_refresh
make hard_refresh
```

## Repo Structure

- `src/` Python pipeline, storage, training, evaluation, orchestration, and query code
- `web/` Next.js dashboard and staging generation
- `configs/` active config surfaces, registries, and generated manifests
- `docs/mlb/` MLB-first rebuild docs and theory traceability
- `data/` raw, interim, and processed MLB outputs
- `artifacts/` validation and reporting outputs
- `statistical_theory/` governing monographs and theory notes
- `tests/` contract, unit, and integration tests

## Theory Boundary

Every major item in the rebuild should be labeled as one of:

- direct theory implementation
- theory-compatible engineering support
- betting overlay
- dashboard/reporting layer

That boundary matters. The repo should never present an operator UX or betting heuristic as if it were direct CAS monograph theory.

## Notes

- The strongest reusable patterns from the legacy system are the code-first registry, manifest generation, immutable prediction ledger, deterministic orchestration, and staging pipeline.
- The statistical core is being rewritten around MLB and the two CAS monographs, not extended from an NHL/NBA-first theory contract.
