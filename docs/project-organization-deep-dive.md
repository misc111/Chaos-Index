# SportsModeling Project Organization Deep Dive

## Purpose

This repository is a multi-league sports forecasting platform for:

- NHL
- NBA
- NCAAM

Its core product is probabilistic home-win forecasting with a repeatable end-to-end workflow:

`fetch -> features -> train/update -> predict -> ingest results -> score -> aggregates -> artifacts`

Around that core pipeline, the repo also provides:

- deterministic local natural-language query answering
- walk-forward backtesting and research comparisons
- validation and diagnostics artifacts
- a Next.js dashboard for live local inspection
- a committed GitHub Pages staging snapshot under `web/public/staging-data/`
- deterministic repo-level refresh orchestration across all supported leagues

The project is organized around one central idea: keep league-specific behavior behind explicit registries, adapters, strategies, and policies, while reusing one shared pipeline shape across all supported leagues.

## High-Level Layout

At the top level, the repository is divided into a few major surfaces:

- `src/`
  Python application code for ingestion, features, modeling, scoring, validation, orchestration, query answering, and storage.
- `web/`
  Next.js dashboard, server-side data shaping, route handlers, staging-data generation, and static Pages build support.
- `configs/`
  Runtime YAML configs, feature registries, model feature maps, and generated manifest JSON.
- `data/`
  Local raw, interim, and processed data outputs.
- `artifacts/`
  Trained model outputs, reports, research outputs, and validation artifacts.
- `docs/generated/`
  Generated reference docs derived from code registries.
- `tests/`
  Python tests for contracts, pipelines, league parity, validation, query behavior, and orchestration.
- `scripts/`
  Small shell and export helpers.
- `statistical_theory/`
  Theory and modeling notes that capture longer-lived thinking outside the runtime pipeline.

## Architectural Philosophy

Several repo-wide design choices shape how the code is organized:

### 1. Code-first registries are the canonical source of truth

The repo explicitly states that `src/registry/` is the canonical source of truth for:

- supported leagues
- supported models
- supported CLI commands
- supported dashboard routes
- subsystem documentation metadata

Those registries generate:

- committed JSON manifests under `configs/generated/`
- generated TypeScript runtime metadata under `web/lib/generated/`
- generated Markdown reference docs under `docs/generated/`
- generated subsystem README files

This reduces drift between Python, the dashboard, and documentation.

### 2. Thin entrypoints, thicker service and training layers

The repo tries to keep:

- CLI parsing in `src/cli.py`
- command dispatch in `src/commands/`
- application-layer orchestration in `src/services/`
- modeling and pipeline math in `src/training/`, `src/features/`, `src/evaluation/`, and `src/models/`

That separation is visible in both the README and the module docstrings.

### 3. League-specific logic is pushed to adapters and strategies

Cross-league orchestration should not branch arbitrarily everywhere. Instead, the project centralizes league differences in:

- `src/registry/leagues.py`
- `src/league_registry.py`
- `src/data_sources/<league>/`
- `src/features/strategies/<league>.py`
- selected query alias and championship helpers
- league-aware policies for uncertainty, naming, and feature maps

### 4. Local dashboard and staging site are separate delivery targets

The repo treats the local Next.js dashboard and the GitHub Pages staging site as distinct products:

- the live local dashboard reads local SQLite-backed data flows
- GitHub Pages publishes committed JSON snapshots under `web/public/staging-data/`

This is a major operational boundary and one of the most important things for an outside reviewer to understand.

### 5. Historical prediction integrity is a first-class data contract

The storage layer separates:

- immutable pregame history in `predictions`
- synthetic or diagnostic rows in `prediction_diagnostics`

That prevents backtests or out-of-fold reconstructions from silently rewriting what the app presents as historical live forecasts.

## Top-Level Execution Surfaces

The main human-facing execution surfaces are:

### `Makefile`

The `Makefile` is the primary ergonomic wrapper for both Python and web workflows. Important targets include:

- install and verification:
  - `install-python`
  - `install-node`
  - `lint`
  - `typecheck`
  - `test`
  - `verify`
- league-scoped Python pipeline commands:
  - `init-db`
  - `fetch`
  - `refresh-data`
  - `fetch-odds`
  - `import-history`
  - `features`
  - `research-features`
  - `train`
  - `validate`
  - `compare-candidates`
  - `backtest`
  - `research-backtest`
  - `run_daily`
  - `smoke`
- repo-level orchestration:
  - `data_refresh`
  - `hard_refresh`
- product surfaces:
  - `dashboard`
  - `smoke-dashboard`
  - `query`
- documentation/codegen:
  - `docs-generate`
  - `docs-check`

The `Makefile` defaults to `CONFIG=configs/nba.yaml`, so NBA is the operational default when context is ambiguous.

### `src/cli.py`

`src/cli.py` is the Python application entrypoint. Its responsibilities are intentionally narrow:

- load `.env` and `web/.env.local` when present
- build argparse subcommands from `src.registry.commands`
- load the selected YAML config via `src.common.config.load_config`
- initialize logging
- dispatch to a command handler in `src.commands`

It does not implement business logic directly.

### `make query` / `src/query/answer.py`

The repo includes a deterministic natural-language query entrypoint for product questions such as:

- next-game win probability
- next few games
- best model over a recent window
- championship odds heuristics
- betting history summaries

This is important because model performance and betting history are treated as product surfaces, not only engineering internals.

### `web/`

The dashboard is an independent Next.js application that consumes the same underlying local artifacts and manifests but through web-facing route services and repositories.

## Registry and Generated Metadata Layer

The registry system is one of the clearest organizing mechanisms in the repo.

### `src/registry/types.py`

Defines the typed primitives used by the rest of the registry system:

- `LeagueRegistryEntry`
- `ModelRegistryEntry`
- `CommandArgumentSpec`
- `CommandRegistryEntry`
- `DashboardRouteRegistryEntry`
- `SubsystemDocEntry`

These types make the registry layer explicit rather than ad hoc.

### `src/registry/leagues.py`

Defines canonical league metadata:

- code
- slug
- display label
- default config path
- config env var
- project name
- DB path
- DB env var
- championship naming
- uncertainty policy name
- aliases

Current registered leagues:

- `NHL`
- `NBA`
- `NCAAM`

This module also provides:

- league canonicalization
- config path resolution
- DB path resolution
- machine-readable manifest payload generation

### `src/registry/models.py`

Defines the canonical model catalog for both Python and web consumers. It captures:

- canonical model key
- display labels and short labels
- model family
- aliases
- legacy keys
- trainable flag
- prediction report ordering

Current registered model families include:

- ratings
- linear
- tree
- hybrid
- goals
- simulation
- bayes
- neural

Current registered models include:

- `elo_baseline`
- `dynamic_rating`
- `glm_ridge`
- `glm_elastic_net`
- `glm_lasso`
- `gbdt`
- `rf`
- `two_stage`
- `goals_poisson`
- `simulation_first`
- `bayes_bt_state_space`
- `bayes_goals`
- `nn_mlp`

### `src/registry/commands.py`

Defines the declarative command registry used to:

- build argparse command-line options
- drive generated command docs
- drive help text
- resolve the handler import path for each command

This keeps the CLI contract centralized rather than duplicated across parser code and docs.

### `src/registry/dashboard_routes.py`

Defines the canonical dashboard/staging routes:

- `actualVsExpected`
- `betHistory`
- `gamesToday`
- `marketBoard`
- `metrics`
- `performance`
- `predictions`
- `validation`

Each entry includes:

- API path
- page path
- staging file name
- payload contract key
- whether it participates in staging
- whether it supports experimental variants

### `src/registry/subsystems.py`

Defines subsystem documentation metadata, which in turn feeds:

- `docs/generated/architecture.md`
- generated subsystem README files

This is effectively a lightweight ownership map for the repo.

### `src/registry/generate.py`

This is the code-generation engine. It produces:

- JSON manifests in `configs/generated/`
- generated TypeScript runtime metadata in `web/lib/generated/`
- generated docs in `docs/generated/`
- generated README files for subsystem folders

This module is a key bridge between Python, docs, and the web app.

### `src/registry/verify.py`

This is the registry-driven contract checker. It verifies:

- generated artifacts are up to date
- required README files exist
- public docstrings exist on designated modules
- oversized files are either avoided or explicitly allowlisted
- forbidden hard-coded config/DB/env literals do not leak outside resolver modules

This is a strong signal that the repo is trying to enforce architectural discipline, not just implement functionality.

## Configuration Model

### Base config

`configs/default.yaml` defines shared defaults for:

- project metadata
- path layout
- data window settings
- modeling settings
- validation split policy
- Bayesian hyperparameters
- runtime toggles
- feature policy behavior
- research workflow settings

### League configs

Each league config extends the default base:

- `configs/nhl.yaml`
- `configs/nba.yaml`
- `configs/ncaam.yaml`

Differences across leagues include:

- `project.name`
- `paths.interim_dir`
- `paths.processed_dir`
- `paths.db_path`
- `data.league`
- season windows
- history windows
- validation split mode

Notable example:

- NBA and NHL use `train_test`
- NCAAM uses `train_validation_test`

### Config loading

`src/common/config.py` loads YAML into a typed `AppConfig` based on Pydantic models such as:

- `ProjectConfig`
- `PathsConfig`
- `DataConfig`
- `ModelingConfig`
- `ValidationSplitConfig`
- `BayesConfig`
- `RuntimeConfig`
- `FeaturePolicyConfig`
- `ResearchConfig`

The loader also supports one level of config inheritance through `extends`.

### Feature-policy configs

Feature-contract management is partly config-driven:

- `configs/feature_registry_<league>.yaml`
- `configs/model_feature_map_<league>.yaml`
- `configs/model_feature_guardrails_<league>.yaml`

These support two related but different concerns:

- feature-contract stability for production
- per-model feature selection and guardrails

## Python Package Map

### `src/common/`

Cross-cutting infrastructure for:

- typed config loading
- generated manifest loading
- logging
- time helpers
- utility functions

This package is the repo’s shared glue layer.

### `src/commands/`

Thin CLI wrappers that translate parsed arguments into service calls.

- `data.py`
  wraps ingest, features, odds refresh, and history import
- `modeling.py`
  wraps train, validate, backtest, compare-candidates, research-backtest, and daily run flows
- `smoke.py`
  runs a reduced-window end-to-end smoke flow and sample local queries

### `src/orchestration/`

Repo-level deterministic multi-league workflows.

- `refresh_pipeline.py`
  builds ordered shell-step plans and executes them sequentially
- `data_refresh.py`
  repo-wide ingest-only pipeline
- `hard_refresh.py`
  repo-wide init/fetch/fetch-odds/train/staging/push/workflow-watch pipeline

This layer is intentionally separate from the league-scoped Python services.

### `src/services/`

Application-layer orchestration over the lower-level modules.

- `ingest.py`
  database initialization, raw snapshot insertion, interim-file persistence, ingestion, odds persistence, feature-build invocation, and data refresh flow
- `train.py`
  load processed features, apply feature policy, call training package, persist predictions and forecasts, persist OOF diagnostics, score outputs, and trigger validation artifacts
- `validate.py`
  regenerate validation artifacts from a saved trained run
- `backtest.py`
  walk-forward backtest orchestration and persistence closeout
- `model_compare.py`
  research-only candidate comparison service
- `research_backtest.py`
  historical research backtest service
- `history_import.py`
  historical import logic for research datasets

The service layer is where repo-level persistence and artifact responsibilities sit.

### `src/storage/`

SQLite schema and persistence helpers.

- `schema.py`
  full DDL for pipeline tables
- `db.py`
  simple database wrapper plus online migration logic
- `prediction_history.py`
  contracts around immutable prediction history versus diagnostics
- `tracker.py`
  run/artifact tracking utilities
- `io.py`
  storage-related helper functions

Important storage tables include:

- `raw_snapshots`
- `games`
- `results`
- `teams`
- `feature_sets`
- `model_runs`
- `predictions`
- `prediction_diagnostics`
- `upcoming_game_forecasts`
- `model_scores`
- `performance_aggregates`
- `change_points`
- `validation_results`
- `odds_snapshots`
- `odds_market_lines`
- `historical_bet_decisions`
- `historical_bet_decisions_by_profile`

The schema strongly suggests the repo is organized around a local SQLite product database, not around purely ephemeral notebook workflows.

### `src/data_sources/`

League-specific ingest adapters and shared source contracts.

- shared:
  - `base.py`
  - `odds_api.py`
- per league:
  - `src/data_sources/nhl/`
  - `src/data_sources/nba/`
  - `src/data_sources/ncaam/`

Per-league modules cover:

- games
- teams
- schedule
- players
- injuries
- odds
- results
- xg
- goalie/game stats

The shared orchestration surface for these modules is `src/league_registry.py`, which assembles a typed `LeagueAdapter` from the registered league metadata plus the league-specific fetch/build functions.

### `src/features/`

Feature engineering is organized as a shared pipeline plus league strategies.

Core shared files:

- `pipeline.py`
  stable pipeline stages for interim loading, team-game expansion, rolling windows, game-level merge, and final feature-frame persistence
- `build_features.py`
  shared caller entrypoint
- `leakage_checks.py`
  leakage detection before training

Reusable feature helpers include:

- `elo.py`
- `dynamic_ratings.py`
- `travel.py`
- `contextual_effects.py`
- `goalie_features.py`
- `rink_adjustments.py`
- `special_teams.py`
- `intermediates.py`

League-specific strategy classes live in:

- `src/features/strategies/nhl.py`
- `src/features/strategies/nba.py`
- `src/features/strategies/ncaam.py`

This is one of the cleanest extension seams in the repo.

### `src/models/`

Concrete model implementations by family.

Examples:

- ratings:
  - `glm_ridge.py`
  - `glm_lasso.py`
  - `glm_elastic_net.py`
  - `glm_penalized.py`
- tree and hybrid:
  - `gbdt.py`
  - `rf.py`
  - `two_stage.py`
- goals and simulation:
  - `glm_goals.py`
  - `ensemble_weighted.py`
  - `ensemble_stack.py`
- bayesian:
  - `bayes_state_space_bt.py`
  - `bayes_state_space_goals.py`
- neural:
  - `nn.py`

This directory mostly contains model math and direct estimator implementations.

### `src/training/`

This package is the main modeling runtime layer. It converts processed features into:

- fit models
- out-of-fold predictions
- ensemble probabilities
- upcoming forecasts
- saved artifacts
- diagnostics and scores

Important files:

- `train.py`
  main orchestration entrypoint `train_and_predict`
- `model_catalog.py`
  canonical model normalization built from the shared registry
- `fit_runner.py`
  model fitting flow
- `predict_runner.py`
  prediction generation flow
- `ensemble_builder.py`
  ensemble weight, stacker, and spread assembly
- `ensemble_policy.py`
  rules for which models participate in ensembles
- `feature_selection.py`
  raw feature selection helpers
- `feature_policy.py`
  production versus research feature-contract enforcement
- `model_feature_research.py`
  per-model feature research and promotion
- `model_feature_guardrails.py`
  feature constraints and checks
- `penalized_glm.py`
  GLM tuning and penalized-model feature handling
- `artifact_writer.py`
  artifact serialization
- `backtest.py`
  walk-forward logic
- `prequential.py`
  scoring logic
- `cv.py`
  cross-validation helpers
- `uncertainty_policy.py`
  uncertainty flag generation
- `progress.py`
  structured progress emission

This is arguably the most important package in the repo.

### `src/evaluation/`

Diagnostics, validation, significance, drift, and scoring surfaces.

Important files include:

- `metrics.py`
- `calibration.py`
- `brier_decomposition.py`
- `performance_timeseries.py`
- `slice_analysis.py`
- `change_detection.py`
- `drift.py`
- `diagnostics_glm.py`
- `diagnostics_ml.py`
- `validation_pipeline.py`
- `validation_classification.py`
- `validation_significance.py`
- `validation_nonlinearity.py`
- `validation_stability.py`
- `validation_influence.py`
- `validation_fragility.py`
- `validation_backtest_integrity.py`
- `research_betting.py`

This package turns model outputs into analysis surfaces that can be stored, plotted, or surfaced in the dashboard.

### `src/query/`

Deterministic natural-language product query answering.

Important files:

- `answer.py`
  top-level router
- `intent_parser.py`
  query parsing and intent resolution
- `team_aliases.py`
  cross-league alias handling
- `team_handlers.py`
  team-centric forecast answers
- `bet_history_handlers.py`
  bet-history summaries and breakdowns
- `report_handlers.py`
  best-model and report answers
- `championship_estimators.py`
  heuristic championship answers
- `contracts.py`
  query adapter interfaces
- `templates.py`
  deterministic answer text formatting

This package is important because it exposes the project’s internal data model as a user-facing local QA/product interface.

### `src/research/`

Research-only comparison and experimentation.

Primary files:

- `candidate_models.py`
- `model_comparison.py`

This area appears to support broader experimentation outside the stricter production training contract.

### `src/bayes/`

Bayesian state-space flows and diagnostics:

- offline fit
- daily update
- posterior predictive diagnostics

This package is narrower than `src/training/` and seems to support the Bayesian model family specifically.

### `src/simulation/`

Simulation-specific support code for forecast and betting workflows.

Currently centered on:

- `game_simulator.py`

## End-to-End Data Flow

The typical league-scoped pipeline is:

### 1. Config and DB initialization

- select config such as `configs/nba.yaml`
- optionally initialize the SQLite schema with `init-db`

### 2. Data ingestion

`src/services/ingest.py` uses:

- `src.league_registry.get_league_adapter`
- league-specific `src/data_sources/<league>/...`
- `HttpClient` from `src.data_sources.base`

It persists:

- raw snapshot metadata into `raw_snapshots`
- structured `games`
- structured `results`
- `teams`
- odds snapshots and market lines
- interim parquet/csv files under the configured `interim_dir`

Interim file names are standardized via `INTERIM_FILES`:

- `games`
- `schedule`
- `teams`
- `players`
- `goalies`
- `injuries`
- `odds`
- `xg`

### 3. Feature building

`src.features.build_features` enters the shared pipeline in `src/features/pipeline.py`.

Core steps:

- load interim data
- expand each game into home and away team-game records
- compute per-team rolling windows
- merge back to game level
- apply league strategy transforms
- persist final features to `features.parquet` or `features.csv`

The resulting `FeatureBuildResult` includes:

- finalized dataframe
- numeric feature columns
- `feature_set_version`
- metadata

### 4. Feature policy enforcement

Before training or backtesting, the repo enforces a feature contract through `src/training/feature_policy.py`.

Modes:

- `production`
  blocks feature drift unless explicitly approved
- `research`
  allows broader discovery while tracking new candidates

This is a notable governance mechanism: the repo treats feature drift as a managed contract, not an incidental side effect.

### 5. Model training and prediction

`src/services/train.py` loads features and passes them into `src/training/train.py::train_and_predict`.

The training flow includes:

- split train versus upcoming rows
- feature selection
- leakage checks
- optional GLM hyperparameter tuning
- model suite fitting
- artifact saving
- out-of-fold prediction generation
- stacking/weighted ensemble construction
- upcoming forecast generation
- uncertainty flags
- run payload serialization

### 6. Persistence closeout

The service layer persists:

- immutable live-style predictions into `predictions`
- upcoming forecast rows into `upcoming_game_forecasts`
- synthetic OOF/backtest data into `prediction_diagnostics`
- model run metadata into `model_runs`
- feature set metadata into `feature_sets`

### 7. Scoring and aggregates

Scoring is handled through `src/training/prequential.py` and related evaluation modules, producing:

- `model_scores`
- `performance_aggregates`
- change detection and validation surfaces

### 8. Validation artifacts

`src/evaluation/validation_pipeline.py` produces saved validation sections and artifacts that later feed:

- local analysis
- dashboard payloads
- staging snapshots

## Multi-League Orchestration

The repo distinguishes between:

- league-scoped commands
- repo-wide deterministic orchestration

### League-scoped commands

Examples:

- `make fetch CONFIG=configs/nhl.yaml`
- `make features CONFIG=configs/nba.yaml`
- `make train CONFIG=configs/ncaam.yaml`

These remain independently runnable.

### Repo-wide data refresh

`make data_refresh` calls `src/orchestration/data_refresh.py`, which uses `src/orchestration/refresh_pipeline.py`.

Its deterministic sequence is:

- fetch NHL
- fetch NBA
- fetch NCAAM
- fetch odds NHL
- fetch odds NBA
- fetch odds NCAAM

No features, training, or staging regeneration occur in this flow.

### Repo-wide hard refresh

`make hard_refresh` calls `src/orchestration/hard_refresh.py`.

Its deterministic sequence is:

- init DB for NHL/NBA/NCAAM
- fetch for NHL/NBA/NCAAM
- fetch odds for NHL/NBA/NCAAM
- train for NHL/NBA/NCAAM
- `web` staging-data generation
- optional Pages build
- commit
- push
- GitHub Actions workflow watch

This is implemented as a strict shell-step plan, not as loose orchestration logic. That improves determinism and auditability.

## Web Application Organization

The `web/` app is a second major subsystem, not just a thin view over Python scripts.

### Core stack

From `web/package.json`:

- Next.js 16
- React 18
- TypeScript
- Playwright for smoke tests

Important scripts:

- `npm run dev`
- `npm run build`
- `npm run build:pages`
- `npm run generate:staging-data`
- `npm run lint`
- `npm run typecheck`
- `npm run test:smoke`

### App shell

`web/app/layout.tsx` defines the top-level dashboard shell:

- header
- sidebar
- theme bootstrap
- main content area

The app brands itself as `Chaos Index`.

### App pages

Current page surfaces include:

- `/`
- `/games-today`
- `/bet-sizing`
- `/market-board`
- `/bet-history`
- `/actual-vs-expected`
- `/predictions`
- `/leaderboard`
- `/performance`
- `/calibration`
- `/diagnostics`
- `/slices`
- `/validation`

### API routes

`web/app/api/` contains thin route handlers. Canonical dashboard routes are registry-backed, while additional operational routes also exist:

- registry-backed:
  - `actual-vs-expected`
  - `bet-history`
  - `games-today`
  - `market-board`
  - `metrics`
  - `performance`
  - `predictions`
  - `validation`
- operational routes:
  - `refresh-data`
  - `refresh-odds`
  - `train-models`

The generated registry only covers the public/staging route contract, not every operational API endpoint.

### Server-side web architecture

`web/lib/server/` is the main backend-for-frontend layer.

It is organized around:

- `repositories/`
  direct data access and SQL helpers
- `services/`
  route-specific orchestration and payload shaping
- `payload-contracts.ts`
  stable payload shapes
- `manifests.ts`
  generated manifest and runtime resolution helpers

This mirrors the Python pattern of thin entrypoints and deeper service layers.

### Generated web metadata

The dashboard consumes generated runtime metadata from:

- `web/lib/generated/league-registry.ts`
- `web/lib/generated/model-manifest.ts`
- `web/lib/generated/dashboard-routes.ts`

This means the web app and Python app share the same code-generated canonical metadata instead of maintaining separate enums by hand.

## Staging and Deployment Model

This project has a very explicit staging model.

### Local dashboard

The local dashboard reads current local data and code behavior.

### GitHub Pages staging

GitHub Pages does not rebuild live SQLite data. It publishes committed JSON snapshots from:

- `web/public/staging-data/manifest.json`
- `web/public/staging-data/<league>/*.json`

That staging data is generated by:

- `web/scripts/generate-staging-data.ts`

This script:

- iterates all supported leagues
- iterates the registry-declared staging routes
- calls the route handlers directly
- sanitizes validation payloads for public staging
- writes per-league JSON files plus metadata

It also writes experimental performance variants where supported.

### Static export

`web/next.config.js` supports a static export mode with:

- `STATIC_EXPORT=1`
- optional `PAGES_BASE_PATH`

The Pages build is performed by:

- `npm run build:pages`

### Operational implication

If a dashboard or payload change should appear on staging, code changes alone are not enough. The developer must also regenerate and commit the staging snapshot.

This is one of the most important repo conventions.

## Query and Product-Surface Logic

The repo is not only a training pipeline. It also encodes product logic for how users ask questions about forecasts and betting.

### Intent parsing

`src/query/intent_parser.py` handles:

- league hints
- team alias resolution
- next-game and next-N-games parsing
- championship request detection
- report detection
- betting-history time-scope parsing

It defaults ambiguous contexts to NBA when no stronger signal is present.

### Team and championship handlers

`team_handlers.py` uses `upcoming_game_forecasts` plus per-model probabilities to answer:

- next-game probability
- next few games
- expected wins over the next span

### Betting history

`bet_history_handlers.py` is a substantial product-facing reporting layer that works over:

- historical decision tables
- profile selection logic
- realized results
- profit/loss computation
- rationale rendering

This is more sophisticated than a simple SQL summary and suggests the repo treats bankroll and betting explanations as core product features.

## Testing and Verification

The repo includes several layers of quality checks.

### Python tests

The `tests/` directory covers:

- league parity
- feature pipeline behavior
- leakage checks
- GLM hardening and diagnostics
- candidate model comparison
- validation pipeline and significance
- research backtest
- query parsing and answers
- odds API behavior
- hard-refresh orchestration
- registry contracts

This gives the project broad contract-level coverage across both infra and modeling logic.

### Web tests

The dashboard has Playwright smoke coverage in:

- `web/tests/playwright/dashboard-smoke.spec.ts`

That suite exercises:

- all supported leagues
- major pages
- common navigation and interactive flows
- UI error detection
- layout sanity checks

### Repo verification

`make verify` combines:

- registry verification
- pytest
- Python lint
- Python type checking
- web lint
- web typecheck

This is the main “architecture and contract drift” gate.

## Important Cross-Cutting Contracts

Several contracts recur across the repo:

### Canonical league order

The supported league order comes from `src/registry/leagues.py` and is reused by orchestration:

- NHL
- NBA
- NCAAM

### Canonical model naming

Model names are normalized through the shared registry and `src/training/model_catalog.py`, which keeps the Python and web naming aligned.

### Prediction history immutability

User-facing historical replay must come from `predictions`, not reconstructed diagnostics.

### Feature drift approval

Production feature changes must be explicitly approved.

### Registry-driven documentation and runtime metadata

Generated artifacts are not optional niceties; they are part of the architecture.

### Dashboard route registry

Public dashboard/staging routes are expected to be declared centrally, not only implemented ad hoc.

## Data and Artifact Layout

### `data/`

The data directory is organized by pipeline stage:

- `data/raw/`
  cached raw source outputs and historical imports
- `data/interim/`
  structured intermediate ingest outputs, often league-scoped
- `data/processed/`
  final processed league artifacts such as databases and feature tables

League-specific configs often point to:

- `data/interim/<league>/`
- `data/processed/<league>/`

while the DB filenames remain top-level league forecast DBs such as:

- `data/processed/nba_forecast.db`
- `data/processed/nhl_forecast.db`
- `data/processed/ncaam_forecast.db`

### `artifacts/`

The artifact directory collects multiple output classes:

- `artifacts/models/`
  saved model artifacts by run hash
- `artifacts/reports/`
  generated metrics and train-run outputs
- `artifacts/research/`
  research-only experiments and width evaluations
- `artifacts/validation/`
  current validation outputs
- `artifacts/validation-runs/`
  archived validation runs by league/date/run

This separation suggests the repo is designed both for current product delivery and for historical forensic analysis.

## How to Mentally Navigate the Repo

A reviewer can navigate the codebase efficiently using this path:

### For runtime entrypoints

Start at:

- `README.md`
- `Makefile`
- `src/cli.py`
- `src/registry/commands.py`

### For “what exists in the system”

Start at:

- `src/registry/leagues.py`
- `src/registry/models.py`
- `src/registry/dashboard_routes.py`
- `src/registry/subsystems.py`

### For ingestion and local data movement

Start at:

- `src/services/ingest.py`
- `src/league_registry.py`
- `src/data_sources/`
- `src/storage/schema.py`

### For feature engineering

Start at:

- `src/features/pipeline.py`
- `src/features/strategies/`

### For training

Start at:

- `src/services/train.py`
- `src/training/train.py`
- `src/training/model_catalog.py`
- `src/training/ensemble_builder.py`

### For validation and diagnostics

Start at:

- `src/services/validate.py`
- `src/evaluation/validation_pipeline.py`

### For product queries

Start at:

- `src/query/answer.py`
- `src/query/intent_parser.py`
- `src/query/bet_history_handlers.py`

### For the dashboard

Start at:

- `web/package.json`
- `web/app/layout.tsx`
- `web/app/api/`
- `web/lib/server/`
- `web/scripts/generate-staging-data.ts`

## Extension Seams

This repo already advertises several extension seams clearly.

### Add a league

Main touchpoints:

- `src/registry/leagues.py`
- `src/league_registry.py`
- `src/data_sources/<league>/`
- `src/features/strategies/<league>.py`
- query alias support
- tests and staging coverage

### Add a model

Main touchpoints:

- `src/registry/models.py`
- model implementation in `src/models/`
- training integration in `src/training/`
- validation/report integration
- tests

### Add a dashboard payload

Main touchpoints:

- `src/registry/dashboard_routes.py`
- route implementation under `web/app/api/`
- payload shaping under `web/lib/server/services/`
- staging generation
- tests

## Things an External Reviewer Should Pay Attention To

If this document is being handed to another model or reviewer for architectural feedback, the highest-value review angles are probably:

- whether the service/training/storage boundaries are clean enough
- whether the registry/codegen system is paying off relative to its complexity
- whether the split between local dashboard data and committed staging snapshots is operationally robust
- whether the feature-contract governance is in the right place and at the right abstraction level
- whether `src/services/train.py` and `src/services/ingest.py` are still manageable or becoming overloaded
- whether the query layer should remain SQLite/product-coupled or be abstracted further
- whether the repo-wide orchestration logic is appropriately separated from league-scoped services
- whether the web server/repository layer mirrors the Python architecture in a good way or adds duplication

## Concise Summary

This project is best understood as five interconnected systems living in one repo:

- a registry-driven multi-league metadata system
- a Python ingestion and modeling pipeline over SQLite and parquet/csv artifacts
- a validation and research system for model diagnostics and comparison
- a deterministic local query/product-answering interface
- a Next.js dashboard plus committed staging snapshot publishing flow

Its strongest organizing ideas are:

- code-first registries
- thin entrypoints
- explicit league adapters and feature strategies
- deterministic orchestration
- immutable historical prediction contracts
- explicit separation between local live behavior and published staging snapshots

That makes it a fairly disciplined production-style research/product repo rather than a loose collection of scripts.
