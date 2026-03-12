# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NHL/NBA/NCAAM home-win probability forecasting system with a daily pipeline (`fetch → features → train → predict → ingest results → score → aggregates → artifacts`), walk-forward backtesting, prequential scoring, and a Next.js dashboard deployed as a static GitHub Pages site via committed JSON snapshots.

## Commands

All commands run via `make` from the repo root. Default config is `configs/nba.yaml`.

### Python

```bash
make install-python              # pip install -e '.[dev]'
make test                        # pytest (all tests)
pytest tests/test_foo.py         # single test file
pytest tests/test_foo.py::test_bar  # single test
```

### Node (web/)

```bash
make install-node                # cd web && npm install
make dashboard                   # Next.js dev server on localhost:3000
cd web && npm run lint           # ESLint (0 warnings allowed)
cd web && npm run build          # Next.js build
cd web && npm run build:pages    # static GitHub Pages export
```

### Dashboard smoke tests (Playwright)

```bash
make smoke-dashboard             # installs chromium + runs playwright tests
# or manually:
cd web && npm run playwright:install && npm run test:smoke
```

### Pipeline operations

```bash
make init-db CONFIG=configs/nhl.yaml
make fetch CONFIG=configs/nba.yaml
make fetch-odds CONFIG=configs/nba.yaml
make features CONFIG=configs/nba.yaml
make train CONFIG=configs/nba.yaml MODELS=glm_ridge,rf
make train CONFIG=configs/nba.yaml APPROVE_FEATURE_CHANGES=1
make validate CONFIG=configs/nba.yaml
make backtest CONFIG=configs/nba.yaml
make run_daily CONFIG=configs/nba.yaml

make data_refresh                # all-league data-only refresh (no training)
make hard_refresh                # all-league fetch + train + staging + commit + push
make hard_refresh DRY_RUN=1      # preview step plan without executing
make query Q="How much did I win last night?"
```

### Staging snapshots

```bash
cd web && npm run generate:staging-data   # regenerate JSON from local SQLite
```

Dashboard code changes that affect the shipped staging experience require regenerating and committing `web/public/staging-data/` before pushing.

## Architecture

### Python pipeline (`src/`)

| Package | Role |
|---------|------|
| `cli.py` | Typer-style CLI entry; `commands/` dispatches to handlers |
| `league_registry.py` | `LeagueAdapter` — unified interface for NHL/NBA/NCAAM config, fetchers, feature builders |
| `data_sources/{nhl,nba,ncaam}/` | League-specific HTTP clients (NHLE, ESPN APIs) |
| `features/` | Feature engineering with leakage checks and guardrails |
| `models/` | GLM (ridge/lasso/elastic/vanilla), RF, GBDT, NN, Bayesian state-space |
| `training/` | Fit/predict runners, ensemble builders (weighted avg + stacking), feature selection, CV |
| `evaluation/` | Metrics, calibration, diagnostics, validation pipeline |
| `services/` | High-level orchestration: train, backtest, validate, ingest |
| `orchestration/` | Multi-league pipelines: `data_refresh`, `hard_refresh` |
| `storage/` | SQLite schema + I/O (`schema.py`, `db.py`) |
| `query/` | Deterministic local query system for betting history, team forecasts, model analysis |
| `bayes/` | Bayesian state-space model with offline fit and daily sequential updates |

### Configuration (`configs/`)

YAML inheritance: `default.yaml` → league-specific `nhl.yaml` / `nba.yaml` / `ncaam.yaml`. Key config sections: `project`, `paths`, `data`, `modeling`, `validation_split`, `bayes`, `runtime`, `feature_policy`.

Per-league feature contracts:
- `model_feature_map_{league}.yaml` — per-model feature subsets
- `model_feature_guardrails_{league}.yaml` — blocked/watchlist features
- `feature_registry_{league}.yaml` — feature discovery registry

Feature policy defaults to `production` mode, which blocks training if feature drift is detected. Pass `APPROVE_FEATURE_CHANGES=1` to accept new contracts.

### Next.js dashboard (`web/`)

- **Framework**: Next.js 16 with App Router, React 18, TypeScript
- **Routing**: App Router pages under `web/app/` (predictions, performance, calibration, diagnostics, validation, bet-history, market-board, bet-sizing, etc.)
- **API routes**: `web/app/api/` — serve data from local SQLite or static staging JSON
- **Styling**: CSS Modules (`.module.css` alongside components)
- **Static staging**: `web/public/staging-data/{league}/*.json` + `manifest.json` — committed snapshots served by GitHub Pages
- **Path alias**: `@/*` maps to `web/` root

### Database

SQLite with key tables: `games`, `results`, `predictions` (immutable pregame ledger), `upcoming_game_forecasts`, `model_runs`, `model_scores`, `performance_aggregates`, `change_points`, `odds_snapshots`, `odds_market_lines`, `feature_sets`, `validation_results`.

### Deployment

`.github/workflows/pages-staging.yml` builds and deploys the static site on every push to `main`. The Pages build uses committed files under `web/public/staging-data/`, not live SQLite data.

## Key Conventions

### Git workflow (from AGENTS.md)

- Stay on `main`. Do not create feature branches unless explicitly asked.
- After edits, commit and push to `main` by default.
- After every push, watch the `Publish Sanitized Staging Site` GitHub Actions workflow:
  ```bash
  gh run list --workflow "Publish Sanitized Staging Site" --limit 5 --json databaseId,headSha,status,conclusion,url,displayTitle
  gh run watch <databaseId> --interval 5
  ```

### Cross-league parity

If a bug is found in one league, investigate the same failure mode in all supported leagues (NHL, NBA, NCAAM) before closing. Fix shared or league-specific paths as appropriate and regenerate affected staging snapshots.

### Hard refresh protocol

`make hard_refresh` runs a deterministic sequential pipeline: init-db → fetch → fetch-odds → train for each league (NHL → NBA → NCAAM), then generates staging snapshots, commits, pushes, and watches the workflow. Must be fail-fast, no parallelization, no step reordering.

### Data refresh protocol

`make data_refresh` runs fetch + fetch-odds for all leagues sequentially. Does not rebuild features, train, or regenerate staging.

### Query system

Use `make query Q="..."` for betting history, team forecasts, model leaderboard, and team report questions. Defaults to NBA config. Interpret model "performance" as betting profitability unless the user explicitly asks about statistical metrics.

### Predictions immutability

The `predictions` table is the immutable pregame ledger. `prediction_diagnostics` is for synthetic diagnostics only.
