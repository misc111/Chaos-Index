# NHL + NBA Probabilistic Forecasting System

Production-grade NHL/NBA home-win probability forecasting with:
- daily pipeline (`fetch -> features -> train/update -> predict -> ingest results -> score -> aggregates -> artifacts`)
- walk-forward backtesting and prequential scoring
- model calibration/diagnostics/validation artifacts
- SQLite-backed deterministic local query command
- Next.js dashboard for predictions, performance, calibration, diagnostics, slices, and validation

## Repo Structure

- `src/` Python pipeline, models, evaluation, storage, query tools
- `web/` Next.js dashboard + API routes
- `configs/` runtime configuration
- `data/` cached raw snapshots + interim/processed outputs
- `artifacts/` plots/reports/validation artifacts
- `tests/` smoke + unit tests

## Install

### Python deps
```bash
make install-python
```

### Node deps
```bash
make install-node
```

## Core Commands

Initialize DB schema (defaults to NHL config):
```bash
make init-db
```

Fetch/cache league data + ingest results:
```bash
make fetch
```
```bash
make fetch CONFIG=configs/nba.yaml
```

Build features with leakage checks/fallback metadata:
```bash
make features
```

Research per-model feature subsets and optionally promote them into the active league map:
```bash
make research-features CONFIG=configs/nba.yaml
make research-features CONFIG=configs/nba.yaml APPROVE_FEATURE_CHANGES=1
```

Train model suite + ensemble + upcoming forecasts + validation artifacts:
```bash
make train
```
First run (or any intentional feature-contract update):
```bash
make train APPROVE_FEATURE_CHANGES=1
```

Train only selected models (comma-separated):
```bash
make train MODELS=glm_logit
make train MODELS=glm_logit,rf
```

Walk-forward backtest + prequential scoring:
```bash
make backtest
```
With explicit feature-contract approval:
```bash
make backtest APPROVE_FEATURE_CHANGES=1
```

Backtest only selected models:
```bash
make backtest MODELS=glm_logit
```

Daily end-to-end:
```bash
make run_daily
```
With explicit feature-contract approval:
```bash
make run_daily APPROVE_FEATURE_CHANGES=1
```

Daily run with selected models:
```bash
make run_daily MODELS=glm_logit
```
```bash
make run_daily CONFIG=configs/nba.yaml MODELS=glm_logit
```

NBA rebuild with explicit research phase:
```bash
make fetch CONFIG=configs/nba.yaml
make features CONFIG=configs/nba.yaml
make research-features CONFIG=configs/nba.yaml APPROVE_FEATURE_CHANGES=1
make train CONFIG=configs/nba.yaml APPROVE_FEATURE_CHANGES=1
```

Launch dashboard:
```bash
make dashboard
LEAGUE=NBA NBA_DB_PATH=data/processed/nba_forecast.db make dashboard
```

Maintainer note:
- Local dashboard changes do not automatically update the GitHub Pages staging site.
- If a dashboard/API change affects what staging should show, you must regenerate and commit `web/public/staging-data/` before pushing.

Generate the GitHub Pages staging snapshot from your current local dashboard data:
```bash
cd web
npm run generate:staging-data
```

Build the static GitHub Pages staging site locally:
```bash
cd web
npm run build:pages
```

Deployment notes:
- `.github/workflows/pages-staging.yml` deploys the staging site on every push to `main`.
- The Pages build publishes the committed files under `web/public/staging-data/`; it does not rebuild live SQLite data on GitHub Actions.
- When you want the staging site to reflect newer local forecasts, run `npm run generate:staging-data`, commit the updated JSON in `web/public/staging-data/`, and push.
- Treat this as part of dashboard maintenance: if you ship a local dashboard change that should appear on staging, update the committed staging snapshot in the same change.

Deterministic local query command:
```bash
make query Q="What's the chance the Leafs win their next game?"
make query CONFIG=configs/nba.yaml Q="What's the chance the Raptors win their next game?"
make query Q="Which model has performed best the last 60 days?"
make query Q="Give me the report of all teams in a table."
```

Python entrypoint equivalent:
```bash
python3 -m src.query.answer --config configs/nhl.yaml --question "What's the chance the Leafs win their next game?"
python3 -m src.query.answer --config configs/nba.yaml --question "What's the chance the Raptors win their next game?"
```

Run tests:
```bash
make test
```

Run smoke e2e:
```bash
make smoke
# or
scripts/smoke_e2e.sh
```

## Data + Temporal Integrity

- Uses public NHL (`api-web.nhle.com`) and NBA (`site.api.espn.com`) endpoints with retries and caching.
- Raw pulls cached under `data/raw/{source}/{YYYY-MM-DD}/...`.
- Offline fallback uses latest cached payload when live fetch fails (or when `offline_mode: true`).
- Features are generated with lagged/rolling calculations only; leakage checks run before training.

## Feature Contract Policy

- Feature contract is controlled by `feature_policy` in config (`production` or `research`).
- Default mode is `production`, which blocks silent model-feature entry/exit.
- Registry path defaults to `configs/feature_registry_{league}.yaml`.
- In production mode:
  - if drift is detected (added/removed model features), training/backtest fail fast
  - pass `--approve-feature-changes` (or `APPROVE_FEATURE_CHANGES=1` in `make`) to explicitly accept and persist the new contract
- In research mode:
  - runs are not blocked
  - newly seen features are tracked as `candidate_features` in the registry

## Model Feature Research

- Per-model research maps can be generated with `research-features`.
- Promoted model maps are stored at `configs/model_feature_map_{league}.yaml`.
- The NBA path now uses a league-specific basketball feature builder and a promoted per-model feature map during training/backtesting.

## Forecast Outputs Per Upcoming Game

Persisted in SQLite (`upcoming_game_forecasts`, `predictions`) with `as_of_utc`:
- ensemble home-win probability + predicted winner
- per-model probabilities
- spread stats (min/median/max, mean+/-sd, IQR)
- Bayes credible interval (5-95%)
- uncertainty/data-quality flags

## Performance Tracking

- `predictions` table is the prediction registry
- finalized outcomes ingested into `results`
- per-game prequential scoring in `model_scores` (log loss, Brier, accuracy)
- rolling and cumulative aggregates in `performance_aggregates`
- change-point alerts in `change_points` (CUSUM/Page-Hinkley)

## Validation Artifacts

Generated under `artifacts/validation/` and surfaced in `/validation`:
- blockwise nested model tests (LRT + OOS uplift + AME CI)
- coefficient stability paths, VIF/condition diagnostics, trade-deadline break test
- influence diagnostics (leverage/Cook's/dfbetas + refit impact)
- calibration robustness + Brier decomposition
- fragility tests (missingness + perturbation)
- backtest integrity checks

## Notes

- Odds snapshots now persist to `odds_snapshots` and `odds_market_lines` on each `fetch` run (including bookmaker/market/outcome rows) for historical line tracking.
- Configure The Odds API via env vars: `ODDS_API_KEY`, `ODDS_API_REGIONS`, `ODDS_API_MARKETS`, `ODDS_API_ODDS_FORMAT`, `ODDS_API_DATE_FORMAT`, `ODDS_API_THROTTLE_SECONDS`.
- xG, injuries, and odds adapters include explicit graceful fallback behavior when stable no-auth feeds are unavailable.
- Bayesian state-space model runs in offline fit mode and supports daily sequential updates.
- Dashboard is visualization-only; no in-dashboard chat UI.
