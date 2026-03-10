# NHL + NBA + NCAAM Probabilistic Forecasting System

Production-grade NHL/NBA/NCAAM home-win probability forecasting with:
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
- `artifacts/` plots/reports/validation artifacts, plus archived validation runs under `validation-runs/`
- `statistical_theory/` durable notes on modeling heuristics and theory
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

Initialize DB schema (defaults to NBA config):
```bash
make init-db
```

Fetch/cache league data + ingest results:
```bash
make fetch
```
```bash
make fetch CONFIG=configs/nba.yaml
make fetch CONFIG=configs/ncaam.yaml
```

Fetch the latest standalone odds snapshot for the configured league:
```bash
make fetch-odds
```
```bash
make fetch-odds CONFIG=configs/nba.yaml
make fetch-odds CONFIG=configs/ncaam.yaml
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
make train MODELS=glm_ridge
make train MODELS=glm_ridge,rf
```

Regenerate validation artifacts from the latest saved trained run without retraining:
```bash
make validate
make validate CONFIG=configs/nba.yaml MODELS=glm_ridge
make validate CONFIG=configs/nba.yaml MODELS=glm_ridge MODEL_RUN_ID=run_1eb998e42080fd86
```

Run the research-only candidate model comparison suite against the current local feature table:
```bash
make compare-candidates
make compare-candidates CONFIG=configs/nba.yaml
make compare-candidates CONFIG=configs/nba.yaml \
  CANDIDATE_MODELS=glm_ridge,glm_lasso,glm_elastic_net,glm_vanilla \
  FEATURE_POOL=production_model_map \
  FEATURE_MAP_MODEL=glm_ridge
```
Outputs are written to `artifacts/reports/history/`.

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
make backtest MODELS=glm_ridge
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
make run_daily MODELS=glm_ridge
```
```bash
make run_daily CONFIG=configs/nba.yaml MODELS=glm_ridge
```

League-scoped data-only refresh:
```bash
make refresh-data CONFIG=configs/nhl.yaml
make refresh-data CONFIG=configs/nba.yaml
make refresh-data CONFIG=configs/ncaam.yaml
```

Deterministic repo-wide data-only refresh across all supported leagues:
```bash
make data_refresh
```
Preview the exact data-refresh step plan without executing it:
```bash
make data_refresh DRY_RUN=1
```

Deterministic repo-wide hard refresh across all supported leagues:
```bash
make hard_refresh
```
This retrains from the current processed feature snapshot, does not run `make features`, then commits the resulting repo changes, pushes `main` to `origin`, and watches the `Publish Sanitized Staging Site` GitHub Actions workflow. Start from a clean `main` worktree so the automated commit only contains refresh output.
Preview the exact step plan without executing it:
```bash
make hard_refresh DRY_RUN=1
```
Run the full refresh but skip the static Pages build:
```bash
make hard_refresh PAGES_BUILD=0
```

NBA rebuild with explicit research phase:
```bash
make fetch CONFIG=configs/nba.yaml
make features CONFIG=configs/nba.yaml
make research-features CONFIG=configs/nba.yaml APPROVE_FEATURE_CHANGES=1
make train CONFIG=configs/nba.yaml APPROVE_FEATURE_CHANGES=1
```

NCAAM build/train/validate flow:
```bash
make fetch CONFIG=configs/ncaam.yaml
make features CONFIG=configs/ncaam.yaml
make research-features CONFIG=configs/ncaam.yaml APPROVE_FEATURE_CHANGES=1
make compare-candidates CONFIG=configs/ncaam.yaml
make train CONFIG=configs/ncaam.yaml APPROVE_FEATURE_CHANGES=1
```

Launch dashboard:
```bash
make dashboard
LEAGUE=NBA NBA_DB_PATH=data/processed/nba_forecast.db make dashboard
```

Execution architecture:
- Atomic league-scoped commands stay independently runnable: `init-db`, `fetch`, `refresh-data`, `fetch-odds`, `features`, `research-features`, `train`, `backtest`, and `run_daily`.
- Model-level execution remains available through `MODELS=...`, for example `make train MODELS=glm_ridge`.
- The repo-wide ingest-only trigger is `make data_refresh`, which runs the deterministic NHL, NBA, then NCAAM data-collection sequence and stops before features, training, or staging snapshots.
- The repo-wide composite trigger is `make hard_refresh`, which runs a fixed sequential plan across NHL, NBA, then NCAAM, retrains from the existing processed feature snapshot without rebuilding features, regenerates `web/public/staging-data/`, optionally builds the static Pages output, then commits the refresh results, pushes `origin/main`, and watches the publish workflow for that pushed `HEAD`.

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
make query Q="Tell me how much money I won or lost from last night's games. both in total and by games."
make query Q="How much money did I win/lose last night?"
make query Q="How'd I do last night on my bets?"
make query Q="Recap my bets from yesterday."
make query Q="What are my cumulative net profits or losses since the beginning of tracking?"
make query Q="How much have I risked since the beginning of tracking?"
make query CONFIG=configs/nba.yaml Q="What's the chance the Raptors win their next game?"
make query Q="Which model has performed best the last 60 days?"
make query Q="Give me the report of all teams in a table."
```

Python entrypoint equivalent:
```bash
python3 -m src.query.answer --config configs/nba.yaml --question "Tell me how much money I won or lost from last night's games. both in total and by games."
python3 -m src.query.answer --config configs/nba.yaml --question "How much money did I win/lose last night?"
python3 -m src.query.answer --config configs/nba.yaml --question "How'd I do last night on my bets?"
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

Run the reusable Playwright dashboard smoke:
```bash
make smoke-dashboard
```
Equivalent manual commands:
```bash
cd web
npm run playwright:install
npm run test:smoke
```

Validation artifact layout:
- `artifacts/validation/<league>/` is the latest validation snapshot for that league.
- `artifacts/validation/<league>/split/` stores split metadata, including whether validation used `70/30` train/holdout or `40/30/30` train/validation/holdout, and whether the split was `time`-based or `random` by record.
- `artifacts/validation/<league>/glm/residuals/` stores GLM residual tables plus `plots/`.
- `artifacts/validation/<league>/diagnostics/` stores grouped classification, calibration, significance, stability, influence, fragility, nonlinearity, collinearity, and permutation-importance artifacts.
- `artifacts/validation/backtest/` stores repo-level backtest integrity and reliability tables.
- `artifacts/validation/bayes/offline/` stores offline Bayes diagnostics.
- `artifacts/validation-runs/<league>/YYYY-MM-DD/<timestamp>_<model_run_id>/` stores archived validation snapshots.

## Data + Temporal Integrity

- Uses public NHL (`api-web.nhle.com`) and NBA public roster/schedule/summary endpoints (`site.api.espn.com`) with retries and caching.
- Uses public NHL (`api-web.nhle.com`) plus NBA and NCAAM public roster/schedule/summary endpoints (`site.api.espn.com`) with retries and caching.
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
- Guardrail logs live at `configs/model_feature_guardrails_{league}.yaml`.
- `blocked_features` are enforced for the active map: research/save paths strip them out, and training/load fails fast if someone manually puts them back.
- `watchlist_features` and `watchlist_pairs` are the persistent notebook for validation findings such as multicollinearity and non-linearity that do not yet justify removal.
- The NBA and NCAAM paths use league-specific basketball feature builders and promoted per-model feature maps during training/backtesting.

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
- manifest-driven task pipeline so new validation directions can publish sections without growing `src/cli.py`
- train/validation/test split summaries plus holdout-first logistic validation outputs for NHL, NBA, and NCAAM
- blockwise nested-model deviance F-tests + information-criteria candidate tables
- coefficient stability paths, CV coefficient stability, bootstrap coefficient intervals, trade-deadline break test
- production multicollinearity suite: structural flags, pairwise correlation scan, VIF/tolerance, condition indices, variance decomposition, summary risk report
- production non-linearity suite: spline-vs-linear and hinge-vs-linear holdout comparisons, per-feature curve outputs, GAM/MARS guidance
- GLM residual diagnostics: deviance residual distribution/Q-Q checks, randomized quantile residuals, working residual plots, binned working residual summaries, weight plots, and partial residual plots for all active `glm_ridge` features
- influence diagnostics (leverage/Cook's/dfbetas + refit impact)
- calibration robustness + Brier decomposition + actual-vs-predicted, lift, Lorenz/Gini, ROC/AUROC, and threshold operating points
- fragility tests (missingness + perturbation)
- backtest integrity checks

## Notes

- Odds snapshots now persist to `odds_snapshots` and `odds_market_lines` on each `fetch` run (including bookmaker/market/outcome rows) for historical line tracking.
- Configure The Odds API via env vars: `ODDS_API_KEY`, `ODDS_API_REGIONS`, `ODDS_API_MARKETS`, `ODDS_API_ODDS_FORMAT`, `ODDS_API_DATE_FORMAT`, `ODDS_API_THROTTLE_SECONDS`.
- xG, injuries, and odds adapters include explicit graceful fallback behavior when stable no-auth feeds are unavailable.
- Bayesian state-space model runs in offline fit mode and supports daily sequential updates.
- Dashboard is visualization-only; no in-dashboard chat UI.
