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

Train model suite + ensemble + upcoming forecasts + validation artifacts:
```bash
make train
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

Backtest only selected models:
```bash
make backtest MODELS=glm_logit
```

Daily end-to-end:
```bash
make run_daily
```

Daily run with selected models:
```bash
make run_daily MODELS=glm_logit
```
```bash
make run_daily CONFIG=configs/nba.yaml MODELS=glm_logit
```

Launch dashboard:
```bash
make dashboard
LEAGUE=NBA NBA_DB_PATH=data/processed/nba_forecast.db make dashboard
```

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

- xG, injuries, and odds adapters include explicit graceful fallback behavior when stable no-auth feeds are unavailable.
- Bayesian state-space model runs in offline fit mode and supports daily sequential updates.
- Dashboard is visualization-only; no in-dashboard chat UI.
