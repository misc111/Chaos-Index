# Weather Context Broad Platt Gate Autopsy

Date of reviewed run: April 29, 2026 (`real_all_target_market_calibration_smoke_20260428T0000Z`)

## Subject

Moneyline family champion:

- `target_name`: `moneyline_home_win`
- `model_name`: `glm_ridge`
- `variant_key`: `weather_context_broad_platt_calibrated`
- `candidate_key`: `moneyline_home_win::glm_ridge::weather_context_broad_platt_calibrated`
- Governance lane: direct theory implementation for ridge/logit scoring and calibration diagnostics; theory-compatible engineering support for structured slate guardrails.

Primary evidence:

- `artifacts/reports/mlb/tournament/nested/real_all_target_market_calibration_smoke_20260428T0000Z/family_champions.csv`
- `artifacts/reports/mlb/tournament/nested/real_all_target_market_calibration_smoke_20260428T0000Z/validation_autopsy/family_champion_gate_autopsy.csv`
- `artifacts/reports/mlb/tournament/nested/real_all_target_market_calibration_smoke_20260428T0000Z/validation_autopsy/validation_autopsy.md`
- `artifacts/reports/mlb/tournament/nested/real_all_target_market_calibration_smoke_20260428T0000Z/diagnostics/moneyline_home_win/validation/glm_ridge/weather_context_broad_platt_calibrated/`
- `web/public/staging-data/mlb/nested-tournament.json`

## Gate Failure

The model is `validation-blocked` with two gate reasons:

| Gate | Observed | Threshold | Status |
| --- | ---: | ---: | --- |
| Calibration intercept | `0.348176` | `abs(alpha) <= 0.15` | fail |
| Calibration slope | `-0.019015` | `abs(beta - 1.0) <= 0.15` | fail |
| CV-validation log-loss drift | `-0.069629` | `abs(delta) <= 0.05` | fail |
| ECE / quantile mean abs gap | `0.032341` | `<= 0.04` | pass |
| AUROC | `0.509483` | `>= 0.50` | pass, but weak |
| Top-vs-bottom lift ratio | `1.071429` | `> 1.0` | pass, but weak |

The quantile and calibration artifacts show the model is not failing because Platt calibration did nothing. Platt reduced the broad weather-context ECE from the raw row's `0.137403` to `0.032341`, but it did so by compressing predictions into a narrow band of roughly `0.5944` to `0.6418`.

The Lorenz and ROC artifacts show the remaining signal is too weak for promotion:

- normalized Gini: `0.027106`
- AUROC: `0.509483`
- best Youden accuracy: `0.478992`
- at threshold `0.50`, predicted positive rate is `1.0`, sensitivity is `1.0`, and specificity is `0.0`
- quantile bins realized: `7` of `10`
- quantile monotonicity violations: `3`

## Root Cause

The variant name overstates the actual retained slate. In the current processed data, the configured `weather_context.broad` slate resolves to only retained rest/travel aliases:

- `rest_edge` retained as `rest_diff`
- `travel_miles_edge` retained as `travel_diff`

The actual weather/run-environment features in the configured slate are absent, missing, or screened out for the fit window. That means `weather_context_broad_platt_calibrated` is not a real weather-context ridge variant in this run; it is a thin scheduling-context ridge variant plus a tail Platt calibrator.

## Targeted Redesign

The fix is to guard structured weather-context variants at construction time instead of loosening validation gates.

The weather-context slate now requires:

- at least four retained features
- at least 50% of the requested width retained
- at least one real run-environment/context feature from `park_run_factor`, `temperature_f`, `wind_out_mph`, `day_game`, or `division_game`

This prevents an under-retained weather slate from becoming a family champion simply because Platt calibration improves log loss on a narrow probability band.

## Proof Criteria

A redesigned weather-context ridge family can be considered for validation only if a fresh nested tournament run shows:

- the guarded weather-context variant is actually emitted with the required retained context features
- leakage checks pass before target enrichment and after market/target enrichment
- `fit_status=ok` and class contrast is present
- quantile ECE / mean absolute calibration gap is `<= 0.04`
- `abs(calibration_alpha) <= 0.15`
- `abs(calibration_beta - 1.0) <= 0.15`
- AUROC is `>= 0.50`, with Lorenz/Gini and quantile lift not merely coin-flip by inspection
- top-vs-bottom actual lift ratio is `> 1.0`
- `abs(validation_to_cv_log_loss_delta) <= 0.05`
- final-holdout evidence is produced only after the family champion clears validation

Until those conditions are met, this family should remain `validation-blocked` or omitted by availability guardrails, not promoted through betting overlay metrics.
