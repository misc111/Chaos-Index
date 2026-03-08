`artifacts/validation` stores the latest local validation outputs that are not archived by date.

Layout:
- `nba/` and `nhl/`: latest league-scoped validation snapshots
- `nba/validation_manifest.json` and `nhl/validation_manifest.json`: manifest index consumed by the validation API
- `nba/validation_run_metadata.json` and `nhl/validation_run_metadata.json`: metadata for the latest archived snapshot linkage
- `<league>/split/`: train/validation/holdout split metadata
- `<league>/glm/residuals/`: GLM residual tables and `plots/` for deviance, working-residual, and partial-residual diagnostics
- `<league>/diagnostics/`: grouped classification, calibration, collinearity, nonlinearity, significance, stability, influence, fragility, and permutation-importance artifacts
- `backtest/`: repo-level backtest integrity and reliability tables
- `bayes/offline/`: offline Bayes diagnostic outputs
- `_legacy/`: preserved historical files from older flat layouts

Historical validation snapshots do not live here. They are archived under `artifacts/validation-runs/`.
