`artifacts/validation` stores the latest local validation outputs that are not archived by date.

Layout:
- `mlb/`: canonical latest MLB validation snapshot lane for the rebuild
- `nba/` and `nhl/`: legacy latest league-scoped validation snapshots during migration
- `<league>/validation_manifest.json`: manifest index consumed by the validation API
- `<league>/validation_outputs_contract.json`: contract-backed validation task inventory
- `<league>/validation_run_metadata.json`: metadata for the latest archived snapshot linkage, including fixture/demo labels when the run is not production-grade
- `<league>/split/`: train/validation/holdout split metadata
- `<league>/glm/residuals/`: GLM residual tables and `plots/` for deviance, working-residual, and partial-residual diagnostics
- `<league>/diagnostics/`: grouped classification, calibration, collinearity, nonlinearity, significance, stability, influence, fragility, and permutation-importance artifacts
- `backtest/`: repo-level backtest integrity and reliability tables
- `bayes/offline/`: offline Bayes diagnostic outputs
- `_legacy/`: preserved historical files from older flat layouts

Historical validation snapshots do not live here. They are archived under `artifacts/validation-runs/`.

MLB fixture or demo validation runs must be labeled in `mlb/validation_run_metadata.json` with `execution_data_scope`, `execution_source_label`, and `execution_label_class` so staged artifacts cannot be mistaken for full production MLB coverage.
