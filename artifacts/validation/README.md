`artifacts/validation` stores the latest local validation outputs that are not archived by date.

Layout:
- `mlb/`: canonical latest MLB validation snapshot lane for the rebuild
- `mlb/validation_manifest.json`: manifest index consumed by the validation API
- `mlb/validation_outputs_contract.json`: contract-backed validation task inventory
- `mlb/validation_run_metadata.json`: metadata for the latest archived snapshot linkage
- `mlb/split/`: train/validation/holdout split metadata
- `mlb/glm/residuals/`: GLM residual tables and `plots/` for deviance, working-residual, and partial-residual diagnostics
- `mlb/diagnostics/`: grouped classification, calibration, collinearity, nonlinearity, significance, stability, influence, fragility, and permutation-importance artifacts
- `backtest/`: repo-level backtest integrity and reliability tables
- `bayes/offline/`: offline Bayes diagnostic outputs
- `_legacy/`: preserved historical files from older flat layouts

Historical validation snapshots do not live here. They are archived under `artifacts/validation-runs/`.

Retired NBA/NHL validation snapshots are not part of the active working-tree
contract. Recover them from git history only if that historical scope is
explicitly requested.

MLB fixture, demo, or smoke validation runs must be labeled in
`mlb/validation_run_metadata.json` with `execution_data_scope`,
`execution_source_label`, `execution_label_class`, and `production_grade=false`.
Those outputs are never promotion evidence and must not update any
promotion-eligible latest pointer.
