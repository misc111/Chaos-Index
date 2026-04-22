`artifacts/plots` stores lightweight, league-scoped summary plots.

Current layout:
- `mlb/glm/performance/`: holdout summary curves for the active MLB GLM validation run
- `_legacy/`: historical plot outputs preserved from older layouts

Canonical full diagnostic suites live under:
- `artifacts/validation/mlb/glm/residuals/plots/`
- archived snapshots of those validation suites live under `artifacts/validation-runs/`

Those validation plot directories are the source of truth for:
- GLM residual diagnostics
- working residual plots
- partial residual plots
- contact sheets and other validation-specific visuals
