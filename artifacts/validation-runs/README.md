`artifacts/validation-runs` stores archived validation snapshots by league and run date.

Layout:
- `mlb/YYYY-MM-DD/<timestamp>_<model_run_id>/`
- `<league>/YYYY-MM-DD/<timestamp>_<model_run_id>/` for any retained legacy league snapshots

Each archived run contains:
- the full contents of `artifacts/validation/<league>/`
- `glm/residuals/plots/` for GLM residual and partial-residual diagnostics
- `performance/` for holdout summary curves copied from `artifacts/plots/<league>/glm/performance/`
- `validation_run_metadata.json` with archive metadata and a full artifact inventory

The active latest-view locations remain:
- `artifacts/validation/<league>/`
- `artifacts/plots/<league>/glm/performance/`

The archive tree is intentionally git-ignored because validation runs can be large and high-churn.
