`artifacts/validation-runs` stores archived validation snapshots by league and run date.

Layout:
- `nba/YYYY-MM-DD/<timestamp>_<model_run_id>/`
- `nhl/YYYY-MM-DD/<timestamp>_<model_run_id>/`

Each archived run contains:
- the full contents of `artifacts/validation/<league>/`
- `plots/` for GLM residual and partial-residual diagnostics
- `performance/` for holdout summary curves copied from `artifacts/plots/<league>/glm/performance/`
- `validation_run_metadata.json` with archive metadata and a full artifact inventory

The active latest-view locations remain:
- `artifacts/validation/<league>/`
- `artifacts/plots/<league>/glm/performance/`

The archive tree is intentionally git-ignored because validation runs can be large and high-churn.
