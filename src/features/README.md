# Features

This subsystem builds processed MLB model features from interim data.

Shared feature pipeline:
- `src/features/pipeline.py` owns the stable stages: interim loading, team-game expansion, rolling windows, game-level merge, and final frame persistence.
- Callers should enter through `src/features/build_features.py`.

MLB-specific:
- `src/features/strategies/mlb.py` contains baseball heuristics such as starting-pitcher, bullpen, lineup, park, weather, and schedule transforms.

Input/output contract:
- Input is an interim directory containing at least `games` plus MLB support tables such as `team_stats`, `starting_pitchers`, `weather`, `players`, and `injuries`.
- Output is a `FeatureBuildResult` with the finalized dataframe, numeric feature column list, feature-set version, and persisted artifact metadata.
