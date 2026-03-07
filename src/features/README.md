# Features

This subsystem builds processed model features from interim league data.

Shared across leagues:
- `src/features/pipeline.py` owns the stable stages: interim loading, team-game expansion, rolling windows, game-level merge, and final frame persistence.
- Shared callers should enter through `src/features/build_features.py`.

League-specific:
- `src/features/strategies/nhl.py` contains hockey heuristics such as goalie, rink, and special-teams transforms.
- `src/features/strategies/nba.py` contains basketball heuristics such as availability, rotation, and arena effects.

Input/output contract:
- Input is an interim directory containing at least `games` plus league-appropriate `goalies`, `players`, and `injuries` tables.
- Output is a `FeatureBuildResult` with the finalized dataframe, numeric feature column list, feature-set version, and persisted artifact metadata.
