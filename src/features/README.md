# Features

This subsystem builds processed model features from interim league data.

Shared across leagues:
- `src/features/pipeline.py` owns the stable stages: interim loading, team-game expansion, rolling windows, game-level merge, and final frame persistence.
- Shared callers should enter through `src/features/build_features.py`.
- Root-level modules in `src/features/` should stay genuinely cross-league. League-only transforms live under an explicit boundary such as `src/features/nhl/`.

League-specific:
- `src/features/strategies/mlb.py` contains baseball heuristics such as starting-pitcher, bullpen, lineup, park, weather, and schedule transforms.
- `src/features/strategies/nhl.py` contains hockey heuristics such as goalie, rink, and special-teams transforms, with the underlying helpers isolated in `src/features/nhl/`.
- `src/features/strategies/nba.py` contains basketball heuristics such as availability, rotation, and arena effects.

Input/output contract:
- Input is an interim directory containing at least `games` plus league-appropriate support tables such as `team_stats`/`starting_pitchers`/`weather` for MLB or `goalies`/`xg` for legacy NHL-style lanes, alongside `players` and `injuries`. The `goalies` and `xg` names are legacy NHL data aliases, not shared feature concepts for the package as a whole.
- Output is a `FeatureBuildResult` with the finalized dataframe, numeric feature column list, feature-set version, and persisted artifact metadata.
