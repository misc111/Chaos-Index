# Data Sources

This subsystem owns MLB ingestion adapters and nothing else.

Shared ingest boundary:
- The fetch lifecycle is orchestrated by `src/services/ingest.py` through the typed `LeagueAdapter` in `src/league_registry.py`.
- The MLB package implements games, team metadata, players, injuries, optional odds, schedule, results shaping, team stats, starting pitchers, and weather scaffolds.

MLB-specific:
- `src/data_sources/mlb/` contains baseball-only fetch logic such as lineup cards, starting pitchers, bullpen/team boxscore stats, and weather scaffolds.

Input/output contract:
- Inputs are HTTP clients plus MLB fetch arguments supplied by orchestration.
- Outputs are `SourceFetchResult` payloads or pandas frames that can be persisted by the ingest service without knowing source internals.
