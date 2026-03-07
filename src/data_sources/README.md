# Data Sources

This subsystem owns league-specific ingestion adapters and nothing else.

Shared across leagues:
- The fetch lifecycle is orchestrated by `src/services/ingest.py` through the typed `LeagueAdapter` in `src/league_registry.py`.
- Every league package is expected to implement the same responsibilities: games, team metadata, player/lineup proxies, injuries, optional odds, optional xG, schedule, and results shaping.

League-specific:
- `src/data_sources/nhl/` contains hockey-only fetch logic such as goalie and xG inputs.
- `src/data_sources/nba/` contains basketball-only fetch logic such as availability and market-oriented inputs.

Input/output contract:
- Inputs are HTTP clients plus league-specific fetch arguments supplied by shared orchestration.
- Outputs are `SourceFetchResult` payloads or pandas frames that can be persisted by the ingest service without knowing league internals.
