# Query

This subsystem answers local NHL/NBA forecasting questions by intent.

Shared across leagues:
- `intent_parser.py` resolves intent and league/team targeting.
- `team_handlers.py`, `report_handlers.py`, and `championship_estimators.py` answer specific request types over the shared `Queryable` contract.
- `answer.py` is only the thin router and CLI entry point.

League-specific:
- `team_aliases.py` contains league/team naming rules.
- Championship naming and heuristic probability keys are sourced from `src/league_registry.py`, so league-specific labels stay isolated.

Input/output contract:
- Input is a `Queryable` database adapter plus a natural-language question and optional default league.
- Output is a deterministic `(answer_text, payload_dict)` pair whose payload is safe for downstream automation or dashboard use.
