# Query

This subsystem answers local forecasting questions by intent, with the active rebuild contract centered on MLB.

MLB product lane:
- `intent_parser.py` resolves intent and MLB team targeting.
- `team_handlers.py`, `report_handlers.py`, and `championship_estimators.py` answer specific request types over the shared `Queryable` contract.
- `answer.py` is only the thin router and CLI entry point.

Legacy/comparison boundary:
- `team_aliases.py` contains MLB naming rules and empty NBA/NHL compatibility shims so retired leagues do not re-enter product query routing.
- Championship naming and heuristic probability keys are sourced from `src/league_registry.py`, so MLB product labels stay isolated.

Input/output contract:
- Input is a `Queryable` database adapter plus a natural-language question and optional default league.
- Output is a deterministic `(answer_text, payload_dict)` pair whose payload is safe for downstream automation or dashboard use.
