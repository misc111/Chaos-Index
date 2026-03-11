"""Shared contract for immutable pregame prediction history.

`predictions` is the frozen ledger of forecasts that actually existed before a
game started. Synthetic rows like OOF diagnostics or walk-forward backtests
must never live there because user-facing history, scoring, and replay screens
must reflect what the model knew at the time, not what a later retrain can
reconstruct.
"""

from __future__ import annotations

FROZEN_PREDICTION_SOURCE = "train_upcoming"
DIAGNOSTIC_PREDICTION_SOURCES = (
    "train_oof_history",
    "walk_forward_backtest",
)


def frozen_prediction_source_sql(alias: str = "p") -> str:
    return f"COALESCE(json_extract({alias}.metadata_json, '$.source'), '') = '{FROZEN_PREDICTION_SOURCE}'"
