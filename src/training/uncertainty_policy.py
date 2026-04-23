"""MLB uncertainty flags for dashboard payloads."""

from __future__ import annotations

import json

import pandas as pd


def _float_or_default(value: object, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if pd.notna(numeric) else default


def build_uncertainty_flags(upcoming_df: pd.DataFrame) -> list[str]:
    flags = []
    for _, row in upcoming_df.iterrows():
        home_starter_uncertainty = _float_or_default(row.get("home_starting_pitcher_uncertainty", row.get("home_starter_unknown", 1)), 1.0)
        away_starter_uncertainty = _float_or_default(row.get("away_starting_pitcher_uncertainty", row.get("away_starter_unknown", 1)), 1.0)
        game_flags = {
            "starter_unknown": bool(max(home_starter_uncertainty, away_starter_uncertainty) > 0),
            "lineup_uncertainty": bool(_float_or_default(row.get("lineup_uncertainty", 1), 1.0) > 0),
            "weather_unavailable": bool(row.get("fallback_weather_proxy_used", 0) == 1),
        }
        flags.append(json.dumps(game_flags, sort_keys=True))
    return flags
