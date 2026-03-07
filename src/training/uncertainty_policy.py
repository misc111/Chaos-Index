"""League-aware uncertainty flags for dashboard payloads.

Shared training orchestration should ask this module for public uncertainty
flags rather than embedding league heuristics inline.
"""

from __future__ import annotations

import json

import pandas as pd


def build_uncertainty_flags(upcoming_df: pd.DataFrame) -> list[str]:
    flags = []
    nba_style_flags = "home_availability_uncertainty" in upcoming_df.columns or "fallback_shot_profile_proxy_used" in upcoming_df.columns
    for _, row in upcoming_df.iterrows():
        if nba_style_flags:
            game_flags = {
                "availability_uncertainty": bool(
                    (row.get("home_availability_uncertainty", 1) + row.get("away_availability_uncertainty", 1)) > 0
                ),
                "shot_profile_proxy_used": bool(row.get("fallback_shot_profile_proxy_used", 1) == 1),
                "availability_proxy_used": bool(row.get("fallback_availability_proxy_used", 1) == 1),
            }
        else:
            game_flags = {
                "starter_unknown": bool((row.get("home_goalie_uncertainty_feature", 1) + row.get("away_goalie_uncertainty_feature", 1)) > 0),
                "xg_unavailable": bool(row.get("fallback_xg_proxy_used", 1) == 1),
                "lineup_uncertainty": bool((row.get("home_lineup_uncertainty", 1) + row.get("away_lineup_uncertainty", 1)) > 0),
            }
        flags.append(json.dumps(game_flags, sort_keys=True))
    return flags
