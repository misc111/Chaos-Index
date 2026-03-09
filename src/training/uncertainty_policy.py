"""League-aware uncertainty flags for dashboard payloads.

Shared training orchestration should ask this module for public uncertainty
flags rather than embedding league heuristics inline.
"""

from __future__ import annotations

import json

import pandas as pd


# Temporary NBA public-flag threshold, added after the March 8, 2026 audit:
# the raw availability feature still carries signal, but the old "sum > 0"
# rule was effectively turning the public warning into an always-on badge. That
# made it useless for user trust and postmortem work. Revisit this threshold
# after we accumulate a meaningful post-fix sample and re-check the live
# distribution. If future feature engineering changes move the uncertainty
# scale, re-measure this threshold instead of treating 0.35 as permanent.
NBA_PUBLIC_AVAILABILITY_UNCERTAINTY_THRESHOLD = 0.35


def _float_or_default(value: object, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if pd.notna(numeric) else default


def build_uncertainty_flags(upcoming_df: pd.DataFrame) -> list[str]:
    flags = []
    nba_style_flags = "home_availability_uncertainty" in upcoming_df.columns or "fallback_shot_profile_proxy_used" in upcoming_df.columns
    for _, row in upcoming_df.iterrows():
        if nba_style_flags:
            home_uncertainty = _float_or_default(row.get("home_availability_uncertainty", 1), 1.0)
            away_uncertainty = _float_or_default(row.get("away_availability_uncertainty", 1), 1.0)
            game_flags = {
                "availability_uncertainty": bool(
                    max(home_uncertainty, away_uncertainty) >= NBA_PUBLIC_AVAILABILITY_UNCERTAINTY_THRESHOLD
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
