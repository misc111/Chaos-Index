from __future__ import annotations

import re

import pandas as pd


BANNED_DIRECT_FEATURES = {
    "home_score",
    "away_score",
    "goal_diff",
    "home_win",
}

BANNED_PATTERN_RULES = [
    r"(^|_)home_score($|_)",
    r"(^|_)away_score($|_)",
    r"(^|_)status_final($|_)",
    r"(^|_)home_goal_diff($|_)",
    r"(^|_)away_goal_diff($|_)",
    r"(^|_)home_home_win($|_)",
    r"(^|_)away_home_win($|_)",
]

ALLOWED_HISTORICAL_GOALS_MARKERS = ("ewm_", "r5_", "r14_")
DIRECT_EVENT_TOKENS = (
    "goals_for",
    "goals_against",
    "shots_for",
    "shots_against",
    "penalties_taken",
    "penalties_drawn",
    "pp_goals",
    "starter_save_pct",
    "goalie_quality_raw",
    "team_save_pct_proxy",
    "xg_share_proxy",
    "penalty_diff_proxy",
    "pace_proxy",
)



def run_leakage_checks(features_df: pd.DataFrame, feature_columns: list[str] | None = None) -> list[str]:
    issues: list[str] = []
    if features_df.empty:
        issues.append("features_empty")
        return issues

    cols = set(feature_columns) if feature_columns is not None else set(features_df.columns)
    forbidden_present = sorted(BANNED_DIRECT_FEATURES.intersection(cols))
    if forbidden_present:
        issues.append(f"forbidden_columns_present={forbidden_present}")

    pattern_forbidden = []
    for c in cols:
        if any(re.search(rule, c) for rule in BANNED_PATTERN_RULES):
            pattern_forbidden.append(c)
            continue
        if ("goals_for" in c or "goals_against" in c) and not any(m in c for m in ALLOWED_HISTORICAL_GOALS_MARKERS):
            pattern_forbidden.append(c)
            continue
        if any(tok in c for tok in DIRECT_EVENT_TOKENS) and not any(m in c for m in ALLOWED_HISTORICAL_GOALS_MARKERS):
            pattern_forbidden.append(c)
            continue
        if "home_win" in c and "win_rate" not in c:
            pattern_forbidden.append(c)
            continue
    if pattern_forbidden:
        issues.append(f"pattern_forbidden_columns={sorted(pattern_forbidden)[:30]}")

    if "status_final" in features_df.columns and "home_win" in features_df.columns:
        nonfinal = features_df[(features_df["status_final"] == 0) & features_df["home_win"].notna()]
        if not nonfinal.empty:
            issues.append(f"nonfinal_has_outcome_count={len(nonfinal)}")

    if "home_games_played_prior" in features_df.columns:
        if (features_df["home_games_played_prior"] < 0).any():
            issues.append("negative_home_games_played_prior")

    if "away_games_played_prior" in features_df.columns:
        if (features_df["away_games_played_prior"] < 0).any():
            issues.append("negative_away_games_played_prior")

    return issues
