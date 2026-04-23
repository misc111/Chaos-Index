from __future__ import annotations

import re

import pandas as pd


BANNED_DIRECT_FEATURES = {
    "home_score",
    "away_score",
    "goal_diff",
    "point_margin",
    "home_win",
}

BANNED_PATTERN_RULES = [
    r"(^|_)home_score($|_)",
    r"(^|_)away_score($|_)",
    r"(^|_)status_final($|_)",
    r"(^|_)home_goal_diff($|_)",
    r"(^|_)away_goal_diff($|_)",
    r"(^|_)home_point_margin($|_)",
    r"(^|_)away_point_margin($|_)",
    r"(^|_)home_home_win($|_)",
    r"(^|_)away_home_win($|_)",
]

ALLOWED_HISTORICAL_GOALS_MARKERS = ("ewm_", "r5_", "r14_", "form_", "diff_form_")
ALLOWED_PREGAME_STARTER_DIFF_PREFIXES = (
    "diff_starter_",
    "diff_form_starter_",
)
DIRECT_EVENT_TOKENS = (
    "goals_for",
    "goals_against",
    "points_for",
    "points_against",
    "runs_for",
    "runs_against",
    "run_diff",
    "home_runs",
    "away_runs",
    "total_runs",
    "shots_for",
    "shots_against",
    "field_goal_attempts_for",
    "field_goal_attempts_against",
    "penalties_taken",
    "penalties_drawn",
    "fouls_committed",
    "fouls_drawn",
    "pp_goals",
    "free_throws_made",
    "starter_innings_pitched",
    "starter_era",
    "starter_whip",
    "starter_pitcher_strikeouts",
    "starter_pitcher_walks",
    "starter_runs_allowed",
    "starter_save_pct",
    "goalie_quality_raw",
    "team_save_pct_proxy",
    "xg_share_proxy",
    "penalty_diff_proxy",
    "pace_proxy",
    "scoring_efficiency_proxy",
    "possession_proxy",
    "winner",
)



def run_leakage_checks(features_df: pd.DataFrame, feature_columns: list[str] | None = None) -> list[str]:
    issues: list[str] = []
    if features_df.empty:
        issues.append("features_empty")
        return issues

    if "available_as_of_utc" in features_df.columns and "start_time_utc" in features_df.columns:
        available = pd.to_datetime(features_df["available_as_of_utc"], errors="coerce", utc=True)
        start = pd.to_datetime(features_df["start_time_utc"], errors="coerce", utc=True)
        bad = available.notna() & start.notna() & available.gt(start)
        if bad.any():
            issues.append(f"available_as_of_after_start_count={int(bad.sum())}")

    cols = set(feature_columns) if feature_columns is not None else set(features_df.columns)
    forbidden_present = sorted(BANNED_DIRECT_FEATURES.intersection(cols))
    if forbidden_present:
        issues.append(f"forbidden_columns_present={forbidden_present}")

    pattern_forbidden = []
    for c in cols:
        if any(re.search(rule, c) for rule in BANNED_PATTERN_RULES):
            pattern_forbidden.append(c)
            continue
        if ("goals_for" in c or "goals_against" in c or "points_for" in c or "points_against" in c) and not any(
            m in c for m in ALLOWED_HISTORICAL_GOALS_MARKERS
        ):
            pattern_forbidden.append(c)
            continue
        if (
            any(tok in c for tok in DIRECT_EVENT_TOKENS)
            and not any(m in c for m in ALLOWED_HISTORICAL_GOALS_MARKERS)
            and not c.startswith(ALLOWED_PREGAME_STARTER_DIFF_PREFIXES)
        ):
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
