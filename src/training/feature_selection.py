"""Feature selection helpers shared by fit, predict, and backtest runners.

These functions define the contract between the feature pipeline and the model
catalog. League-specific feature engineering may change upstream, but model
selection rules should stay centralized here.
"""

from __future__ import annotations

import re

import pandas as pd

from src.training.model_catalog import LEGACY_MODEL_KEYS


RESERVED_NON_FEATURES = {
    "game_id",
    "season",
    "game_date_utc",
    "start_time_utc",
    "home_team",
    "away_team",
    "venue",
    "status_final",
    "home_win",
    "as_of_utc",
    "home_score",
    "away_score",
}


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    banned_exact = {
        "home_goals_for",
        "away_goals_for",
        "home_goals_against",
        "away_goals_against",
        "home_goal_diff",
        "away_goal_diff",
        "home_points_for",
        "away_points_for",
        "home_points_against",
        "away_points_against",
        "home_point_margin",
        "away_point_margin",
        "home_home_score",
        "home_away_score",
        "away_home_score",
        "away_away_score",
        "home_home_win",
        "away_home_win",
        "home_status_final",
        "away_status_final",
    }
    lag_markers = ("ewm_", "r5_", "r14_")
    direct_event_tokens = (
        "goals_for",
        "goals_against",
        "points_for",
        "points_against",
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
        "starter_save_pct",
        "goalie_quality_raw",
        "team_save_pct_proxy",
        "xg_share_proxy",
        "penalty_diff_proxy",
        "pace_proxy",
        "scoring_efficiency_proxy",
        "possession_proxy",
    )

    cols = []
    for c in df.columns:
        if c in RESERVED_NON_FEATURES:
            continue
        if c.startswith("target_"):
            continue
        if c in banned_exact:
            continue
        if re.search(r"(^|_)home_score($|_)|(^|_)away_score($|_)|(^|_)status_final($|_)", c):
            continue
        if "home_win" in c and "win_rate" not in c:
            continue
        if ("goals_for" in c or "goals_against" in c or "points_for" in c or "points_against" in c) and not any(
            m in c for m in lag_markers
        ):
            continue
        if any(tok in c for tok in direct_event_tokens) and not any(m in c for m in lag_markers):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def bayes_feature_subset(feature_cols: list[str]) -> list[str]:
    keep = []
    keywords = [
        "diff_",
        "travel",
        "rest",
        "goalie",
        "special",
        "discipline",
        "availability",
        "rink",
        "arena",
        "elo",
        "dyn",
        "lineup",
    ]
    for c in feature_cols:
        if any(k in c for k in keywords):
            keep.append(c)
    if len(keep) < 12:
        keep = feature_cols[: min(40, len(feature_cols))]
    return keep


def glm_feature_subset(feature_cols: list[str]) -> list[str]:
    keep_exact = {
        "travel_diff",
        "rest_diff",
        "rink_goal_effect",
        "rink_shot_effect",
        "arena_margin_effect",
        "arena_shot_volume_effect",
    }
    keep_prefix = ("diff_", "special_", "discipline_", "goalie_", "availability_", "elo_", "dyn_")

    out = [
        c
        for c in feature_cols
        if (c.startswith(keep_prefix) or c in keep_exact) and not c.endswith("_goalie_id")
    ]

    if len(out) < 12:
        banned = {"home_season", "away_season", "home_is_home", "away_is_home"}
        out = [c for c in feature_cols if c not in banned and not c.endswith("_goalie_id")]
    return out


def resolve_model_feature_columns(
    all_feature_cols: list[str],
    *,
    model_name: str,
    model_feature_columns: dict[str, list[str]] | None,
    fallback_columns: list[str],
) -> list[str]:
    if not model_feature_columns:
        return list(fallback_columns)

    requested_key = model_name
    requested = model_feature_columns.get(requested_key, [])
    if not requested:
        for legacy_key in LEGACY_MODEL_KEYS.get(model_name, ()):
            requested = model_feature_columns.get(legacy_key, [])
            if requested:
                requested_key = legacy_key
                break
    if not requested:
        return list(fallback_columns)

    missing = [c for c in requested if c not in all_feature_cols]
    if missing:
        raise ValueError(f"model_feature_columns[{requested_key}] includes missing columns: {missing}")
    return list(requested)
