"""NCAA men's basketball feature transforms for the shared pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.contextual_effects import compute_causal_group_effects
from src.features.dynamic_ratings import compute_dynamic_rating_features
from src.features.elo import compute_elo_features
from src.features.strategies.base import BaseFeatureStrategy


NCAAM_GLM_HINGE_KNOTS = {
    "diff_form_point_margin": 0.0,
    "diff_shot_volume_share": 0.01,
    "elo_home_prob": 0.55,
    "dyn_home_prob": 0.55,
}


def _positive_part(series: pd.Series, knot: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return (values - float(knot)).clip(lower=0.0)


def _compute_arena_effects(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(columns=["game_id", "arena_margin_effect", "arena_shot_volume_effect"])

    tmp = games_df.copy()
    tmp["point_margin"] = tmp["home_score"].fillna(0) - tmp["away_score"].fillna(0)
    tmp["shot_volume_diff"] = tmp["home_field_goal_attempts_for"].fillna(0) - tmp["away_field_goal_attempts_for"].fillna(0)
    return compute_causal_group_effects(
        tmp,
        group_col="venue",
        metric_columns={
            "arena_margin_effect": "point_margin",
            "arena_shot_volume_effect": "shot_volume_diff",
        },
        shrinkage=20.0,
    )


class NcaamFeatureStrategy(BaseFeatureStrategy):
    def __init__(self) -> None:
        super().__init__(
            league="NCAAM",
            summary_aggregations={
                "field_goal_attempts_for": ("shots_for", "max"),
                "field_goal_attempts_against": ("shots_against", "max"),
                "fouls_committed": ("penalties_taken", "max"),
                "fouls_drawn": ("penalties_drawn", "max"),
                "free_throws_made": ("pp_goals", "max"),
            },
            starter_aggregations={},
            team_value_for_column="points_for",
            team_value_against_column="points_against",
            team_result_column="won",
            rolling_value_columns=[
                "points_for",
                "points_against",
                "point_margin",
                "field_goal_attempts_for",
                "field_goal_attempts_against",
                "shot_volume_share",
                "free_throw_pressure",
                "foul_margin",
                "scoring_efficiency_proxy",
                "possession_proxy",
            ],
            diff_pairs=[
                ("ewm_point_margin", "form_point_margin"),
                ("win_rate_ewm", "form_win_rate"),
                ("ewm_shot_volume_share", "shot_volume_share"),
                ("ewm_scoring_efficiency_proxy", "scoring_efficiency"),
                ("ewm_foul_margin", "foul_margin"),
            ],
            direct_event_drop_columns=[
                "home_home_score",
                "home_away_score",
                "away_home_score",
                "away_away_score",
                "home_points_for",
                "away_points_for",
                "home_points_against",
                "away_points_against",
                "home_point_margin",
                "away_point_margin",
                "home_field_goal_attempts_for",
                "away_field_goal_attempts_for",
                "home_field_goal_attempts_against",
                "away_field_goal_attempts_against",
                "home_fouls_committed",
                "away_fouls_committed",
                "home_fouls_drawn",
                "away_fouls_drawn",
                "home_foul_margin",
                "away_foul_margin",
                "home_free_throws_made",
                "away_free_throws_made",
                "home_shot_volume_share",
                "away_shot_volume_share",
                "home_free_throw_pressure",
                "away_free_throw_pressure",
                "home_scoring_efficiency_proxy",
                "away_scoring_efficiency_proxy",
                "home_possession_proxy",
                "away_possession_proxy",
                "home_home_win",
                "away_home_win",
                "home_status_final",
                "away_status_final",
            ],
        )

    def prepare_team_games(self, team_games: pd.DataFrame, players_df: pd.DataFrame, injuries_df: pd.DataFrame) -> pd.DataFrame:
        del players_df, injuries_df

        df = team_games.sort_values(["team", "start_time_utc"]).copy()
        df["point_margin"] = df["points_for"].fillna(0) - df["points_against"].fillna(0)
        current_shot_profile_proxy_used = (
            pd.to_numeric(df.get("field_goal_attempts_for", pd.Series(index=df.index, dtype=float)), errors="coerce").isna()
            | pd.to_numeric(df.get("field_goal_attempts_against", pd.Series(index=df.index, dtype=float)), errors="coerce").isna()
            | pd.to_numeric(df.get("free_throws_made", pd.Series(index=df.index, dtype=float)), errors="coerce").isna()
        ).astype(int)
        df["field_goal_attempts_for"] = df["field_goal_attempts_for"].fillna(df["points_for"] * 0.82)
        df["field_goal_attempts_against"] = df["field_goal_attempts_against"].fillna(df["points_against"] * 0.82)
        df["fouls_committed"] = df["fouls_committed"].fillna(18)
        df["fouls_drawn"] = df["fouls_drawn"].fillna(18)
        df["free_throws_made"] = df["free_throws_made"].fillna(df["points_for"] * 0.16)
        df["shot_volume_share"] = df["field_goal_attempts_for"] / (
            df["field_goal_attempts_for"] + df["field_goal_attempts_against"]
        ).replace(0, np.nan)
        df["shot_volume_share"] = df["shot_volume_share"].fillna(0.5)
        df["free_throw_pressure"] = df["free_throws_made"] / df["field_goal_attempts_for"].replace(0, np.nan)
        df["free_throw_pressure"] = df["free_throw_pressure"].fillna(0.16)
        df["foul_margin"] = df["fouls_drawn"] - df["fouls_committed"]
        df["scoring_efficiency_proxy"] = df["points_for"] / df["field_goal_attempts_for"].replace(0, np.nan)
        df["scoring_efficiency_proxy"] = df["scoring_efficiency_proxy"].fillna(1.02)
        df["possession_proxy"] = df["field_goal_attempts_for"] + df["field_goal_attempts_against"]
        df["availability_uncertainty"] = 1.0
        df["availability_reliability"] = 0.0
        df["roster_depth_index"] = 0.0
        df["absence_load_proxy"] = 1.0
        df["shot_profile_proxy_used"] = current_shot_profile_proxy_used.groupby(df["team"], sort=False).shift(1).fillna(0).astype(int)
        return df

    def finalize_team_games(self, team_games: pd.DataFrame) -> pd.DataFrame:
        df = team_games.copy()
        season_dates = pd.to_datetime(df["game_date_utc"], errors="coerce")
        season_start = season_dates.min()
        df["days_into_season"] = (season_dates - season_start).dt.days.fillna(0)
        df["days_into_season_spline"] = np.sqrt(df["days_into_season"].clip(lower=0))
        df["season_phase"] = pd.cut(df["days_into_season"], bins=[-1, 30, 90, 220], labels=["early", "mid", "late"]).astype(str)
        df["season_phase_early"] = (df["season_phase"] == "early").astype(int)
        df["season_phase_mid"] = (df["season_phase"] == "mid").astype(int)
        df["season_phase_late"] = (df["season_phase"] == "late").astype(int)
        df["conference_tournament_window"] = season_dates.dt.month.ge(3).fillna(False).astype(int)
        df["shot_profile_proxy_used"] = (
            pd.to_numeric(df.get("shot_profile_proxy_used", pd.Series(index=df.index, dtype=float)), errors="coerce")
            .fillna(0)
            .clip(lower=0, upper=1)
            .astype(int)
        )
        return df

    def enrich_game_level(self, merged: pd.DataFrame, games_df: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
        del team_games

        out = merged.copy()
        for col in [
            "home_fouls_drawn",
            "away_fouls_drawn",
            "home_fouls_committed",
            "away_fouls_committed",
            "home_field_goal_attempts_for",
            "away_field_goal_attempts_for",
        ]:
            if col not in out.columns:
                out[col] = 0.0
            else:
                out[col] = out[col].fillna(0)

        out["discipline_free_throw_pressure_diff"] = out["home_ewm_free_throw_pressure"].fillna(0) - out["away_ewm_free_throw_pressure"].fillna(0)
        out["discipline_foul_margin_diff"] = out["home_ewm_foul_margin"].fillna(0) - out["away_ewm_foul_margin"].fillna(0)
        out["discipline_foul_pressure_diff"] = out["away_fouls_committed"].fillna(0) - out["home_fouls_committed"].fillna(0)
        out["target_shot_volume_share"] = (
            out["home_field_goal_attempts_for"] / (out["home_field_goal_attempts_for"] + out["away_field_goal_attempts_for"]).replace(0, np.nan)
        ).fillna(0.5)
        out["target_free_throw_pressure"] = out["home_free_throws_made"].fillna(0) - out["away_free_throws_made"].fillna(0)
        out["target_possession_volume"] = (out["home_field_goal_attempts_for"] + out["away_field_goal_attempts_for"]).fillna(0)
        out["rest_diff"] = out["home_rest_days"].fillna(7) - out["away_rest_days"].fillna(7)
        out["travel_diff"] = 0.0
        out["home_travel_miles"] = 0.0
        out["away_travel_miles"] = 0.0
        out["home_tz_change"] = 0.0
        out["away_tz_change"] = 0.0
        out["home_local_start_mismatch"] = 0.0
        out["away_local_start_mismatch"] = 0.0

        arena = _compute_arena_effects(out)
        out = out.merge(arena, on="game_id", how="left")
        out["arena_margin_effect"] = out["arena_margin_effect"].fillna(0)
        out["arena_shot_volume_effect"] = out["arena_shot_volume_effect"].fillna(0)

        elo = compute_elo_features(games_df)
        dyn = compute_dynamic_rating_features(games_df)
        return out.merge(elo, on="game_id", how="left").merge(dyn, on="game_id", how="left")

    def add_model_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "elo_home_prob" in out.columns:
            out["elo_home_prob_hinge_055"] = _positive_part(out["elo_home_prob"], NCAAM_GLM_HINGE_KNOTS["elo_home_prob"])
        if "dyn_home_prob" in out.columns:
            out["dyn_home_prob_hinge_055"] = _positive_part(out["dyn_home_prob"], NCAAM_GLM_HINGE_KNOTS["dyn_home_prob"])
        if "diff_shot_volume_share" in out.columns:
            out["diff_shot_volume_share_hinge_001"] = _positive_part(out["diff_shot_volume_share"], NCAAM_GLM_HINGE_KNOTS["diff_shot_volume_share"])
        if "diff_form_point_margin" in out.columns:
            out["diff_form_point_margin_hinge_000"] = _positive_part(out["diff_form_point_margin"], NCAAM_GLM_HINGE_KNOTS["diff_form_point_margin"])
        return out

    def feature_hash_payload(self, feature_columns: list[str]) -> dict:
        return {"league": self.league, "n_features": len(feature_columns), "cols": feature_columns}
