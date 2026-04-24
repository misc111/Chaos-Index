"""MLB-specific feature transforms for the shared pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.dynamic_ratings import compute_dynamic_rating_features
from src.features.elo import compute_elo_features
from src.features.strategies.base import BaseFeatureStrategy
from src.features.travel import build_travel_features


MLB_GLM_HINGE_KNOTS = {
    "diff_form_run_diff": 0.0,
    "diff_starter_era": 0.0,
    "elo_home_prob": 0.52,
}


def _positive_part(series: pd.Series, knot: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return (values - float(knot)).clip(lower=0.0)


def _numeric_column(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in df.columns:
        values = df[column]
    else:
        values = pd.Series(default, index=df.index)
    return pd.to_numeric(values, errors="coerce")


def _text_column(df: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column in df.columns:
        values = df[column]
    else:
        values = pd.Series(default, index=df.index)
    return values.astype(str)


def _lineup_meta(players_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["game_id", "team", "lineup_card_size", "lineup_availability", "lineup_confirmed"]
    required = {"game_id", "team", "player_id", "batting_order_slot"}
    if players_df.empty or not required.issubset(players_df.columns):
        return pd.DataFrame(columns=columns)

    players = players_df.copy()
    players["batting_order_slot"] = pd.to_numeric(players["batting_order_slot"], errors="coerce")
    players = players[players["game_id"].notna() & players["team"].notna() & players["batting_order_slot"].notna()].copy()
    if players.empty:
        return pd.DataFrame(columns=columns)

    players["lineup_confirmed"] = _numeric_column(players, "lineup_confirmed", 0).fillna(0).clip(lower=0, upper=1)
    lineup = (
        players.sort_values(["game_id", "team", "batting_order_slot"])
        .groupby(["game_id", "team"], as_index=False)
        .agg(
            lineup_card_size=("player_id", "nunique"),
            lineup_confirmed=("lineup_confirmed", "max"),
        )
    )
    lineup["lineup_card_size"] = lineup["lineup_card_size"].clip(lower=0, upper=9)
    lineup["lineup_availability"] = (lineup["lineup_card_size"] / 9.0).clip(lower=0.0, upper=1.0)
    return lineup[columns]


def _injury_meta(injuries_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["game_id", "team", "position_player_out_count", "pitcher_out_count", "bullpen_availability"]
    required = {"game_id", "team"}
    if injuries_df.empty or not required.issubset(injuries_df.columns):
        return pd.DataFrame(columns=columns)

    injuries = injuries_df.copy()
    injuries["position_player_out_count"] = _numeric_column(injuries, "position_player_out_count", 0).fillna(0).clip(lower=0)
    injuries["pitcher_out_count"] = _numeric_column(injuries, "pitcher_out_count", 0).fillna(0).clip(lower=0)
    grouped = (
        injuries.groupby(["game_id", "team"], as_index=False)
        .agg(
            position_player_out_count=("position_player_out_count", "sum"),
            pitcher_out_count=("pitcher_out_count", "sum"),
        )
    )
    grouped["bullpen_availability"] = (1.0 / (1.0 + grouped["pitcher_out_count"])).clip(lower=0.0, upper=1.0)
    return grouped[columns]


class MlbFeatureStrategy(BaseFeatureStrategy):
    def __init__(self) -> None:
        super().__init__(
            league="MLB",
            team_stats_artifact_name="team_stats",
            starter_context_artifact_name="starting_pitchers",
            players_artifact_name="players",
            injuries_artifact_name="injuries",
            context_metrics_artifact_name="weather",
            summary_aggregations={
                "hits": ("hits", "max"),
                "walks": ("walks", "max"),
                "strikeouts": ("strikeouts", "max"),
                "home_runs": ("home_runs", "max"),
                "total_bases": ("total_bases", "max"),
                "stolen_bases": ("stolen_bases", "max"),
            },
            starter_aggregations={
                "starter_pitcher_id": ("starter_pitcher_id", "first"),
                "starter_pitcher_name": ("starter_pitcher_name", "first"),
                "starter_status": ("starter_status", "first"),
                "starter_innings_pitched": ("innings_pitched", "mean"),
                "starter_era": ("starter_era", "mean"),
                "starter_whip": ("starter_whip", "mean"),
                "starter_pitcher_strikeouts": ("pitcher_strikeouts", "mean"),
                "starter_pitcher_walks": ("pitcher_walks", "mean"),
                "starter_runs_allowed": ("runs_allowed", "mean"),
            },
            team_value_for_column="runs_for",
            team_value_against_column="runs_against",
            team_result_column="won",
            rolling_value_columns=[
                "runs_for",
                "runs_against",
                "run_diff",
                "hits",
                "walks",
                "strikeouts",
                "home_runs",
                "total_bases",
                "starter_era",
                "starter_whip",
                "bullpen_innings",
            ],
            diff_pairs=[
                ("ewm_run_diff", "form_run_diff"),
                ("win_rate_ewm", "form_win_rate"),
                ("ewm_hits", "hits"),
                ("ewm_walks", "walks"),
                ("ewm_strikeouts", "strikeouts"),
                ("ewm_home_runs", "home_runs"),
                ("ewm_total_bases", "total_bases"),
                ("ewm_starter_era", "starter_era"),
                ("ewm_starter_whip", "starter_whip"),
                ("lineup_availability", "lineup_availability"),
                ("bullpen_availability", "bullpen_availability"),
            ],
            direct_event_drop_columns=[
                "home_home_score",
                "home_away_score",
                "away_home_score",
                "away_away_score",
                "home_run_diff",
                "away_run_diff",
                "home_home_win",
                "away_home_win",
                "home_status_final",
                "away_status_final",
                "home_runs_for",
                "away_runs_for",
                "home_runs_against",
                "away_runs_against",
            ],
        )

    def prepare_team_games(self, team_games: pd.DataFrame, players_df: pd.DataFrame, injuries_df: pd.DataFrame) -> pd.DataFrame:
        df = team_games.sort_values(["team", "start_time_utc"]).copy()
        df["run_diff"] = df["runs_for"].fillna(0.0) - df["runs_against"].fillna(0.0)
        for col in ["hits", "walks", "strikeouts", "home_runs", "total_bases", "stolen_bases"]:
            df[col] = _numeric_column(df, col).fillna(0.0)

        df["starter_innings_pitched"] = _numeric_column(df, "starter_innings_pitched")
        df["starter_era"] = _numeric_column(df, "starter_era")
        df["starter_whip"] = _numeric_column(df, "starter_whip")
        df["starter_pitcher_strikeouts"] = _numeric_column(df, "starter_pitcher_strikeouts").fillna(0.0)
        df["starter_pitcher_walks"] = _numeric_column(df, "starter_pitcher_walks").fillna(0.0)
        df["starter_runs_allowed"] = _numeric_column(df, "starter_runs_allowed").fillna(df["runs_against"].fillna(0.0))
        df["starter_innings_pitched"] = df["starter_innings_pitched"].fillna(5.0).clip(lower=0.0, upper=9.0)
        df["starter_era"] = df["starter_era"].fillna(4.2).clip(lower=0.5, upper=12.0)
        df["starter_whip"] = df["starter_whip"].fillna(1.3).clip(lower=0.6, upper=3.0)
        df["bullpen_innings"] = (9.0 - df["starter_innings_pitched"]).clip(lower=0.0, upper=9.0)
        df["starter_confirmed"] = _text_column(df, "starter_status").str.lower().eq("confirmed").astype(int)

        lineup = _lineup_meta(players_df)
        if not lineup.empty:
            df = df.merge(lineup, on=["game_id", "team"], how="left")
        injuries = _injury_meta(injuries_df)
        if not injuries.empty:
            df = df.merge(injuries, on=["game_id", "team"], how="left")

        df["lineup_card_size"] = _numeric_column(df, "lineup_card_size").fillna(0.0).clip(lower=0.0, upper=9.0)
        df["lineup_availability"] = _numeric_column(df, "lineup_availability").fillna(0.0).clip(lower=0.0, upper=1.0)
        df["lineup_confirmed"] = _numeric_column(df, "lineup_confirmed").fillna(0).clip(lower=0, upper=1)
        df["position_player_out_count"] = _numeric_column(df, "position_player_out_count").fillna(0.0).clip(lower=0.0)
        df["pitcher_out_count"] = _numeric_column(df, "pitcher_out_count").fillna(0.0).clip(lower=0.0)
        df["bullpen_availability"] = _numeric_column(df, "bullpen_availability", 1.0).fillna(1.0).clip(lower=0.0, upper=1.0)
        return df

    def finalize_team_games(self, team_games: pd.DataFrame) -> pd.DataFrame:
        df = team_games.copy()
        season_dates = pd.to_datetime(df["game_date_utc"], errors="coerce")
        season_start = season_dates.min()
        df["days_into_season"] = (season_dates - season_start).dt.days.fillna(0)
        df["days_into_season_sqrt"] = np.sqrt(df["days_into_season"].clip(lower=0))
        df["season_game_number"] = pd.to_numeric(df["games_played_prior"], errors="coerce").fillna(0.0) + 1.0
        df["lineup_card_known"] = (df["lineup_card_size"] > 0).astype(int)
        df["injury_report_present"] = ((df["position_player_out_count"] > 0) | (df["pitcher_out_count"] > 0)).astype(int)
        df["starter_feature_available"] = (df["starter_pitcher_id"].notna()).astype(int)
        df["bullpen_rest_proxy"] = 1.0 / (1.0 + df["r5_bullpen_innings"].fillna(df["bullpen_innings"]))
        return df

    def enrich_game_level(
        self,
        merged: pd.DataFrame,
        games_df: pd.DataFrame,
        team_games: pd.DataFrame,
        context_df: pd.DataFrame,
    ) -> pd.DataFrame:
        del team_games
        out = merged.copy()
        out["home_hits"] = _numeric_column(out, "home_hits").fillna(0.0)
        out["away_hits"] = _numeric_column(out, "away_hits").fillna(0.0)
        out["home_total_bases"] = _numeric_column(out, "home_total_bases").fillna(0.0)
        out["away_total_bases"] = _numeric_column(out, "away_total_bases").fillna(0.0)
        out["target_total_runs"] = _numeric_column(out, "home_score").fillna(0.0) + _numeric_column(out, "away_score").fillna(0.0)
        out["target_run_margin"] = _numeric_column(out, "home_score").fillna(0.0) - _numeric_column(out, "away_score").fillna(0.0)
        out["target_hit_margin"] = out["home_hits"] - out["away_hits"]
        total_bases = (out["home_total_bases"] + out["away_total_bases"]).replace(0, np.nan)
        out["slugging_share_proxy"] = (out["home_total_bases"] / total_bases).fillna(0.5)
        out["lineup_confirmation_share"] = (
            _numeric_column(out, "home_lineup_confirmed").fillna(0.0)
            + _numeric_column(out, "away_lineup_confirmed").fillna(0.0)
        ) / 2.0

        travel = build_travel_features(games_df, league="MLB")
        if not travel.empty:
            out = out.merge(travel, on="game_id", how="left")
        if context_df is not None and not context_df.empty:
            out = out.merge(context_df, on="game_id", how="left")

        elo = compute_elo_features(games_df)
        dyn = compute_dynamic_rating_features(games_df)
        return out.merge(elo, on="game_id", how="left").merge(dyn, on="game_id", how="left")

    def add_model_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "diff_form_run_diff" in out.columns:
            out["diff_form_run_diff_hinge_000"] = _positive_part(out["diff_form_run_diff"], MLB_GLM_HINGE_KNOTS["diff_form_run_diff"])
        if "diff_starter_era" in out.columns:
            out["diff_starter_era_hinge_000"] = _positive_part(out["diff_starter_era"], MLB_GLM_HINGE_KNOTS["diff_starter_era"])
            out["starter_quality_edge"] = -pd.to_numeric(out["diff_starter_era"], errors="coerce")
        if "diff_starter_whip" in out.columns:
            out["starter_whip_edge"] = -pd.to_numeric(out["diff_starter_whip"], errors="coerce")
        if "diff_lineup_availability" in out.columns:
            lineup_edge = pd.to_numeric(out["diff_lineup_availability"], errors="coerce")
            out["lineup_quality_edge"] = lineup_edge
            confirmation = _numeric_column(out, "lineup_confirmation_share", 0.0).fillna(0.0)
            out["lineup_confirmed_quality_edge"] = lineup_edge * confirmation
        if "elo_home_prob" in out.columns:
            out["elo_home_prob_hinge_052"] = _positive_part(out["elo_home_prob"], MLB_GLM_HINGE_KNOTS["elo_home_prob"])
        return out

    def feature_hash_payload(self, feature_columns: list[str]) -> dict:
        return {"league": self.league, "n_features": len(feature_columns), "cols": feature_columns}
