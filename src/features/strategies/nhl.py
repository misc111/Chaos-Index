"""NHL-specific feature transforms for the shared pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.dynamic_ratings import compute_dynamic_rating_features
from src.features.elo import compute_elo_features
from src.features.goalie_features import add_goalie_features, combine_goalie_game_features
from src.features.intermediates import add_intermediate_targets
from src.features.rink_adjustments import compute_rink_effects
from src.features.special_teams import add_special_teams_features, combine_special_teams_game_features
from src.features.strategies.base import BaseFeatureStrategy
from src.features.travel import build_travel_features


NHL_GLM_HINGE_KNOTS = {
    "diff_form_goal_diff": (-1.0, 1.0),
    "dyn_home_prob": 0.55,
    "dyn_home_mean": 0.0,
    "elo_home_prob": 0.54,
}


def _positive_part(series: pd.Series, knot: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return (values - float(knot)).clip(lower=0.0)


def _lineup_meta(players_df: pd.DataFrame, injuries_df: pd.DataFrame) -> pd.DataFrame:
    lineup_strength = pd.DataFrame(columns=["team", "roster_strength_index", "lineup_uncertainty"])
    if not players_df.empty:
        players = players_df.copy()
        players["games_played"] = pd.to_numeric(players["games_played"], errors="coerce").fillna(0)
        players["points"] = pd.to_numeric(players["points"], errors="coerce").fillna(0)
        players = players[players["games_played"] > 0]
        if not players.empty:
            players["pts_per_game"] = players["points"] / players["games_played"]
            lineup_strength = players.sort_values("pts_per_game", ascending=False).groupby("team", as_index=False).head(6)
            lineup_strength = lineup_strength.groupby("team", as_index=False).agg(roster_strength_index=("pts_per_game", "mean"))
            lineup_strength["lineup_uncertainty"] = 0

    injuries = injuries_df[["team", "lineup_uncertainty", "man_games_lost_proxy"]].copy() if not injuries_df.empty else pd.DataFrame(columns=["team", "lineup_uncertainty", "man_games_lost_proxy"])
    team_meta = lineup_strength.merge(injuries, on="team", how="outer")
    if "lineup_uncertainty_x" in team_meta.columns:
        team_meta["lineup_uncertainty"] = team_meta["lineup_uncertainty_x"].fillna(team_meta["lineup_uncertainty_y"])
        team_meta = team_meta.drop(columns=["lineup_uncertainty_x", "lineup_uncertainty_y"])
    team_meta["roster_strength_index"] = team_meta.get("roster_strength_index", pd.Series(dtype=float)).fillna(0.1)
    team_meta["lineup_uncertainty"] = team_meta.get("lineup_uncertainty", pd.Series(dtype=float)).fillna(1)
    team_meta["man_games_lost_proxy"] = team_meta.get("man_games_lost_proxy", pd.Series(dtype=float)).fillna(0)
    return team_meta


class NhlFeatureStrategy(BaseFeatureStrategy):
    def __init__(self) -> None:
        super().__init__(
            league="NHL",
            summary_aggregations={
                "shots_for": ("shots_for", "max"),
                "shots_against": ("shots_against", "max"),
                "penalties_taken": ("penalties_taken", "max"),
                "penalties_drawn": ("penalties_drawn", "max"),
                "pp_goals": ("pp_goals", "max"),
            },
            starter_aggregations={
                "starter_goalie_id": ("goalie_id", "first"),
                "starter_name": ("goalie_name", "first"),
                "starter_save_pct": ("save_pct", "mean"),
                "starter_status": ("starter_status", "first"),
            },
            team_value_for_column="goals_for",
            team_value_against_column="goals_against",
            team_result_column="won",
            rolling_value_columns=["goals_for", "goals_against", "goal_diff", "shots_for", "shots_against", "team_save_pct_proxy"],
            diff_pairs=[
                ("ewm_goal_diff", "form_goal_diff"),
                ("win_rate_ewm", "form_win_rate"),
                ("xg_share_ewm", "xg_share"),
                ("ewm_shots_for", "shots_for"),
                ("ewm_shots_against", "shots_against"),
                ("penalty_diff_ewm", "penalty_diff"),
                ("roster_strength_index", "roster_strength"),
                ("lineup_uncertainty", "lineup_uncertainty"),
            ],
            direct_event_drop_columns=[
                "home_home_score",
                "home_away_score",
                "away_home_score",
                "away_away_score",
                "home_goal_diff",
                "away_goal_diff",
                "home_home_win",
                "away_home_win",
                "home_status_final",
                "away_status_final",
                "home_goals_for",
                "away_goals_for",
                "home_goals_against",
                "away_goals_against",
            ],
        )

    def prepare_team_games(self, team_games: pd.DataFrame, players_df: pd.DataFrame, injuries_df: pd.DataFrame) -> pd.DataFrame:
        df = team_games.sort_values(["team", "start_time_utc"]).copy()
        team_meta = _lineup_meta(players_df, injuries_df)
        df = df.merge(team_meta, on="team", how="left")
        df["goal_diff"] = df["goals_for"].fillna(0) - df["goals_against"].fillna(0)
        df["shots_for"] = df["shots_for"].fillna(df["goals_for"] * 6 + 25)
        df["shots_against"] = df["shots_against"].fillna(df["goals_against"] * 6 + 25)
        df["team_save_pct_proxy"] = 1 - (df["goals_against"].fillna(0) / df["shots_against"].replace(0, np.nan))
        df["team_save_pct_proxy"] = df["team_save_pct_proxy"].fillna(0.905)
        df["roster_strength_index"] = df["roster_strength_index"].fillna(0.1)
        df["lineup_uncertainty"] = df["lineup_uncertainty"].fillna(1)
        df["man_games_lost_proxy"] = df["man_games_lost_proxy"].fillna(0)
        return df

    def finalize_team_games(self, team_games: pd.DataFrame) -> pd.DataFrame:
        df = team_games.copy()
        df["xg_available"] = 0
        df["xg_for_ewm"] = df["ewm_shots_for"] / 10.0
        df["xg_against_ewm"] = df["ewm_shots_against"] / 10.0
        df["xg_share_ewm"] = df["xg_for_ewm"] / (df["xg_for_ewm"] + df["xg_against_ewm"]).replace(0, np.nan)
        df["xg_share_ewm"] = df["xg_share_ewm"].fillna(0.5)
        df = add_special_teams_features(df)
        df = add_goalie_features(df)
        df = add_intermediate_targets(df)

        season_start = pd.to_datetime(df["game_date_utc"]).min()
        season_dates = pd.to_datetime(df["game_date_utc"])
        df["days_into_season"] = (season_dates - season_start).dt.days.fillna(0)
        df["days_into_season_spline"] = np.sqrt(df["days_into_season"].clip(lower=0))
        df["season_phase"] = pd.cut(df["days_into_season"], bins=[-1, 45, 120, 1000], labels=["early", "mid", "late"]).astype(str)
        df["season_phase_early"] = (df["season_phase"] == "early").astype(int)
        df["season_phase_mid"] = (df["season_phase"] == "mid").astype(int)
        df["season_phase_late"] = (df["season_phase"] == "late").astype(int)
        df["post_trade_deadline"] = (season_dates.dt.month >= 3).astype(int)
        df["coaching_change_indicator"] = 0
        return df

    def enrich_game_level(self, merged: pd.DataFrame, games_df: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
        out = merged.copy()
        out["home_shots_for"] = out["home_shots_for"].fillna(0)
        out["away_shots_for"] = out["away_shots_for"].fillna(0)
        out["home_penalties_drawn"] = out["home_penalties_drawn"].fillna(0)
        out["away_penalties_drawn"] = out["away_penalties_drawn"].fillna(0)
        out["home_penalties_taken"] = out["home_penalties_taken"].fillna(0)
        out["away_penalties_taken"] = out["away_penalties_taken"].fillna(0)

        total_shots = (out["home_shots_for"] + out["away_shots_for"]).replace(0, np.nan)
        out["target_xg_share"] = (out["home_shots_for"] / total_shots).fillna(0.5)
        out["target_penalty_diff"] = (out["home_penalties_drawn"] - out["home_penalties_taken"]) - (
            out["away_penalties_drawn"] - out["away_penalties_taken"]
        )
        out["target_pace"] = (out["home_shots_for"] + out["away_shots_for"]).fillna(0)

        out = combine_special_teams_game_features(out)
        out = combine_goalie_game_features(out)

        travel = build_travel_features(games_df)
        if not travel.empty:
            out = out.merge(travel, on="game_id", how="left")

        rink = compute_rink_effects(out[out["status_final"] == 1])
        out = out.merge(rink, on="venue", how="left")
        out["rink_goal_effect"] = out["rink_goal_effect"].fillna(0)
        out["rink_shot_effect"] = out["rink_shot_effect"].fillna(0)
        out["fallback_xg_proxy_used"] = 1
        out["fallback_goalie_unknown"] = (
            out["home_goalie_uncertainty_feature"].fillna(1) + out["away_goalie_uncertainty_feature"].fillna(1) > 0
        ).astype(int)
        out["fallback_lineup_proxy_used"] = 1

        elo = compute_elo_features(games_df)
        dyn = compute_dynamic_rating_features(games_df)
        return out.merge(elo, on="game_id", how="left").merge(dyn, on="game_id", how="left")

    def add_model_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "diff_xg_share" in out.columns:
            xg_share = pd.to_numeric(out["diff_xg_share"], errors="coerce")
            out["diff_xg_share_cubic"] = xg_share.pow(3)
        if "diff_form_goal_diff" in out.columns:
            left_knot, right_knot = NHL_GLM_HINGE_KNOTS["diff_form_goal_diff"]
            out["diff_form_goal_diff_hinge_m1"] = _positive_part(out["diff_form_goal_diff"], left_knot)
            out["diff_form_goal_diff_hinge_p1"] = _positive_part(out["diff_form_goal_diff"], right_knot)
        if "dyn_home_prob" in out.columns:
            out["dyn_home_prob_hinge_055"] = _positive_part(out["dyn_home_prob"], NHL_GLM_HINGE_KNOTS["dyn_home_prob"])
        if "dyn_home_mean" in out.columns:
            out["dyn_home_mean_hinge_000"] = _positive_part(out["dyn_home_mean"], NHL_GLM_HINGE_KNOTS["dyn_home_mean"])
        if "elo_home_prob" in out.columns:
            out["elo_home_prob_hinge_054"] = _positive_part(out["elo_home_prob"], NHL_GLM_HINGE_KNOTS["elo_home_prob"])
        return out

    def feature_hash_payload(self, feature_columns: list[str]) -> dict:
        return {"league": self.league, "n_features": len(feature_columns), "cols": feature_columns}
