from __future__ import annotations

import numpy as np
import pandas as pd



def add_goalie_features(team_games: pd.DataFrame) -> pd.DataFrame:
    if team_games.empty:
        return team_games

    df = team_games.sort_values(["team", "start_time_utc"]).copy()

    def _per_team(grp: pd.DataFrame) -> pd.DataFrame:
        g = grp.copy()
        g["starter_known"] = (g["starter_status"] == "confirmed").astype(int)
        g["starter_unknown"] = 1 - g["starter_known"]
        g["goalie_quality_raw"] = g["starter_save_pct"].fillna(g["team_save_pct_proxy"])
        g["goalie_quality_raw"] = g["goalie_quality_raw"].fillna(0.905)
        g["goalie_quality_ewm"] = g["goalie_quality_raw"].shift(1).ewm(alpha=0.25, adjust=False).mean()
        g["goalie_quality_ewm"] = g["goalie_quality_ewm"].fillna(0.905)
        g["goalie_starts_last7"] = g["starter_known"].shift(1).rolling(7, min_periods=1).sum().fillna(0)
        g["goalie_starts_last14"] = g["starter_known"].shift(1).rolling(14, min_periods=1).sum().fillna(0)
        g["goalie_b2b_starter"] = (g["starter_known"].shift(1).fillna(0) * g["b2b"].fillna(0)).astype(int)
        g["goalie_uncertainty_feature"] = g["starter_unknown"].shift(1).fillna(1)
        return g

    out = df.groupby("team", group_keys=False).apply(_per_team)
    return out


def combine_goalie_game_features(game_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "game_id",
        "home_goalie_quality_ewm",
        "away_goalie_quality_ewm",
        "home_goalie_starts_last7",
        "away_goalie_starts_last7",
        "home_goalie_starts_last14",
        "away_goalie_starts_last14",
        "home_goalie_b2b_starter",
        "away_goalie_b2b_starter",
        "home_goalie_uncertainty_feature",
        "away_goalie_uncertainty_feature",
    ]
    for c in cols:
        if c not in game_df.columns:
            game_df[c] = np.nan

    game_df["goalie_quality_diff"] = game_df["home_goalie_quality_ewm"] - game_df["away_goalie_quality_ewm"]
    game_df["goalie_workload_diff_7"] = game_df["home_goalie_starts_last7"] - game_df["away_goalie_starts_last7"]
    game_df["goalie_workload_diff_14"] = game_df["home_goalie_starts_last14"] - game_df["away_goalie_starts_last14"]
    game_df["goalie_uncertainty_diff"] = game_df["home_goalie_uncertainty_feature"] - game_df["away_goalie_uncertainty_feature"]
    return game_df
