from __future__ import annotations

import pandas as pd



def add_special_teams_features(team_games: pd.DataFrame) -> pd.DataFrame:
    if team_games.empty:
        return team_games

    df = team_games.sort_values(["team", "start_time_utc"]).copy()

    def _per_team(grp: pd.DataFrame) -> pd.DataFrame:
        g = grp.copy()
        g["penalties_taken"] = g["penalties_taken"].fillna(0)
        g["penalties_drawn"] = g["penalties_drawn"].fillna(0)
        g["pp_goals"] = g["pp_goals"].fillna(0)
        denom = g["penalties_drawn"].replace(0, 1)
        g["pp_eff"] = g["pp_goals"] / denom
        g["penalty_diff"] = g["penalties_drawn"] - g["penalties_taken"]
        g["pp_eff_ewm"] = g["pp_eff"].shift(1).ewm(alpha=0.25, adjust=False).mean().fillna(g["pp_eff"].mean())
        g["penalty_diff_ewm"] = g["penalty_diff"].shift(1).ewm(alpha=0.25, adjust=False).mean().fillna(0)
        g["penalties_taken_ewm"] = g["penalties_taken"].shift(1).ewm(alpha=0.25, adjust=False).mean().fillna(0)
        return g

    return df.groupby("team", group_keys=False).apply(_per_team)


def combine_special_teams_game_features(game_df: pd.DataFrame) -> pd.DataFrame:
    game_df["special_pp_diff"] = game_df["home_pp_eff_ewm"] - game_df["away_pp_eff_ewm"]
    game_df["special_penalty_diff"] = game_df["home_penalty_diff_ewm"] - game_df["away_penalty_diff_ewm"]
    game_df["special_pk_pressure_diff"] = game_df["away_penalties_taken_ewm"] - game_df["home_penalties_taken_ewm"]
    return game_df
