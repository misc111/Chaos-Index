from __future__ import annotations

import pandas as pd


def build_results_from_games(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "season",
                "game_date_utc",
                "final_utc",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "home_win",
                "ingested_at_utc",
            ]
        )

    final_games = games_df[games_df["status_final"] == 1].copy()
    final_games["final_utc"] = final_games["start_time_utc"]
    final_games["ingested_at_utc"] = final_games["as_of_utc"]
    keep_cols = [
        "game_id",
        "season",
        "game_date_utc",
        "final_utc",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_win",
        "ingested_at_utc",
    ]
    return final_games[keep_cols].drop_duplicates(subset=["game_id"]).reset_index(drop=True)
