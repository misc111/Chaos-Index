from __future__ import annotations

import pandas as pd



def compute_rink_effects(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(columns=["venue", "rink_goal_effect", "rink_shot_effect"])

    tmp = games_df.copy()
    tmp["goal_diff"] = tmp["home_score"].fillna(0) - tmp["away_score"].fillna(0)
    tmp["shot_diff"] = tmp["home_shots_for"].fillna(0) - tmp["away_shots_for"].fillna(0)
    venue_mean = tmp.groupby("venue", dropna=False).agg(
        rink_goal_effect=("goal_diff", "mean"),
        rink_shot_effect=("shot_diff", "mean"),
        n=("game_id", "count"),
    )
    # Shrink toward zero for low-sample venues.
    venue_mean["rink_goal_effect"] = venue_mean["rink_goal_effect"] * (venue_mean["n"] / (venue_mean["n"] + 25))
    venue_mean["rink_shot_effect"] = venue_mean["rink_shot_effect"] * (venue_mean["n"] / (venue_mean["n"] + 25))
    return venue_mean.reset_index()[["venue", "rink_goal_effect", "rink_shot_effect"]]
