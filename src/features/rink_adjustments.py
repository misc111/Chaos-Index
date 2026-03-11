from __future__ import annotations

import pandas as pd

from src.features.contextual_effects import compute_causal_group_effects



def compute_rink_effects(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(columns=["game_id", "rink_goal_effect", "rink_shot_effect"])

    tmp = games_df.copy()
    tmp["goal_diff"] = tmp["home_score"].fillna(0) - tmp["away_score"].fillna(0)
    tmp["shot_diff"] = tmp["home_shots_for"].fillna(0) - tmp["away_shots_for"].fillna(0)
    return compute_causal_group_effects(
        tmp,
        group_col="venue",
        metric_columns={
            "rink_goal_effect": "goal_diff",
            "rink_shot_effect": "shot_diff",
        },
        shrinkage=25.0,
    )
