from __future__ import annotations

import pandas as pd


INTERMEDIATE_TARGETS = ["xg_share_proxy", "penalty_diff_proxy", "pace_proxy"]



def add_intermediate_targets(team_games: pd.DataFrame) -> pd.DataFrame:
    df = team_games.copy()
    total_shots = (df["shots_for"].fillna(0) + df["shots_against"].fillna(0)).replace(0, 1)
    df["xg_share_proxy"] = df["shots_for"].fillna(0) / total_shots
    df["penalty_diff_proxy"] = df["penalties_drawn"].fillna(0) - df["penalties_taken"].fillna(0)
    df["pace_proxy"] = total_shots
    return df
