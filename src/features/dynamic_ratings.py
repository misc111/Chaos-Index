from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.utils import sigmoid



def compute_dynamic_rating_features(
    games_df: pd.DataFrame,
    process_var: float = 0.07,
    obs_scale: float = 1.0,
) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(columns=["game_id", "dyn_home_mean", "dyn_away_mean", "dyn_home_prob", "dyn_var_diff"])

    teams = sorted(set(games_df["home_team"].dropna()) | set(games_df["away_team"].dropna()))
    mean = {t: 0.0 for t in teams}
    var = {t: 1.0 for t in teams}

    rows = []
    for _, r in games_df.sort_values("start_time_utc").iterrows():
        h, a = r["home_team"], r["away_team"]
        mean[h] += 0.0
        mean[a] += 0.0
        var[h] += process_var
        var[a] += process_var

        z_mean = mean[h] - mean[a] + 0.12
        z_var = var[h] + var[a]
        p = sigmoid(z_mean / max(obs_scale, 1e-6))

        rows.append(
            {
                "game_id": r["game_id"],
                "dyn_home_mean": mean[h],
                "dyn_away_mean": mean[a],
                "dyn_home_prob": p,
                "dyn_var_diff": z_var,
            }
        )

        if pd.notna(r.get("home_win")):
            y = float(r["home_win"])
            grad = y - p
            fisher = max(p * (1 - p), 1e-4)
            k_h = var[h] / (1 + fisher * z_var)
            k_a = var[a] / (1 + fisher * z_var)
            mean[h] += k_h * grad
            mean[a] -= k_a * grad
            var[h] = max(var[h] * (1 - k_h * fisher), 1e-4)
            var[a] = max(var[a] * (1 - k_a * fisher), 1e-4)

    return pd.DataFrame(rows)
