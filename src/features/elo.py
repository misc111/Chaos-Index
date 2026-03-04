from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.common.utils import sigmoid


@dataclass
class EloConfig:
    k: float = 20.0
    home_adv: float = 35.0
    base_rating: float = 1500.0



def compute_elo_features(games_df: pd.DataFrame, config: EloConfig | None = None) -> pd.DataFrame:
    if config is None:
        config = EloConfig()
    if games_df.empty:
        return pd.DataFrame(columns=["game_id", "elo_home_pre", "elo_away_pre", "elo_home_prob"])

    ratings: dict[str, float] = {}
    rows = []
    games = games_df.sort_values("start_time_utc").copy()
    for _, r in games.iterrows():
        home = r["home_team"]
        away = r["away_team"]
        rh = ratings.get(home, config.base_rating)
        ra = ratings.get(away, config.base_rating)
        logit = (rh + config.home_adv - ra) / 400.0
        p_home = sigmoid(logit)
        rows.append(
            {
                "game_id": r["game_id"],
                "elo_home_pre": rh,
                "elo_away_pre": ra,
                "elo_home_prob": p_home,
            }
        )

        if pd.notna(r.get("home_win")):
            y = float(r["home_win"])
            margin = (r.get("home_score", 0) or 0) - (r.get("away_score", 0) or 0)
            margin_mult = 1.0 + min(abs(margin), 4) * 0.1
            delta = config.k * margin_mult * (y - p_home)
            ratings[home] = rh + delta
            ratings[away] = ra - delta

    return pd.DataFrame(rows)
