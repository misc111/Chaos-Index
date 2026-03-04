from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GoalsModelOutput:
    prob_home_win: float
    home_lambda: float
    away_lambda: float


class GoalsPoissonModel:
    model_name = "goals_poisson"

    def __init__(self, shrinkage_alpha: float = 18.0, random_seed: int = 42):
        self.shrinkage_alpha = shrinkage_alpha
        self.random = np.random.default_rng(random_seed)
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.league_avg = 3.0
        self.home_adv = 1.05

    def fit(self, games_df: pd.DataFrame) -> None:
        train = games_df[games_df["home_win"].notna()].copy()
        if train.empty:
            return

        self.league_avg = float(
            np.nanmean(np.concatenate([train["home_score"].fillna(0).values, train["away_score"].fillna(0).values]))
        )
        mean_home = float(train["home_score"].mean())
        mean_away = float(train["away_score"].mean())
        self.home_adv = max(0.8, min(1.2, mean_home / max(mean_away, 1e-6)))

        team_rows = []
        for _, r in train.iterrows():
            team_rows.append({"team": r["home_team"], "goals_for": r["home_score"], "goals_against": r["away_score"]})
            team_rows.append({"team": r["away_team"], "goals_for": r["away_score"], "goals_against": r["home_score"]})
        tg = pd.DataFrame(team_rows)

        grouped = tg.groupby("team").agg(gf=("goals_for", "sum"), ga=("goals_against", "sum"), gp=("goals_for", "count"))
        for team, row in grouped.iterrows():
            gp = row["gp"]
            gf = row["gf"]
            ga = row["ga"]
            self.attack[team] = ((gf + self.shrinkage_alpha * self.league_avg) / (gp + self.shrinkage_alpha)) / max(self.league_avg, 1e-6)
            self.defense[team] = ((ga + self.shrinkage_alpha * self.league_avg) / (gp + self.shrinkage_alpha)) / max(self.league_avg, 1e-6)

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        ah = self.attack.get(home_team, 1.0)
        aa = self.attack.get(away_team, 1.0)
        dh = self.defense.get(home_team, 1.0)
        da = self.defense.get(away_team, 1.0)
        lam_home = max(0.8, self.league_avg * ah * da * self.home_adv)
        lam_away = max(0.8, self.league_avg * aa * dh)
        return lam_home, lam_away

    def predict_row(self, home_team: str, away_team: str, n_sim: int = 3000) -> GoalsModelOutput:
        lam_home, lam_away = self.expected_goals(home_team, away_team)
        home_goals = self.random.poisson(lam_home, size=n_sim)
        away_goals = self.random.poisson(lam_away, size=n_sim)
        # OT/SO approximation: ties split 50/50
        p = float(np.mean(home_goals > away_goals) + 0.5 * np.mean(home_goals == away_goals))
        return GoalsModelOutput(prob_home_win=float(np.clip(p, 1e-6, 1 - 1e-6)), home_lambda=lam_home, away_lambda=lam_away)

    def predict_proba(self, games_df: pd.DataFrame, n_sim: int = 3000) -> np.ndarray:
        probs = []
        for _, r in games_df.iterrows():
            out = self.predict_row(r["home_team"], r["away_team"], n_sim=n_sim)
            probs.append(out.prob_home_win)
        return np.array(probs)
