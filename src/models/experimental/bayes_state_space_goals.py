from __future__ import annotations

import numpy as np
import pandas as pd


class BayesGoalsModel:
    """Lightweight Bayesian goal-rate model with Gamma-Poisson updating."""

    model_name = "bayes_goals"

    def __init__(self, prior_alpha: float = 5.0, prior_beta: float = 1.6, random_seed: int = 42):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.rng = np.random.default_rng(random_seed)
        self.team_alpha: dict[str, float] = {}
        self.team_beta: dict[str, float] = {}

    def fit(self, games_df: pd.DataFrame) -> None:
        train = games_df[games_df["home_win"].notna()].copy()
        for _, r in train.iterrows():
            for team, goals in [(r["home_team"], r["home_score"]), (r["away_team"], r["away_score"])]:
                self.team_alpha[team] = self.team_alpha.get(team, self.prior_alpha) + float(goals)
                self.team_beta[team] = self.team_beta.get(team, self.prior_beta) + 1.0

    def _rate(self, team: str) -> float:
        a = self.team_alpha.get(team, self.prior_alpha)
        b = self.team_beta.get(team, self.prior_beta)
        return max(a / max(b, 1e-6), 0.8)

    def predict_proba(self, games_df: pd.DataFrame, draws: int = 2500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        means, lows, highs = [], [], []
        for _, r in games_df.iterrows():
            lam_h = self._rate(r["home_team"]) * 1.05
            lam_a = self._rate(r["away_team"])
            hg = self.rng.poisson(lam_h, size=draws)
            ag = self.rng.poisson(lam_a, size=draws)
            p_samples = (hg > ag).astype(float) + 0.5 * (hg == ag).astype(float)
            means.append(float(p_samples.mean()))
            lows.append(float(np.quantile(p_samples, 0.05)))
            highs.append(float(np.quantile(p_samples, 0.95)))
        return np.array(means), np.array(lows), np.array(highs)
