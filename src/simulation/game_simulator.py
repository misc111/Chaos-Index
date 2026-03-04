from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    prob_home_win: float
    variance: float
    home_goal_mean: float
    away_goal_mean: float


class GameSimulator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _lambdas_from_row(self, row: pd.Series) -> tuple[float, float]:
        base_h = max(0.8, float(row.get("home_ewm_goals_for", 3.0)))
        base_a = max(0.8, float(row.get("away_ewm_goals_for", 2.9)))
        def_h = max(0.8, float(row.get("home_ewm_goals_against", 3.0)))
        def_a = max(0.8, float(row.get("away_ewm_goals_against", 3.0)))

        lam_h = 0.5 * base_h + 0.5 * def_a + 0.08
        lam_a = 0.5 * base_a + 0.5 * def_h

        st_diff = float(row.get("special_pp_diff", 0.0))
        goalie_unc = float(row.get("goalie_uncertainty_diff", 0.0))
        lam_h *= np.exp(0.08 * st_diff - 0.04 * goalie_unc)
        lam_a *= np.exp(-0.08 * st_diff + 0.04 * goalie_unc)

        return max(lam_h, 0.6), max(lam_a, 0.6)

    def simulate_game(self, row: pd.Series, n_sims: int = 5000) -> SimulationResult:
        lam_h, lam_a = self._lambdas_from_row(row)
        hg = self.rng.poisson(lam_h, size=n_sims)
        ag = self.rng.poisson(lam_a, size=n_sims)
        p = (hg > ag).astype(float) + 0.5 * (hg == ag).astype(float)
        return SimulationResult(
            prob_home_win=float(np.mean(p)),
            variance=float(np.var(p)),
            home_goal_mean=float(np.mean(hg)),
            away_goal_mean=float(np.mean(ag)),
        )

    def simulate_dataframe(self, df: pd.DataFrame, n_sims: int = 5000) -> pd.DataFrame:
        rows = []
        for _, r in df.iterrows():
            out = self.simulate_game(r, n_sims=n_sims)
            rows.append(
                {
                    "game_id": r["game_id"],
                    "sim_prob_home_win": out.prob_home_win,
                    "sim_pred_var": out.variance,
                    "sim_home_goal_mean": out.home_goal_mean,
                    "sim_away_goal_mean": out.away_goal_mean,
                }
            )
        return pd.DataFrame(rows)
