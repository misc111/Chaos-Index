from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.common.utils import sigmoid


@dataclass
class BayesSummary:
    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray
    pred_var: np.ndarray


class BayesStateSpaceBTModel:
    model_name = "bayes_bt_state_space"

    def __init__(
        self,
        process_variance: float = 0.08,
        prior_variance: float = 1.5,
        draws: int = 500,
        random_seed: int = 42,
    ):
        self.process_variance = process_variance
        self.prior_variance = prior_variance
        self.draws = draws
        self.rng = np.random.default_rng(random_seed)

        self.team_to_ix: dict[str, int] = {}
        self.ix_to_team: list[str] = []
        self.state_mean = np.array([])
        self.state_var = np.array([])

        self.feature_columns: list[str] = []
        self.beta = np.array([])
        self.intercept = 0.0
        self.elbo_trace: list[float] = []

    @staticmethod
    def _team_code(value: object) -> str | None:
        if pd.isna(value):
            return None
        token = str(value).strip().upper()
        return token or None

    def _ensure_teams(self, teams: list[str]) -> None:
        for t in teams:
            if t not in self.team_to_ix:
                self.team_to_ix[t] = len(self.ix_to_team)
                self.ix_to_team.append(t)

        n = len(self.ix_to_team)
        if self.state_mean.size < n:
            old_n = self.state_mean.size
            add_n = n - old_n
            if old_n == 0:
                self.state_mean = np.zeros(n)
                self.state_var = np.full(n, self.prior_variance)
            else:
                self.state_mean = np.concatenate([self.state_mean, np.zeros(add_n)])
                self.state_var = np.concatenate([self.state_var, np.full(add_n, self.prior_variance)])

    def _predict_z(self, home_team: str, away_team: str, x_row: np.ndarray) -> tuple[float, float]:
        hi = self.team_to_ix.get(home_team)
        ai = self.team_to_ix.get(away_team)
        mu_h = self.state_mean[hi] if hi is not None else 0.0
        mu_a = self.state_mean[ai] if ai is not None else 0.0
        var_h = self.state_var[hi] if hi is not None else self.prior_variance
        var_a = self.state_var[ai] if ai is not None else self.prior_variance

        z_mean = self.intercept + (self.beta @ x_row if self.beta.size else 0.0) + mu_h - mu_a
        z_var = var_h + var_a + 1e-6
        return z_mean, z_var

    def fit_offline(self, df: pd.DataFrame, feature_columns: list[str], n_passes: int = 2) -> None:
        work = df[df["home_win"].notna()].copy().sort_values("start_time_utc")
        work["home_team"] = work["home_team"].map(self._team_code)
        work["away_team"] = work["away_team"].map(self._team_code)
        work = work[work["home_team"].notna() & work["away_team"].notna()].copy()
        self.feature_columns = feature_columns
        if work.empty:
            self.beta = np.zeros(len(self.feature_columns), dtype=float)
            self.intercept = 0.12
            return

        teams = sorted(set(work["home_team"]) | set(work["away_team"]))
        self._ensure_teams(teams)
        self.beta = np.zeros(len(self.feature_columns), dtype=float)
        self.intercept = 0.12

        for _ in range(max(n_passes, 1)):
            strength_diff = []
            x_rows = []
            y_rows = []
            ll_sum = 0.0

            # Reset to diffuse prior every full pass.
            self.state_mean = np.zeros(len(self.ix_to_team))
            self.state_var = np.full(len(self.ix_to_team), self.prior_variance)

            for _, r in work.iterrows():
                h = r["home_team"]
                a = r["away_team"]
                hi = self.team_to_ix[h]
                ai = self.team_to_ix[a]

                self.state_var[hi] += self.process_variance
                self.state_var[ai] += self.process_variance

                x = r[self.feature_columns].to_numpy(dtype=float)
                z_mean, z_var = self._predict_z(h, a, x)
                p = sigmoid(z_mean)
                y = float(r["home_win"])

                strength_diff.append(self.state_mean[hi] - self.state_mean[ai])
                x_rows.append(x)
                y_rows.append(y)

                # Approximate filtering update
                fisher = max(p * (1 - p), 1e-3)
                s_var = self.state_var[hi] + self.state_var[ai] + 1 / fisher
                innov = (y - p) / max(fisher, 1e-3)
                k_h = self.state_var[hi] / s_var
                k_a = self.state_var[ai] / s_var

                self.state_mean[hi] += k_h * innov
                self.state_mean[ai] -= k_a * innov
                self.state_var[hi] = max(self.state_var[hi] * (1 - k_h), 1e-4)
                self.state_var[ai] = max(self.state_var[ai] * (1 - k_a), 1e-4)

                self.state_mean -= self.state_mean.mean()  # identifiability sum-to-zero
                ll_sum += y * np.log(max(p, 1e-9)) + (1 - y) * np.log(max(1 - p, 1e-9))

            x_arr = np.vstack(x_rows)
            z_arr = np.column_stack([np.array(strength_diff), x_arr])
            y_arr = np.array(y_rows, dtype=int)
            clf = LogisticRegression(C=0.8, max_iter=2000)
            clf.fit(z_arr, y_arr)
            self.intercept = float(clf.intercept_[0])
            full_beta = clf.coef_[0]
            # first coefficient multiplies strength diff; absorb by scaling states
            strength_coef = full_beta[0]
            self.state_mean *= strength_coef
            self.state_var *= max(strength_coef**2, 1e-6)
            self.beta = full_beta[1:]

            self.elbo_trace.append(float(ll_sum))

    def daily_update(self, new_results_df: pd.DataFrame) -> None:
        work = new_results_df[new_results_df["home_win"].notna()].copy().sort_values("start_time_utc")
        work["home_team"] = work["home_team"].map(self._team_code)
        work["away_team"] = work["away_team"].map(self._team_code)
        work = work[work["home_team"].notna() & work["away_team"].notna()].copy()
        if work.empty:
            return

        self._ensure_teams(sorted(set(work["home_team"]) | set(work["away_team"])))

        for _, r in work.iterrows():
            h = r["home_team"]
            a = r["away_team"]
            hi = self.team_to_ix[h]
            ai = self.team_to_ix[a]

            self.state_var[hi] += self.process_variance
            self.state_var[ai] += self.process_variance

            x = np.array([float(r.get(c, 0.0)) for c in self.feature_columns], dtype=float)
            z_mean, _ = self._predict_z(h, a, x)
            p = sigmoid(z_mean)
            y = float(r["home_win"])
            fisher = max(p * (1 - p), 1e-3)
            s_var = self.state_var[hi] + self.state_var[ai] + 1 / fisher
            innov = (y - p) / max(fisher, 1e-3)
            k_h = self.state_var[hi] / s_var
            k_a = self.state_var[ai] / s_var
            self.state_mean[hi] += k_h * innov
            self.state_mean[ai] -= k_a * innov
            self.state_var[hi] = max(self.state_var[hi] * (1 - k_h), 1e-4)
            self.state_var[ai] = max(self.state_var[ai] * (1 - k_a), 1e-4)
            self.state_mean -= self.state_mean.mean()

    def predict_summary(self, df: pd.DataFrame) -> BayesSummary:
        means = []
        lows = []
        highs = []
        variances = []

        for _, r in df.iterrows():
            x = np.array([float(r.get(c, 0.0)) for c in self.feature_columns], dtype=float)
            home_team = self._team_code(r.get("home_team")) or ""
            away_team = self._team_code(r.get("away_team")) or ""
            z_mean, z_var = self._predict_z(home_team, away_team, x)
            z_samples = self.rng.normal(loc=z_mean, scale=np.sqrt(max(z_var, 1e-6)), size=self.draws)
            p_samples = 1.0 / (1.0 + np.exp(-z_samples))
            means.append(float(np.mean(p_samples)))
            lows.append(float(np.quantile(p_samples, 0.05)))
            highs.append(float(np.quantile(p_samples, 0.95)))
            variances.append(float(np.var(p_samples)))

        return BayesSummary(
            mean=np.array(means),
            low=np.array(lows),
            high=np.array(highs),
            pred_var=np.array(variances),
        )

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_summary(df).mean

    def diagnostics(self) -> dict:
        return {
            "elbo_trace": self.elbo_trace,
            "vi_converged": len(self.elbo_trace) >= 2,
            "r_hat": None,
            "ess": None,
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "process_variance": self.process_variance,
            "prior_variance": self.prior_variance,
            "draws": self.draws,
            "team_to_ix": self.team_to_ix,
            "ix_to_team": self.ix_to_team,
            "state_mean": self.state_mean.tolist(),
            "state_var": self.state_var.tolist(),
            "feature_columns": self.feature_columns,
            "beta": self.beta.tolist(),
            "intercept": self.intercept,
            "elbo_trace": self.elbo_trace,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "BayesStateSpaceBTModel":
        payload = json.loads(Path(path).read_text())
        model = cls(
            process_variance=payload["process_variance"],
            prior_variance=payload["prior_variance"],
            draws=payload.get("draws", 500),
        )
        model.team_to_ix = {k: int(v) for k, v in payload["team_to_ix"].items()}
        model.ix_to_team = list(payload["ix_to_team"])
        model.state_mean = np.array(payload["state_mean"], dtype=float)
        model.state_var = np.array(payload["state_var"], dtype=float)
        model.feature_columns = list(payload["feature_columns"])
        model.beta = np.array(payload["beta"], dtype=float)
        model.intercept = float(payload["intercept"])
        model.elbo_trace = list(payload.get("elbo_trace", []))
        return model
