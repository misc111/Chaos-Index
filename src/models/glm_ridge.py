from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseProbModel


class GLMRidgeModel(BaseProbModel):
    model_name = "glm_ridge"

    def __init__(self, c: float = 1.0, random_state: int = 42):
        super().__init__()
        self.c = float(c)
        self.model = LogisticRegression(
            penalty="l2",
            C=self.c,
            max_iter=4000,
            solver="lbfgs",
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self.feature_medians: dict[str, float] = {}

    def _prepare_x(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        x = df[self.feature_columns].astype(float).replace([np.inf, -np.inf], np.nan)
        if fit:
            med = x.median(numeric_only=True).fillna(0.0)
            self.feature_medians = {str(k): float(v) for k, v in med.items()}
        fill = pd.Series(self.feature_medians)
        x = x.fillna(fill).fillna(0.0)
        if fit:
            return self.scaler.fit_transform(x.to_numpy(dtype=float))
        return self.scaler.transform(x.to_numpy(dtype=float))

    def fit(self, df: pd.DataFrame, feature_columns: list[str], target_col: str = "home_win") -> None:
        train = df[df[target_col].notna()].copy()
        self.feature_columns = feature_columns
        y = train[target_col].astype(int).to_numpy()
        x = self._prepare_x(train, fit=True)
        self.model.fit(x, y)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = self._prepare_x(df, fit=False)
        p = self.model.predict_proba(x)[:, 1]
        return np.clip(p, 1e-6, 1 - 1e-6)

    def coef_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": self.feature_columns,
                "coef": self.model.coef_[0],
            }
        ).sort_values("coef", key=lambda s: np.abs(s), ascending=False)
