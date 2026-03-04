from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.base import BaseProbModel


class GLMLogitModel(BaseProbModel):
    model_name = "glm_logit"

    def __init__(self, c: float = 1.0, random_state: int = 42):
        super().__init__()
        self.model = LogisticRegression(
            penalty="l2",
            C=c,
            max_iter=2000,
            solver="lbfgs",
            random_state=random_state,
        )

    def fit(self, df: pd.DataFrame, feature_columns: list[str], target_col: str = "home_win") -> None:
        train = df[df[target_col].notna()].copy()
        self.feature_columns = feature_columns
        x = train[self.feature_columns].astype(float)
        y = train[target_col].astype(int).to_numpy()
        self.model.fit(x, y)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.feature_columns].astype(float)
        return self.model.predict_proba(x)[:, 1]

    def coef_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": self.feature_columns,
                "coef": self.model.coef_[0],
            }
        ).sort_values("coef", key=lambda s: np.abs(s), ascending=False)
