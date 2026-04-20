from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from src.models.base import BaseProbModel


class NNModel(BaseProbModel):
    model_name = "nn_mlp"

    def __init__(self, random_state: int = 42):
        super().__init__()
        self.model = MLPClassifier(
            hidden_layer_sizes=(48, 24),
            activation="relu",
            alpha=1e-3,
            learning_rate_init=2e-3,
            max_iter=600,
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
        p = self.model.predict_proba(x)[:, 1]
        return np.clip(p, 1e-6, 1 - 1e-6)
