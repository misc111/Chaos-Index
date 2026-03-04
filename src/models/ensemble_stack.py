from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class StackingEnsemble:
    model_name = "ensemble_stack"

    def __init__(self):
        self.model = LogisticRegression(max_iter=2000, C=0.7)
        self.base_columns: list[str] = []

    def fit(self, oof_df: pd.DataFrame, base_columns: list[str], target_col: str = "home_win") -> None:
        self.base_columns = base_columns
        x = oof_df[self.base_columns].to_numpy(dtype=float)
        y = oof_df[target_col].astype(int).to_numpy()
        self.model.fit(x, y)

    def predict_proba(self, base_pred_df: pd.DataFrame) -> np.ndarray:
        x = base_pred_df[self.base_columns].to_numpy(dtype=float)
        p = self.model.predict_proba(x)[:, 1]
        return np.clip(p, 1e-6, 1 - 1e-6)
