from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from src.models.base import BaseProbModel


class TwoStageModel(BaseProbModel):
    model_name = "two_stage"

    def __init__(self, random_state: int = 42):
        super().__init__()
        self.stage1_targets = ["target_xg_share", "target_penalty_diff", "target_pace"]
        self.stage1_models = {
            t: RandomForestRegressor(
                n_estimators=250,
                max_depth=8,
                random_state=random_state,
                n_jobs=-1,
            )
            for t in self.stage1_targets
        }
        self.stage2 = LogisticRegression(max_iter=1500, C=1.0)

    def fit(self, df: pd.DataFrame, feature_columns: list[str], target_col: str = "home_win") -> None:
        train = df[df[target_col].notna()].copy()
        self.feature_columns = feature_columns
        x = train[self.feature_columns].to_numpy(dtype=float)

        inter_preds = []
        for t in self.stage1_targets:
            y_t = train[t].fillna(train[t].median()).to_numpy(dtype=float)
            self.stage1_models[t].fit(x, y_t)
            inter_preds.append(self.stage1_models[t].predict(x))

        z = np.column_stack([x] + inter_preds)
        y = train[target_col].astype(int).to_numpy()
        self.stage2.fit(z, y)

    def predict_intermediates(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.feature_columns].to_numpy(dtype=float)
        out = {}
        for t in self.stage1_targets:
            out[t.replace("target_", "pred_")] = self.stage1_models[t].predict(x)
        return pd.DataFrame(out, index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.feature_columns].to_numpy(dtype=float)
        inter = self.predict_intermediates(df)
        z = np.column_stack([x, inter.to_numpy(dtype=float)])
        p = self.stage2.predict_proba(z)[:, 1]
        return np.clip(p, 1e-6, 1 - 1e-6)
