from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import metric_bundle
from src.models.glm_logit import GLMLogitModel



def quick_tune_glm(df: pd.DataFrame, feature_cols: list[str], c_grid: list[float] | None = None) -> dict:
    if c_grid is None:
        c_grid = [0.3, 0.7, 1.0, 1.4, 2.0]

    train = df[df["home_win"].notna()].copy().sort_values("start_time_utc")
    if len(train) < 80:
        return {"best_c": 1.0, "results": []}

    cut = int(len(train) * 0.8)
    tr = train.iloc[:cut]
    va = train.iloc[cut:]

    rows = []
    for c in c_grid:
        m = GLMLogitModel(c=c)
        m.fit(tr, feature_cols)
        p = m.predict_proba(va)
        met = metric_bundle(va["home_win"].to_numpy(), p)
        rows.append({"c": c} | met)

    best = sorted(rows, key=lambda r: r["log_loss"])[0]
    return {"best_c": best["c"], "results": rows}
