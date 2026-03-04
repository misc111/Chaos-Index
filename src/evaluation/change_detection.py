from __future__ import annotations

import pandas as pd



def page_hinkley(values: list[float], delta: float = 0.001, threshold: float = 0.15) -> list[int]:
    mean = 0.0
    cum = 0.0
    min_cum = 0.0
    idxs = []

    for i, x in enumerate(values):
        mean = mean + (x - mean) / (i + 1)
        cum += x - mean - delta
        min_cum = min(min_cum, cum)
        if cum - min_cum > threshold:
            idxs.append(i)
            cum = 0.0
            min_cum = 0.0
    return idxs



def cusum(values: list[float], k: float = 0.01, h: float = 0.2) -> list[int]:
    target = sum(values) / max(len(values), 1)
    s_pos = 0.0
    s_neg = 0.0
    out = []
    for i, x in enumerate(values):
        s_pos = max(0.0, s_pos + x - target - k)
        s_neg = min(0.0, s_neg + x - target + k)
        if s_pos > h or abs(s_neg) > h:
            out.append(i)
            s_pos = 0.0
            s_neg = 0.0
    return out



def detect_change_points(scores_df: pd.DataFrame, metric_col: str = "log_loss") -> pd.DataFrame:
    if scores_df.empty or metric_col not in scores_df.columns:
        return pd.DataFrame(columns=["model_name", "metric_name", "method", "index", "date", "value"])

    out = []
    for model_name, grp in scores_df.groupby("model_name"):
        g = grp.sort_values("game_date_utc")
        vals = g[metric_col].astype(float).tolist()
        ph = page_hinkley(vals)
        cs = cusum(vals)
        for idx in ph:
            out.append(
                {
                    "model_name": model_name,
                    "metric_name": metric_col,
                    "method": "page_hinkley",
                    "index": idx,
                    "date": str(g.iloc[idx]["game_date_utc"]),
                    "value": float(vals[idx]),
                    "threshold": 0.15,
                    "statistic": float(vals[idx]),
                    "detected": 1,
                }
            )
        for idx in cs:
            out.append(
                {
                    "model_name": model_name,
                    "metric_name": metric_col,
                    "method": "cusum",
                    "index": idx,
                    "date": str(g.iloc[idx]["game_date_utc"]),
                    "value": float(vals[idx]),
                    "threshold": 0.2,
                    "statistic": float(vals[idx]),
                    "detected": 1,
                }
            )
    return pd.DataFrame(out)
