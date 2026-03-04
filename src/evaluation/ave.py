from __future__ import annotations

import pandas as pd



def calibration_ave_deciles(df: pd.DataFrame, p_col: str, y_col: str, bins: int = 10) -> pd.DataFrame:
    tmp = df[[p_col, y_col]].dropna().copy()
    if tmp.empty:
        return pd.DataFrame(columns=["decile", "avg_pred", "avg_obs", "ave_gap"])
    tmp["decile"] = pd.qcut(tmp[p_col], q=min(bins, tmp[p_col].nunique()), labels=False, duplicates="drop")
    out = tmp.groupby("decile", as_index=False).agg(avg_pred=(p_col, "mean"), avg_obs=(y_col, "mean"), n=(y_col, "count"))
    out["ave_gap"] = out["avg_pred"] - out["avg_obs"]
    return out



def calibration_ave_timeseries(df: pd.DataFrame, p_col: str, y_col: str, date_col: str) -> pd.DataFrame:
    tmp = df[[p_col, y_col, date_col]].dropna().copy()
    if tmp.empty:
        return pd.DataFrame(columns=[date_col, "ave_gap"])
    g = tmp.groupby(date_col, as_index=False).agg(avg_pred=(p_col, "mean"), avg_obs=(y_col, "mean"))
    g["ave_gap"] = g["avg_pred"] - g["avg_obs"]
    return g
