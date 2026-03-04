from __future__ import annotations

import numpy as np
import pandas as pd



def compute_weights(metrics_df: pd.DataFrame) -> dict[str, float]:
    if metrics_df.empty:
        return {}
    req_cols = ["model_name", "log_loss", "brier", "ece", "calibration_beta"]
    for c in req_cols:
        if c not in metrics_df.columns:
            metrics_df[c] = 0.0

    df = metrics_df.copy()
    df["penalty"] = (
        df["log_loss"].fillna(df["log_loss"].mean())
        + 0.7 * df["brier"].fillna(df["brier"].mean())
        + 0.6 * df["ece"].fillna(df["ece"].mean())
        + 0.5 * (df["calibration_beta"].fillna(1.0) - 1.0).abs()
    )
    raw = np.exp(-df["penalty"].to_numpy(dtype=float))
    raw = np.where(np.isfinite(raw), raw, 0)
    if raw.sum() == 0:
        raw = np.ones_like(raw)
    weights = raw / raw.sum()
    return {m: float(w) for m, w in zip(df["model_name"], weights)}



def weighted_ensemble(pred_df: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    cols = [c for c in pred_df.columns if c in weights]
    if not cols:
        return np.full(len(pred_df), 0.5)
    w = np.array([weights[c] for c in cols], dtype=float)
    w = w / w.sum()
    arr = pred_df[cols].to_numpy(dtype=float)
    p = np.dot(arr, w)
    return np.clip(p, 1e-6, 1 - 1e-6)



def spread_stats(pred_df: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    arr = pred_df[model_cols].to_numpy(dtype=float)
    out = pd.DataFrame(
        {
            "spread_min": np.nanmin(arr, axis=1),
            "spread_median": np.nanmedian(arr, axis=1),
            "spread_max": np.nanmax(arr, axis=1),
            "spread_mean": np.nanmean(arr, axis=1),
            "spread_sd": np.nanstd(arr, axis=1),
            "spread_iqr": np.nanpercentile(arr, 75, axis=1) - np.nanpercentile(arr, 25, axis=1),
        }
    )
    return out
