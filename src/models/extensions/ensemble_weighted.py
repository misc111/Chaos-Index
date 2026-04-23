"""Theory-compatible opt-in weighted ensemble helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_weights(metrics_df: pd.DataFrame) -> dict[str, float]:
    """Compute simple metric-weighted model weights for comparison overlays."""

    if metrics_df.empty:
        return {}
    req_cols = ["model_name", "log_loss", "brier", "ece", "calibration_beta"]
    for column in req_cols:
        if column not in metrics_df.columns:
            metrics_df[column] = 0.0

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
    return {model_name: float(weight) for model_name, weight in zip(df["model_name"], weights)}


def weighted_ensemble(pred_df: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    """Blend model probabilities using precomputed weights."""

    cols = [column for column in pred_df.columns if column in weights]
    if not cols:
        return np.full(len(pred_df), 0.5)
    raw_weights = np.array([weights[column] for column in cols], dtype=float)
    raw_weights = raw_weights / raw_weights.sum()
    arr = pred_df[cols].to_numpy(dtype=float)
    probability = np.dot(arr, raw_weights)
    return np.clip(probability, 1e-6, 1 - 1e-6)


def spread_stats(pred_df: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    """Summarize model-probability dispersion for validation/reporting overlays."""

    arr = pred_df[model_cols].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "spread_min": np.nanmin(arr, axis=1),
            "spread_median": np.nanmedian(arr, axis=1),
            "spread_max": np.nanmax(arr, axis=1),
            "spread_mean": np.nanmean(arr, axis=1),
            "spread_sd": np.nanstd(arr, axis=1),
            "spread_iqr": np.nanpercentile(arr, 75, axis=1) - np.nanpercentile(arr, 25, axis=1),
        }
    )
