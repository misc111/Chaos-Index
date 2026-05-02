"""Metric helpers for current-season MLB predictiveness evidence."""

from __future__ import annotations

from collections.abc import Iterable
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.calibration import calibration_alpha_beta, ece_mce, reliability_table
from src.evaluation.metrics import metric_bundle
from src.evaluation.performance_timeseries import compute_performance_aggregates
from src.services.current_season_predictiveness_contract import METRICS_COLUMNS, clean_payload


def rolling_records(frame: pd.DataFrame, *, model_name: str, as_of_utc: str, windows_days: Iterable[int]) -> dict[str, Any]:
    """Build cumulative and rolling-window metric records for one model."""

    if frame.empty:
        return {}
    aggregates = compute_performance_aggregates(frame.copy(), as_of_utc=as_of_utc, windows_days=list(windows_days))
    if aggregates.empty:
        return {}
    records: dict[str, Any] = {}
    for row in aggregates.sort_values(["window_label"]).to_dict(orient="records"):
        label = str(row["window_label"])
        seg = frame if label == "cumulative" else _window_segment(frame, label)
        records[label] = clean_payload(metric_record(seg, model_name=model_name, window_label=label, as_of_utc=as_of_utc))
    return records


def metric_record(seg: pd.DataFrame, *, model_name: str, window_label: str, as_of_utc: str) -> dict[str, Any]:
    """Compute probability-skill metrics for one model/window slice."""

    y = seg["outcome_home_win"].astype(int).to_numpy()
    p = seg["prob_home_win"].astype(float).to_numpy()
    metrics = metric_bundle(y, p)
    return {
        "as_of_utc": as_of_utc,
        "window_label": window_label,
        "start_date": str(pd.to_datetime(seg["game_date_utc"]).min().date()),
        "end_date": str(pd.to_datetime(seg["game_date_utc"]).max().date()),
        "n_games": int(len(seg)),
        **metrics,
        **ece_mce(y, p),
        **calibration_alpha_beta(y, p),
        **_lift_metrics(y, p),
        "confidence_intervals": _bootstrap_confidence_intervals(y, p),
        "model_name": model_name,
    }


def headline_row(record: dict[str, Any]) -> dict[str, Any]:
    """Project one detailed metric record onto the stable CSV headline schema."""

    return {column: record.get(column) for column in METRICS_COLUMNS if column not in {"row_type", "target_name", "model_name"}}


def reliability_rows(frame: pd.DataFrame, *, target_name: str, model_name: str) -> list[dict[str, Any]]:
    """Render calibration-bin rows for a model into deterministic CSV records."""

    table = reliability_table(frame["outcome_home_win"].astype(int).to_numpy(), frame["prob_home_win"].astype(float).to_numpy())
    return [{"target_name": target_name, "model_name": model_name, **row} for row in table.to_dict(orient="records")]


def _window_segment(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return the rows included in a helper-produced rolling window label."""

    days = int(label.removesuffix("d"))
    dates = pd.to_datetime(frame["game_date_utc"])
    return frame[dates >= dates.max() - pd.Timedelta(days=days)]


def _lift_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    """Measure whether the model ranks higher-probability games as higher risk."""

    if len(y) < 2:
        return {"top_vs_bottom_actual_lift_ratio": None, "top_quantile_actual_lift": None}
    frame = pd.DataFrame({"y": y.astype(int), "p": p.astype(float)}).sort_values("p")
    bucket_size = max(1, int(np.ceil(len(frame) / 10)))
    bottom = frame.head(bucket_size)["y"].mean()
    top = frame.tail(bucket_size)["y"].mean()
    overall = frame["y"].mean()
    return {
        "top_vs_bottom_actual_lift_ratio": None if bottom == 0 else float(top / bottom),
        "top_quantile_actual_lift": None if overall == 0 else float(top / overall),
    }


def _bootstrap_confidence_intervals(
    y: np.ndarray,
    p: np.ndarray,
    *,
    samples: int = 200,
    seed: int = 42,
) -> dict[str, dict[str, float | None]]:
    """Return deterministic bootstrap confidence intervals for headline metrics."""

    if len(y) < 2:
        return {}
    rng = np.random.default_rng(seed)
    metrics: dict[str, list[float]] = {"log_loss": [], "brier": [], "accuracy": [], "auc": []}
    for _ in range(samples):
        sample_y, sample_p = _bootstrap_sample(y, p, rng=rng)
        bundle = _bootstrap_metric_bundle(sample_y, sample_p)
        for metric_name, value in bundle.items():
            if value is not None and isfinite(float(value)):
                metrics[metric_name].append(float(value))
    return {metric_name: _quantile_interval(values) for metric_name, values in metrics.items()}


def _bootstrap_sample(y: np.ndarray, p: np.ndarray, *, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw one deterministic bootstrap sample from paired outcomes/probabilities."""

    idx = rng.integers(0, len(y), size=len(y))
    return y[idx], np.clip(p[idx], 1e-6, 1 - 1e-6)


def _bootstrap_metric_bundle(sample_y: np.ndarray, sample_p: np.ndarray) -> dict[str, float | None]:
    """Compute bootstrap metrics without emitting one-class AUC warnings."""

    if len(np.unique(sample_y)) >= 2:
        return metric_bundle(sample_y, sample_p)
    return {
        "log_loss": float(-(sample_y * np.log(sample_p) + (1 - sample_y) * np.log(1 - sample_p)).mean()),
        "brier": float(((sample_p - sample_y) ** 2).mean()),
        "accuracy": float(((sample_p >= 0.5).astype(int) == sample_y).mean()),
        "auc": None,
    }


def _quantile_interval(values: list[float]) -> dict[str, float | None]:
    """Convert bootstrap draws into a JSON-safe 95 percent interval."""

    if not values:
        return {"lower": None, "upper": None}
    arr = np.asarray(values, dtype=float)
    return {"lower": float(np.quantile(arr, 0.025)), "upper": float(np.quantile(arr, 0.975))}
