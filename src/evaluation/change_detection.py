from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _series_scale(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.05
    scale = float(np.nanstd(arr))
    if not np.isfinite(scale) or scale <= 0:
        return 0.05
    return max(scale, 0.05)


def _burn_in_for(values: Sequence[float]) -> int:
    n = len(values)
    return max(20, min(60, n // 5))


def _cooldown_for(values: Sequence[float]) -> int:
    n = len(values)
    return max(10, min(30, n // 12 if n else 10))


def page_hinkley(
    values: Sequence[float],
    *,
    delta: float | None = None,
    threshold: float | None = None,
    burn_in: int | None = None,
    cooldown: int | None = None,
) -> list[dict[str, float | int]]:
    scale = _series_scale(values)
    delta_value = float(delta if delta is not None else max(0.01, 0.05 * scale))
    threshold_value = float(threshold if threshold is not None else max(0.35, 2.5 * scale))
    burn_in_value = int(burn_in if burn_in is not None else _burn_in_for(values))
    cooldown_value = int(cooldown if cooldown is not None else _cooldown_for(values))
    mean = 0.0
    cum = 0.0
    min_cum = 0.0
    last_alert_idx = -cooldown_value
    alerts: list[dict[str, float | int]] = []

    for i, x in enumerate(values):
        mean = mean + (x - mean) / (i + 1)
        cum += x - mean - delta_value
        min_cum = min(min_cum, cum)
        excursion = cum - min_cum
        if i + 1 < burn_in_value:
            continue
        if excursion > threshold_value and i - last_alert_idx >= cooldown_value:
            alerts.append(
                {
                    "index": i,
                    "value": float(x),
                    "statistic": float(excursion),
                    "threshold": threshold_value,
                }
            )
            last_alert_idx = i
            cum = 0.0
            min_cum = 0.0
    return alerts


def cusum(
    values: Sequence[float],
    *,
    k: float | None = None,
    h: float | None = None,
    burn_in: int | None = None,
    cooldown: int | None = None,
) -> list[dict[str, float | int]]:
    if not values:
        return []
    scale = _series_scale(values)
    k_value = float(k if k is not None else max(0.02, 0.20 * scale))
    h_value = float(h if h is not None else max(0.45, 3.0 * scale))
    burn_in_value = int(burn_in if burn_in is not None else _burn_in_for(values))
    cooldown_value = int(cooldown if cooldown is not None else _cooldown_for(values))
    target = sum(values) / max(len(values), 1)
    s_pos = 0.0
    s_neg = 0.0
    last_alert_idx = -cooldown_value
    alerts: list[dict[str, float | int]] = []
    for i, x in enumerate(values):
        s_pos = max(0.0, s_pos + x - target - k_value)
        s_neg = min(0.0, s_neg + x - target + k_value)
        statistic = max(s_pos, abs(s_neg))
        if i + 1 < burn_in_value:
            continue
        if statistic > h_value and i - last_alert_idx >= cooldown_value:
            alerts.append(
                {
                    "index": i,
                    "value": float(x),
                    "statistic": float(statistic),
                    "threshold": h_value,
                }
            )
            last_alert_idx = i
            s_pos = 0.0
            s_neg = 0.0
    return alerts


def detect_change_points(scores_df: pd.DataFrame, metric_col: str = "log_loss") -> pd.DataFrame:
    if scores_df.empty or metric_col not in scores_df.columns:
        return pd.DataFrame(columns=["model_name", "metric_name", "method", "index", "date", "value"])

    out = []
    for model_name, grp in scores_df.groupby("model_name"):
        g = grp.sort_values("game_date_utc")
        vals = g[metric_col].astype(float).tolist()
        if len(vals) < 30:
            continue
        ph = page_hinkley(vals)
        cs = cusum(vals)
        for alert in ph:
            idx = int(alert["index"])
            out.append(
                {
                    "model_name": model_name,
                    "metric_name": metric_col,
                    "method": "page_hinkley",
                    "index": idx,
                    "date": str(g.iloc[idx]["game_date_utc"]),
                    "value": float(alert["value"]),
                    "threshold": float(alert["threshold"]),
                    "statistic": float(alert["statistic"]),
                    "detected": 1,
                }
            )
        for alert in cs:
            idx = int(alert["index"])
            out.append(
                {
                    "model_name": model_name,
                    "metric_name": metric_col,
                    "method": "cusum",
                    "index": idx,
                    "date": str(g.iloc[idx]["game_date_utc"]),
                    "value": float(alert["value"]),
                    "threshold": float(alert["threshold"]),
                    "statistic": float(alert["statistic"]),
                    "detected": 1,
                }
            )
    return pd.DataFrame(out)
