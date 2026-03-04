from __future__ import annotations

import pandas as pd

from src.evaluation.calibration import calibration_alpha_beta, ece_mce
from src.evaluation.metrics import metric_bundle



def _aggregate_segment(seg: pd.DataFrame, model_name: str, window_label: str, as_of_utc: str) -> dict:
    y = seg["outcome_home_win"].to_numpy()
    p = seg["prob_home_win"].to_numpy()
    m = metric_bundle(y, p)
    c = ece_mce(y, p)
    ab = calibration_alpha_beta(y, p)
    return {
        "as_of_utc": as_of_utc,
        "model_name": model_name,
        "window_label": window_label,
        "start_date": str(pd.to_datetime(seg["game_date_utc"]).min().date()) if not seg.empty else None,
        "end_date": str(pd.to_datetime(seg["game_date_utc"]).max().date()) if not seg.empty else None,
        "n_games": int(len(seg)),
    } | m | c | ab



def compute_performance_aggregates(scores_df: pd.DataFrame, as_of_utc: str, windows_days: list[int]) -> pd.DataFrame:
    if scores_df.empty:
        return pd.DataFrame()

    work = scores_df.copy()
    work["game_date_utc"] = pd.to_datetime(work["game_date_utc"])
    out = []

    for model_name, grp in work.groupby("model_name"):
        grp = grp.sort_values("game_date_utc")
        if len(grp) >= 1:
            out.append(_aggregate_segment(grp, model_name, "cumulative", as_of_utc))

        max_date = grp["game_date_utc"].max()
        for w in windows_days:
            start = max_date - pd.Timedelta(days=w)
            seg = grp[grp["game_date_utc"] >= start]
            if len(seg) < 1:
                continue
            out.append(_aggregate_segment(seg, model_name, f"{w}d", as_of_utc))

    return pd.DataFrame(out)
