from __future__ import annotations

import pandas as pd

from src.evaluation.calibration import calibration_alpha_beta, ece_mce
from src.evaluation.metrics import metric_bundle



def rolling_drift_table(df: pd.DataFrame, p_col: str, y_col: str, date_col: str, windows: list[int]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work = work.sort_values(date_col)
    out = []

    for w in windows:
        for ix in range(len(work)):
            start = max(0, ix - w + 1)
            seg = work.iloc[start : ix + 1]
            if len(seg) < 8:
                continue
            m = metric_bundle(seg[y_col].to_numpy(), seg[p_col].to_numpy())
            c = ece_mce(seg[y_col].to_numpy(), seg[p_col].to_numpy())
            ab = calibration_alpha_beta(seg[y_col].to_numpy(), seg[p_col].to_numpy())
            out.append(
                {
                    "as_of": seg.iloc[-1][date_col],
                    "window_games": w,
                }
                | m
                | c
                | ab
            )

    return pd.DataFrame(out)
