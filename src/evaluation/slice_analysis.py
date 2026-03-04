from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import metric_bundle



def run_slice_analysis(df: pd.DataFrame, p_col: str, y_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    work["favorite_bucket"] = (work[p_col] >= 0.6).map({True: "home_favorite", False: "home_not_favorite"})
    work["goalie_certainty_bucket"] = (work.get("home_goalie_uncertainty_feature", 1) + work.get("away_goalie_uncertainty_feature", 1) == 0).map(
        {True: "goalie_known", False: "goalie_uncertain"}
    )
    work["travel_bucket"] = (work.get("travel_diff", 0).abs() > 600).map({True: "high_travel", False: "normal_travel"})
    work["season_phase_bucket"] = work.get("home_season_phase", "unknown")

    slices = []
    for col in ["favorite_bucket", "goalie_certainty_bucket", "travel_bucket", "season_phase_bucket"]:
        for val, grp in work.groupby(col):
            if len(grp) < 8:
                continue
            m = metric_bundle(grp[y_col].to_numpy(), grp[p_col].to_numpy())
            slices.append({"slice_col": col, "slice_val": val, "n": len(grp)} | m)
    return pd.DataFrame(slices)
