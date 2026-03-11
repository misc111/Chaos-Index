"""Causal venue/rink effects that only use information available at the time."""

from __future__ import annotations

import pandas as pd


def compute_causal_group_effects(
    games_df: pd.DataFrame,
    *,
    group_col: str,
    metric_columns: dict[str, str],
    shrinkage: float,
    finalized_col: str = "status_final",
) -> pd.DataFrame:
    output_columns = ["game_id", *metric_columns.keys()]
    if games_df.empty:
        return pd.DataFrame(columns=output_columns)

    tmp = games_df.copy()
    start_time = pd.to_datetime(
        tmp.get("start_time_utc", pd.Series(index=tmp.index, dtype=object)),
        utc=True,
        errors="coerce",
    )
    game_date = pd.to_datetime(
        tmp.get("game_date_utc", pd.Series(index=tmp.index, dtype=object)),
        utc=True,
        errors="coerce",
    )
    tmp["__effect_sort_ts"] = start_time.fillna(game_date).fillna(pd.Timestamp("1970-01-01T00:00:00Z"))
    tmp["__effect_is_final"] = pd.to_numeric(
        tmp.get(finalized_col, pd.Series(index=tmp.index, dtype=float)),
        errors="coerce",
    ).fillna(0).clip(lower=0, upper=1)

    tmp = tmp.sort_values([group_col, "__effect_sort_ts", "game_id"], kind="mergesort").copy()
    grouped = tmp.groupby(group_col, dropna=False, sort=False)
    tmp["__past_count"] = grouped["__effect_is_final"].cumsum() - tmp["__effect_is_final"]
    credibility = tmp["__past_count"] / (tmp["__past_count"] + float(shrinkage))

    for output_col, source_col in metric_columns.items():
        source_values = pd.to_numeric(
            tmp.get(source_col, pd.Series(index=tmp.index, dtype=float)),
            errors="coerce",
        ).fillna(0.0)
        current_totals_col = f"__{output_col}_current_total"
        past_sum_col = f"__{output_col}_past_sum"
        tmp[current_totals_col] = source_values * tmp["__effect_is_final"]
        tmp[past_sum_col] = tmp.groupby(group_col, dropna=False, sort=False)[current_totals_col].cumsum() - tmp[current_totals_col]
        past_mean = tmp[past_sum_col] / tmp["__past_count"].replace(0, float("nan"))
        tmp[output_col] = past_mean.fillna(0.0) * credibility.fillna(0.0)

    return tmp[output_columns]
