from __future__ import annotations

import numpy as np
import pandas as pd



def time_series_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    min_train_size: int = 200,
) -> list[tuple[list[int], list[int]]]:
    work = df[df["home_win"].notna()].copy().sort_values("start_time_utc")
    idx = work.index.tolist()
    n = len(idx)
    if n <= min_train_size + n_splits:
        # fallback: one holdout split
        cut = max(min_train_size, int(n * 0.7))
        if cut >= n:
            return []
        return [(idx[:cut], idx[cut:])]

    fold = max((n - min_train_size) // n_splits, 1)
    splits: list[tuple[list[int], list[int]]] = []
    for i in range(n_splits):
        train_end = min_train_size + i * fold
        val_end = min(train_end + fold, n)
        if val_end - train_end < 15:
            continue
        splits.append((idx[:train_end], idx[train_end:val_end]))
    return splits


def expanding_window_date_splits(
    df: pd.DataFrame,
    *,
    n_splits: int,
    validation_days: int,
    embargo_days: int = 0,
    min_train_days: int | None = None,
    date_col: str = "start_time_utc",
) -> list[tuple[list[int], list[int], dict[str, str]]]:
    work = df[df["home_win"].notna()].copy()
    if work.empty or date_col not in work.columns:
        return []

    work[date_col] = pd.to_datetime(work[date_col], errors="coerce", utc=True)
    work = work[work[date_col].notna()].sort_values(date_col)
    if work.empty:
        return []

    unique_dates = sorted(pd.Series(work[date_col].dt.normalize().unique()).tolist())
    if len(unique_dates) < 3:
        return []

    validation_days = max(1, int(validation_days))
    embargo_days = max(0, int(embargo_days))
    min_train_days = max(validation_days * 2, 90) if min_train_days is None else max(1, int(min_train_days))
    first_train_end_date = unique_dates[0] + pd.Timedelta(days=min_train_days - 1)
    last_train_end_date = unique_dates[-1] - pd.Timedelta(days=validation_days + embargo_days)
    candidate_train_end_dates = [date for date in unique_dates if first_train_end_date <= date <= last_train_end_date]
    if not candidate_train_end_dates:
        return []

    if len(candidate_train_end_dates) <= n_splits:
        selected_train_end_dates = candidate_train_end_dates
    else:
        positions = pd.Series(np.linspace(0, len(candidate_train_end_dates) - 1, num=n_splits)).round().astype(int)
        selected_train_end_dates = [candidate_train_end_dates[int(position)] for position in positions.drop_duplicates().tolist()]

    splits: list[tuple[list[int], list[int], dict[str, str]]] = []
    for train_end_date in selected_train_end_dates:
        validation_start_date = train_end_date + pd.Timedelta(days=embargo_days + 1)
        validation_end_date = validation_start_date + pd.Timedelta(days=validation_days - 1)
        train_idx = work.index[work[date_col].dt.normalize() <= train_end_date].tolist()
        valid_idx = work.index[
            (work[date_col].dt.normalize() >= validation_start_date)
            & (work[date_col].dt.normalize() <= validation_end_date)
        ].tolist()
        if not train_idx or not valid_idx:
            continue
        splits.append(
            (
                train_idx,
                valid_idx,
                {
                    "train_end": str(train_end_date.date()),
                    "valid_start": str(validation_start_date.date()),
                    "valid_end": str(validation_end_date.date()),
                },
            )
        )
    return splits
