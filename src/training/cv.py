from __future__ import annotations

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
