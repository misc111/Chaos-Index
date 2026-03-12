import pandas as pd

from src.training.cv import expanding_window_date_splits, time_series_splits



def test_time_series_splits_monotonic():
    n = 300
    df = pd.DataFrame(
        {
            "home_win": [i % 2 for i in range(n)],
            "start_time_utc": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
        }
    )
    splits = time_series_splits(df, n_splits=4, min_train_size=120)
    assert len(splits) >= 1
    for tr, va in splits:
        assert max(tr) < min(va)


def test_expanding_window_date_splits_respect_embargo_and_window():
    n = 320
    df = pd.DataFrame(
        {
            "home_win": [i % 2 for i in range(n)],
            "start_time_utc": pd.date_range("2024-10-01", periods=n, freq="D").astype(str),
        }
    )
    splits = expanding_window_date_splits(
        df,
        n_splits=4,
        validation_days=30,
        embargo_days=2,
        min_train_days=120,
    )
    assert len(splits) >= 1
    for train_idx, valid_idx, bounds in splits:
        train_end = pd.Timestamp(bounds["train_end"])
        valid_start = pd.Timestamp(bounds["valid_start"])
        valid_end = pd.Timestamp(bounds["valid_end"])
        assert max(train_idx) < min(valid_idx)
        assert (valid_start - train_end).days == 3
        assert (valid_end - valid_start).days == 29
