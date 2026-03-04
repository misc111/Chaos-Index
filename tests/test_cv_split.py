import pandas as pd

from src.training.cv import time_series_splits



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
