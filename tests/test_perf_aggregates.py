import pandas as pd

from src.evaluation.performance_timeseries import compute_performance_aggregates



def test_perf_aggregates_windows():
    df = pd.DataFrame(
        {
            "model_name": ["m1"] * 12,
            "game_date_utc": pd.date_range("2026-01-01", periods=12, freq="D"),
            "outcome_home_win": [i % 2 for i in range(12)],
            "prob_home_win": [0.55] * 12,
        }
    )
    out = compute_performance_aggregates(df, as_of_utc="2026-01-20T00:00:00Z", windows_days=[7])
    assert not out.empty
    assert {"cumulative", "7d"}.issubset(set(out["window_label"]))
