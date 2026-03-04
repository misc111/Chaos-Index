import pandas as pd

from src.evaluation.change_detection import detect_change_points



def test_change_detection_finds_shift():
    vals = [0.1] * 20 + [0.9] * 20
    df = pd.DataFrame(
        {
            "model_name": ["m"] * len(vals),
            "game_date_utc": pd.date_range("2026-01-01", periods=len(vals), freq="D"),
            "log_loss": vals,
        }
    )
    cp = detect_change_points(df, metric_col="log_loss")
    assert not cp.empty
