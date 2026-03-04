import pandas as pd

from src.features.leakage_checks import run_leakage_checks


def test_leakage_detects_forbidden_columns():
    df = pd.DataFrame(
        {
            "game_id": [1],
            "status_final": [1],
            "home_win": [1],
            "home_score": [3],
            "as_of_utc": ["2026-01-01T00:00:00+00:00"],
            "start_time_utc": ["2026-01-02T00:00:00+00:00"],
            "home_games_played_prior": [2],
        }
    )
    issues = run_leakage_checks(df)
    assert any("forbidden_columns_present" in x for x in issues)
