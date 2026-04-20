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


def test_leakage_rejects_post_start_availability():
    df = pd.DataFrame(
        {
            "game_id": [11],
            "available_as_of_utc": ["2026-04-20T19:00:00Z"],
            "start_time_utc": ["2026-04-20T18:05:00Z"],
            "feature_value": [1.0],
        }
    )

    issues = run_leakage_checks(df)

    assert any("available_as_of_after_start_count=1" in x for x in issues)


def test_leakage_rejects_mlb_postgame_outcome_columns():
    df = pd.DataFrame(
        {
            "game_id": [12],
            "available_as_of_utc": ["2026-04-20T17:30:00Z"],
            "start_time_utc": ["2026-04-20T18:05:00Z"],
            "home_runs": [2],
            "away_runs": [3],
            "run_diff": [-1],
            "winner": ["away"],
            "feature_value": [1.0],
        }
    )

    issues = run_leakage_checks(
        df,
        feature_columns=["feature_value", "home_runs", "away_runs", "run_diff", "winner", "available_as_of_utc", "start_time_utc"],
    )

    assert any("pattern_forbidden_columns" in x for x in issues)
    assert any("home_runs" in x for x in issues)
    assert any("run_diff" in x for x in issues)
    assert any("winner" in x for x in issues)
