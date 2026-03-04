from pathlib import Path

import pandas as pd

from src.bayes.fit_offline import run_bayes_offline_fit



def test_bayes_offline_fit_smoke(tmp_path: Path):
    n = 60
    df = pd.DataFrame(
        {
            "home_team": ["TOR" if i % 2 == 0 else "MTL" for i in range(n)],
            "away_team": ["MTL" if i % 2 == 0 else "TOR" for i in range(n)],
            "start_time_utc": pd.date_range("2025-10-01", periods=n, freq="D").astype(str),
            "home_win": [1 if i % 3 else 0 for i in range(n)],
            "diff_form_goal_diff": [0.1 * ((i % 5) - 2) for i in range(n)],
            "travel_diff": [0.0] * n,
        }
    )
    model, diag = run_bayes_offline_fit(
        df,
        feature_columns=["diff_form_goal_diff", "travel_diff"],
        artifacts_dir=str(tmp_path),
    )
    assert len(model.team_to_ix) == 2
    assert "ppc" in diag
