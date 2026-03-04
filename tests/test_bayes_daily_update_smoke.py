from pathlib import Path

import pandas as pd

from src.models.bayes_state_space_bt import BayesStateSpaceBTModel



def test_bayes_daily_update(tmp_path: Path):
    df = pd.DataFrame(
        {
            "home_team": ["TOR", "MTL", "TOR", "MTL"],
            "away_team": ["MTL", "TOR", "MTL", "TOR"],
            "start_time_utc": ["2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04"],
            "home_win": [1, 0, 1, 0],
            "f1": [0.1, -0.2, 0.05, -0.1],
        }
    )
    m = BayesStateSpaceBTModel()
    m.fit_offline(df, feature_columns=["f1"], n_passes=1)
    p0 = m.predict_proba(df)

    new_df = pd.DataFrame(
        {
            "home_team": ["TOR"],
            "away_team": ["MTL"],
            "start_time_utc": ["2025-10-05"],
            "home_win": [1],
            "f1": [0.2],
        }
    )
    m.daily_update(new_df)
    p1 = m.predict_proba(df)
    assert (p0 != p1).any()
