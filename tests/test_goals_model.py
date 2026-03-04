import pandas as pd

from src.models.glm_goals import GoalsPoissonModel



def test_goals_model_prob_bounds():
    df = pd.DataFrame(
        {
            "home_team": ["TOR", "MTL", "TOR", "MTL"],
            "away_team": ["MTL", "TOR", "MTL", "TOR"],
            "home_score": [3, 1, 2, 4],
            "away_score": [2, 2, 1, 1],
            "home_win": [1, 0, 1, 1],
        }
    )
    m = GoalsPoissonModel()
    m.fit(df)
    p = m.predict_proba(df)
    assert ((p > 0) & (p < 1)).all()
