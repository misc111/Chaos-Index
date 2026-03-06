from pathlib import Path

import pandas as pd

from src.features.build_features import build_features_from_interim



def test_feature_build_smoke(tmp_path: Path):
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    interim.mkdir(parents=True)

    games = pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4],
            "season": [20252026] * 4,
            "game_date_utc": ["2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04"],
            "start_time_utc": [
                "2025-10-01T00:00:00Z",
                "2025-10-02T00:00:00Z",
                "2025-10-03T00:00:00Z",
                "2025-10-04T00:00:00Z",
            ],
            "venue": ["A", "B", "A", "B"],
            "home_team": ["TOR", "MTL", "TOR", "MTL"],
            "away_team": ["MTL", "TOR", "MTL", "TOR"],
            "home_score": [3, 1, 2, None],
            "away_score": [2, 2, 4, None],
            "status_final": [1, 1, 1, 0],
            "home_win": [1, 0, 0, None],
            "as_of_utc": ["2025-10-01T00:00:00Z"] * 4,
        }
    )
    games.to_csv(interim / "games.csv", index=False)

    goalies = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2, 3, 3],
            "team": ["TOR", "MTL", "MTL", "TOR", "TOR", "MTL"],
            "goalie_id": [None] * 6,
            "shots_for": [30, 29, 24, 31, 27, 32],
            "shots_against": [29, 30, 31, 24, 32, 27],
            "penalties_taken": [2, 3, 2, 2, 1, 3],
            "penalties_drawn": [3, 2, 2, 2, 3, 1],
            "pp_goals": [1, 0, 0, 1, 1, 0],
            "starter_status": ["unknown"] * 6,
            "starter_save_pct": [None] * 6,
        }
    )
    goalies.to_csv(interim / "goalies.csv", index=False)

    pd.DataFrame(columns=["team", "games_played", "points"]).to_csv(interim / "players.csv", index=False)
    pd.DataFrame(columns=["team", "lineup_uncertainty", "man_games_lost_proxy"]).to_csv(interim / "injuries.csv", index=False)

    out = build_features_from_interim(str(interim), str(processed))
    assert not out.dataframe.empty
    assert len(out.feature_columns) > 5
    cols = set(out.feature_columns)
    assert "diff_xg_share_cubic" in cols
    assert "diff_form_goal_diff_hinge_m1" in cols
    assert "diff_form_goal_diff_hinge_p1" in cols
    assert "dyn_home_prob_hinge_055" in cols
    assert "dyn_home_mean_hinge_000" in cols
    assert "elo_home_prob_hinge_054" in cols
