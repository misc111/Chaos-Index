from pathlib import Path

import pandas as pd

from src.features.build_features import build_features_from_interim


def test_nba_feature_build_uses_basketball_feature_names(tmp_path: Path) -> None:
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    interim.mkdir(parents=True)

    games = pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4],
            "season": [20252026] * 4,
            "game_date_utc": ["2025-10-22", "2025-10-24", "2025-10-26", "2025-10-28"],
            "start_time_utc": [
                "2025-10-22T00:00:00Z",
                "2025-10-24T00:00:00Z",
                "2025-10-26T00:00:00Z",
                "2025-10-28T00:00:00Z",
            ],
            "venue": ["A", "B", "A", "B"],
            "home_team": ["NYK", "CHI", "NYK", "CHI"],
            "away_team": ["CHI", "NYK", "CHI", "NYK"],
            "home_score": [112, 101, 118, None],
            "away_score": [104, 109, 121, None],
            "status_final": [1, 1, 1, 0],
            "home_win": [1, 0, 0, None],
            "as_of_utc": ["2025-10-28T00:00:00Z"] * 4,
        }
    )
    games.to_csv(interim / "games.csv", index=False)

    boxscore = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2, 3, 3],
            "team": ["NYK", "CHI", "CHI", "NYK", "NYK", "CHI"],
            "goalie_id": [None] * 6,
            "shots_for": [87, 84, 82, 89, 91, 88],
            "shots_against": [84, 87, 89, 82, 88, 91],
            "penalties_taken": [18, 21, 19, 20, 17, 22],
            "penalties_drawn": [21, 18, 20, 19, 22, 17],
            "pp_goals": [18, 14, 16, 20, 19, 17],
            "starter_status": ["unknown"] * 6,
            "starter_save_pct": [None] * 6,
        }
    )
    boxscore.to_csv(interim / "goalies.csv", index=False)

    pd.DataFrame(
        {
            "team": ["NYK", "CHI"],
            "player_id": [1, 2],
        }
    ).to_csv(interim / "players.csv", index=False)
    pd.DataFrame(
        {
            "team": ["NYK", "CHI"],
            "lineup_uncertainty": [0.2, 0.3],
            "man_games_lost_proxy": [1.0, 2.0],
        }
    ).to_csv(interim / "injuries.csv", index=False)

    out = build_features_from_interim(str(interim), str(processed), league="NBA")
    cols = set(out.feature_columns)

    assert "diff_form_point_margin" in cols
    assert "diff_form_point_margin_hinge_000" in cols
    assert "discipline_free_throw_pressure_diff" in cols
    assert "discipline_free_throw_pressure_diff_hinge_000" in cols
    assert "discipline_free_throw_pressure_diff_is_zero" in cols
    assert "discipline_foul_margin_diff_hinge_000" in cols
    assert "elo_home_prob_hinge_055" in cols
    assert "diff_shot_volume_share_hinge_001" in cols
    assert "availability_stress_diff" in cols
    assert "arena_margin_effect" in cols
    assert "travel_diff" in cols
    assert not any("goalie" in c for c in cols)
    assert not any("xg" in c for c in cols)
    assert not any("pp_" in c for c in cols)
    assert not any("penalty" in c for c in cols)
    assert not any("rink" in c for c in cols)
