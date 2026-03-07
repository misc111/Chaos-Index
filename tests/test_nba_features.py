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
            "game_id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "team": ["NYK", "NYK", "CHI", "CHI", "CHI", "CHI", "NYK", "NYK", "NYK", "NYK", "CHI", "CHI"],
            "current_team": ["NYK", "NYK", "CHI", "CHI", "CHI", "CHI", "NYK", "NYK", "NYK", "NYK", "CHI", "CHI"],
            "player_id": [
                "10",
                "11",
                "20",
                "21",
                "20",
                "21",
                "10",
                "11",
                "10",
                "11",
                "20",
                "21",
            ],
            "start_time_utc": [
                "2025-10-22T00:00:00Z",
                "2025-10-22T00:00:00Z",
                "2025-10-22T00:00:00Z",
                "2025-10-22T00:00:00Z",
                "2025-10-24T00:00:00Z",
                "2025-10-24T00:00:00Z",
                "2025-10-24T00:00:00Z",
                "2025-10-24T00:00:00Z",
                "2025-10-26T00:00:00Z",
                "2025-10-26T00:00:00Z",
                "2025-10-26T00:00:00Z",
                "2025-10-26T00:00:00Z",
            ],
            "minutes": [34, 30, 28, 24, 29, 23, 35, 31, 36, 32, 27, 22],
            "points": [28, 18, 16, 10, 17, 11, 30, 19, 29, 20, 15, 9],
            "assists": [8, 4, 5, 2, 5, 3, 9, 5, 7, 4, 4, 2],
            "rebounds_offensive": [2, 1, 1, 1, 1, 0, 2, 1, 2, 1, 1, 0],
            "rebounds_defensive": [7, 5, 5, 4, 4, 3, 8, 5, 7, 5, 4, 4],
            "rebounds_total": [9, 6, 6, 5, 5, 3, 10, 6, 9, 6, 5, 4],
            "steals": [2, 1, 1, 0, 1, 0, 2, 1, 1, 1, 1, 0],
            "blocks": [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
            "turnovers": [3, 2, 2, 1, 2, 1, 3, 2, 2, 2, 2, 1],
            "fouls_personal": [2, 2, 3, 2, 3, 2, 2, 2, 2, 2, 3, 2],
            "field_goals_made": [10, 7, 6, 4, 6, 4, 11, 7, 10, 8, 5, 4],
            "field_goals_attempted": [18, 14, 13, 9, 13, 10, 19, 15, 18, 15, 12, 9],
            "free_throws_made": [5, 2, 2, 1, 3, 1, 5, 2, 6, 2, 3, 1],
            "free_throws_attempted": [6, 3, 3, 2, 4, 2, 6, 2, 7, 3, 4, 2],
            "three_pointers_made": [3, 2, 2, 1, 2, 1, 3, 3, 3, 2, 2, 1],
            "plus_minus_points": [8, 5, -4, -6, -3, -7, 10, 6, -2, -4, -5, -8],
            "played": [1] * 12,
            "starter": [1] * 12,
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
    assert "diff_darko_like_total" in cols
    assert "diff_darko_like_total_hinge_000" in cols
    assert "darko_like_off_matchup" in cols
    assert "darko_like_def_matchup" in cols
    assert "diff_projected_absence_pressure" in cols
    assert "diff_rotation_stability" in cols
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

    upcoming = out.dataframe[out.dataframe["game_id"] == 4].iloc[0]
    assert abs(float(upcoming["diff_darko_like_total"])) > 0
    assert int(upcoming["fallback_availability_proxy_used"]) == 0
