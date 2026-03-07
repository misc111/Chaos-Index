from pathlib import Path

import pandas as pd
import pytest

from src.features.build_features import build_features_from_interim


def _seed_nhl_inputs(interim: Path) -> None:
    pd.DataFrame(
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
    ).to_csv(interim / "games.csv", index=False)

    pd.DataFrame(
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
    ).to_csv(interim / "goalies.csv", index=False)

    pd.DataFrame(
        {
            "team": ["TOR", "MTL"],
            "games_played": [82, 82],
            "points": [92, 70],
        }
    ).to_csv(interim / "players.csv", index=False)

    pd.DataFrame(
        {
            "team": ["TOR", "MTL"],
            "lineup_uncertainty": [0.15, 0.35],
            "man_games_lost_proxy": [1.0, 3.0],
        }
    ).to_csv(interim / "injuries.csv", index=False)


def _seed_nba_inputs(interim: Path) -> None:
    pd.DataFrame(
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
    ).to_csv(interim / "games.csv", index=False)

    pd.DataFrame(
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
    ).to_csv(interim / "goalies.csv", index=False)

    pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "team": ["NYK", "NYK", "CHI", "CHI", "CHI", "CHI", "NYK", "NYK", "NYK", "NYK", "CHI", "CHI"],
            "current_team": ["NYK", "NYK", "CHI", "CHI", "CHI", "CHI", "NYK", "NYK", "NYK", "NYK", "CHI", "CHI"],
            "player_id": ["10", "11", "20", "21", "20", "21", "10", "11", "10", "11", "20", "21"],
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


@pytest.mark.parametrize(
    ("league", "seed_inputs", "must_have", "forbidden_tokens"),
    [
        (
            "NHL",
            _seed_nhl_inputs,
            ("diff_xg_share_cubic", "diff_form_goal_diff_hinge_m1", "dyn_home_prob_hinge_055"),
            ("darko", "arena_margin_effect"),
        ),
        (
            "NBA",
            _seed_nba_inputs,
            ("diff_form_point_margin", "diff_darko_like_total_hinge_000", "arena_margin_effect"),
            ("goalie", "rink", "xg"),
        ),
    ],
    ids=["nhl", "nba"],
)
def test_feature_pipeline_shared_stages_hold_cross_league_contract(
    tmp_path: Path,
    league: str,
    seed_inputs,
    must_have: tuple[str, ...],
    forbidden_tokens: tuple[str, ...],
) -> None:
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    interim.mkdir(parents=True)

    seed_inputs(interim)
    out = build_features_from_interim(str(interim), str(processed), league=league)

    assert not out.dataframe.empty
    assert out.metadata["league"] == league
    assert out.feature_set_version.startswith("fset_")
    assert Path(out.metadata["saved_path"]).exists()

    cols = set(out.feature_columns)
    for col in must_have:
        assert col in cols
    for token in forbidden_tokens:
        assert not any(token in col for col in cols)
