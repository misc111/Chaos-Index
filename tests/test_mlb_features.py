from pathlib import Path

import pandas as pd

from src.features.build_features import build_features_from_interim


def _write_mlb_interim(interim: Path) -> None:
    games = pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4],
            "season": [2026] * 4,
            "game_date_utc": ["2026-04-10", "2026-04-11", "2026-04-12", "2026-04-13"],
            "start_time_utc": [
                "2026-04-10T18:05:00Z",
                "2026-04-11T18:05:00Z",
                "2026-04-12T18:05:00Z",
                "2026-04-13T18:05:00Z",
            ],
            "venue": ["Fenway Park", "Yankee Stadium", "Fenway Park", "Yankee Stadium"],
            "home_team": ["BOS", "NYY", "BOS", "NYY"],
            "away_team": ["NYY", "BOS", "NYY", "BOS"],
            "home_score": [5, 3, 4, None],
            "away_score": [2, 4, 3, None],
            "status_final": [1, 1, 1, 0],
            "home_win": [1, 0, 1, None],
            "as_of_utc": ["2026-04-09T12:00:00Z"] * 4,
        }
    )
    games.to_csv(interim / "games.csv", index=False)

    team_stats = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2, 3, 3],
            "team": ["BOS", "NYY", "NYY", "BOS", "BOS", "NYY"],
            "hits": [9, 6, 7, 8, 8, 7],
            "walks": [3, 2, 3, 2, 2, 4],
            "strikeouts": [10, 8, 9, 7, 8, 9],
            "home_runs": [2, 1, 1, 1, 1, 1],
            "total_bases": [17, 10, 12, 13, 14, 11],
            "stolen_bases": [1, 0, 1, 1, 0, 1],
        }
    )
    team_stats.to_csv(interim / "team_stats.csv", index=False)

    starting_pitchers = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2, 3, 3],
            "team": ["BOS", "NYY", "NYY", "BOS", "BOS", "NYY"],
            "starter_pitcher_id": ["bos_sp_1", "nyy_sp_1", "nyy_sp_2", "bos_sp_2", "bos_sp_3", "nyy_sp_3"],
            "starter_pitcher_name": ["Bello", "Cole", "Schmidt", "Houck", "Crawford", "Rodon"],
            "starter_status": ["actual"] * 6,
            "innings_pitched": [6.0, 5.0, 5.3333333333, 6.6666666667, 5.6666666667, 4.6666666667],
            "starter_era": [3.2, 2.9, 3.6, 3.4, 3.1, 4.0],
            "starter_whip": [1.12, 1.01, 1.18, 1.09, 1.05, 1.24],
            "pitcher_strikeouts": [7, 8, 6, 5, 6, 4],
            "pitcher_walks": [2, 1, 2, 2, 1, 3],
            "runs_allowed": [2, 5, 4, 3, 3, 4],
        }
    )
    starting_pitchers.to_csv(interim / "starting_pitchers.csv", index=False)

    players = pd.DataFrame(
        {
            "game_id": [4] * 17,
            "team": ["BOS"] * 9 + ["NYY"] * 8,
            "player_id": [f"bos_{i}" for i in range(1, 10)] + [f"nyy_{i}" for i in range(1, 9)],
            "batting_order_slot": list(range(1, 10)) + list(range(1, 9)),
            "lineup_confirmed": [1] * 9 + [0] * 8,
        }
    )
    players.to_csv(interim / "players.csv", index=False)

    injuries = pd.DataFrame(
        {
            "game_id": [4, 4],
            "team": ["BOS", "NYY"],
            "position_player_out_count": [1, 2],
            "pitcher_out_count": [0, 1],
        }
    )
    injuries.to_csv(interim / "injuries.csv", index=False)


def test_mlb_feature_build_smoke_uses_baseball_features(tmp_path: Path) -> None:
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    interim.mkdir(parents=True)
    _write_mlb_interim(interim)

    out = build_features_from_interim(str(interim), str(processed), league="MLB")

    assert not out.dataframe.empty
    assert out.metadata["league"] == "MLB"
    cols = set(out.feature_columns)
    assert "diff_form_run_diff" in cols
    assert "diff_starter_era" in cols
    assert "starter_quality_edge" in cols
    assert "diff_lineup_availability" in cols
    assert "lineup_quality_edge" in cols
    assert "lineup_confirmed_quality_edge" in cols
    assert "diff_starting_pitcher_quality" in cols
    assert "diff_lineup_talent" in cols
    assert "lineup_stability_diff" in cols
    assert "diff_bullpen_quality" in cols
    assert "team_strength_edge" in cols
    assert "rest_edge" in cols
    assert "travel_miles_edge" in cols
    assert "bullpen_quality_edge" in cols
    assert "home_field_advantage" in cols
    assert "diff_bullpen_availability" in cols
    assert "home_lineup_confirmed" in cols
    assert "away_lineup_confirmed" in cols
    assert "home_starter_confirmed" in cols
    assert "away_starter_confirmed" in cols
    assert "diff_form_run_diff_hinge_000" in cols
    assert "diff_starter_era_hinge_000" in cols
    assert "elo_home_prob_hinge_052" in cols
    assert "travel_diff" in cols
    assert not any("goalie" in c for c in cols)
    assert not any("xg" in c for c in cols)
    assert not any("darko" in c for c in cols)
    assert not any("arena" in c for c in cols)

    row = out.dataframe[out.dataframe["game_id"] == 4].iloc[0]
    assert float(row["starter_quality_edge"]) == -float(row["diff_starter_era"])
    assert float(row["diff_starting_pitcher_quality"]) == float(row["starter_quality_edge"])
    assert float(row["lineup_quality_edge"]) == float(row["diff_lineup_availability"])
    assert float(row["diff_lineup_talent"]) == float(row["lineup_quality_edge"])
    assert float(row["lineup_stability_diff"]) == float(row["lineup_confirmed_quality_edge"])
    assert float(row["diff_bullpen_quality"]) == float(row["diff_bullpen_availability"])
    assert float(row["team_strength_edge"]) == float(row["diff_form_win_rate"])
    assert float(row["rest_edge"]) == float(row["rest_diff"])
    assert float(row["travel_miles_edge"]) == float(row["travel_diff"])
    assert float(row["bullpen_quality_edge"]) == float(row["diff_bullpen_quality"])
    assert float(row["home_field_advantage"]) == 1.0
    assert float(row["lineup_confirmed_quality_edge"]) == float(row["diff_lineup_availability"]) * float(
        row["lineup_confirmation_share"]
    )


def test_mlb_feature_build_carries_availability_metadata_into_upcoming_game(tmp_path: Path) -> None:
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    interim.mkdir(parents=True)
    _write_mlb_interim(interim)

    out = build_features_from_interim(str(interim), str(processed), league="MLB")

    upcoming = out.dataframe[out.dataframe["game_id"] == 4].iloc[0]
    assert upcoming["home_team"] == "NYY"
    assert upcoming["away_team"] == "BOS"
    assert float(upcoming["home_lineup_availability"]) == 8.0 / 9.0
    assert float(upcoming["away_lineup_availability"]) == 1.0
    assert int(upcoming["home_lineup_confirmed"]) == 0
    assert int(upcoming["away_lineup_confirmed"]) == 1
    assert int(upcoming["home_starter_confirmed"]) == 0
    assert int(upcoming["away_starter_confirmed"]) == 0
    assert float(upcoming["diff_bullpen_availability"]) < 0


def test_mlb_feature_build_emits_available_as_of_contract(tmp_path: Path) -> None:
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    interim.mkdir(parents=True)
    _write_mlb_interim(interim)

    out = build_features_from_interim(str(interim), str(processed), league="MLB")

    assert "available_as_of_utc" in out.dataframe.columns
    available = pd.to_datetime(out.dataframe["available_as_of_utc"], errors="coerce", utc=True)
    feature_as_of = pd.to_datetime(out.dataframe["as_of_utc"], errors="coerce", utc=True)
    start = pd.to_datetime(out.dataframe["start_time_utc"], errors="coerce", utc=True)
    assert available.notna().all()
    assert available.le(feature_as_of).all()
    assert available.le(start).all()
