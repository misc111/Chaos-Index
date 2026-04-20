from __future__ import annotations

import pandas as pd

from src.features.travel import build_travel_features


def test_mlb_travel_features_compute_real_city_travel_and_timezone_shift() -> None:
    games = pd.DataFrame(
        {
            "game_id": [101, 102, 103],
            "start_time_utc": [
                "2026-04-01T13:05:00Z",
                "2026-04-03T02:10:00Z",
                "2026-04-05T23:10:00Z",
            ],
            "home_team": ["BOS", "SEA", "SEA"],
            "away_team": ["NYY", "BOS", "NYY"],
        }
    )

    travel = build_travel_features(games, league="MLB").set_index("game_id")

    assert float(travel.loc[102, "away_travel_miles"]) > 2000.0
    assert int(travel.loc[102, "away_tz_change"]) == -3
    assert float(travel.loc[103, "home_travel_miles"]) == 0.0
    assert int(travel.loc[101, "home_local_start_mismatch"]) == 1
    assert int(travel.loc[101, "away_local_start_mismatch"]) == 1
