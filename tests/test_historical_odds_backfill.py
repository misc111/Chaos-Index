from __future__ import annotations

import pandas as pd

from src.services.historical_odds_backfill import _load_games_frame
from src.services.ingest import upsert_games
from src.storage.db import Database


def test_load_games_frame_accepts_mixed_iso_start_time_formats(tmp_path) -> None:
    db = Database(str(tmp_path / "mlb.db"))
    db.init_schema()
    upsert_games(
        db,
        pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "season": 2026,
                    "game_date_utc": "2026-02-20",
                    "start_time_utc": "2026-02-20T18:05Z",
                    "game_state": "Final",
                    "home_team": "BOS",
                    "away_team": "NYY",
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "venue": "Fenway Park",
                    "is_neutral_site": 0,
                    "home_score": 5,
                    "away_score": 2,
                    "went_ot": 0,
                    "went_so": 0,
                    "home_win": 1,
                    "status_final": 1,
                    "as_of_utc": "2026-02-20T22:05:00Z",
                },
                {
                    "game_id": 2,
                    "season": 2026,
                    "game_date_utc": "2026-02-21",
                    "start_time_utc": "2026-02-21T18:05:00Z",
                    "game_state": "Final",
                    "home_team": "NYY",
                    "away_team": "BOS",
                    "home_team_id": 2,
                    "away_team_id": 1,
                    "venue": "Yankee Stadium",
                    "is_neutral_site": 0,
                    "home_score": 3,
                    "away_score": 4,
                    "went_ot": 0,
                    "went_so": 0,
                    "home_win": 0,
                    "status_final": 1,
                    "as_of_utc": "2026-02-21T22:05:00Z",
                },
            ]
        ),
        league="MLB",
    )

    frame = _load_games_frame(db, seasons=[2026])

    assert frame["game_id"].tolist() == [1, 2]
    assert frame["start_time_utc"].notna().all()

