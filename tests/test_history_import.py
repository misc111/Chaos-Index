import json

import pandas as pd

from src.common.config import load_config
from src.services.history_import import import_historical_data
from src.storage.db import Database


def test_import_historical_data_is_idempotent_and_filters_recent_seasons(tmp_path):
    source_dir = tmp_path / "historical" / "nba"
    source_dir.mkdir(parents=True, exist_ok=True)

    games = pd.DataFrame(
        [
            {
                "game_id": 1,
                "season": 20212022,
                "game_date_utc": "2021-10-19",
                "start_time_utc": "2021-10-19T23:30:00Z",
                "home_team": "BOS",
                "away_team": "NY",
                "home_score": 98,
                "away_score": 88,
                "home_win": 1,
                "status_final": 1,
                "as_of_utc": "2021-10-20T05:00:00Z",
            },
            {
                "game_id": 2,
                "season": 20222023,
                "game_date_utc": "2022-10-19",
                "start_time_utc": "2022-10-19T23:30:00Z",
                "home_team": "LAL",
                "away_team": "GS",
                "home_score": 102,
                "away_score": 110,
                "home_win": 0,
                "status_final": 1,
                "as_of_utc": "2022-10-20T05:00:00Z",
            },
            {
                "game_id": 3,
                "season": 20232024,
                "game_date_utc": "2023-10-25",
                "start_time_utc": "2023-10-25T23:30:00Z",
                "home_team": "DAL",
                "away_team": "DEN",
                "home_score": 121,
                "away_score": 118,
                "home_win": 1,
                "status_final": 1,
                "as_of_utc": "2023-10-26T05:00:00Z",
            },
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "odds_event_id": "evt-1",
                "commence_time_utc": "2021-10-19T23:30:00Z",
                "home_team": "BOS",
                "away_team": "NY",
                "bookmaker_key": "book-a",
                "bookmaker_title": "Book A",
                "market_key": "h2h",
                "outcome_name": "BOS",
                "outcome_side": "home",
                "outcome_price": -135,
            },
            {
                "odds_event_id": "evt-1",
                "commence_time_utc": "2021-10-19T23:30:00Z",
                "home_team": "BOS",
                "away_team": "NY",
                "bookmaker_key": "book-a",
                "bookmaker_title": "Book A",
                "market_key": "h2h",
                "outcome_name": "NY",
                "outcome_side": "away",
                "outcome_price": 120,
            },
            {
                "odds_event_id": "evt-2",
                "commence_time_utc": "2022-10-19T23:30:00Z",
                "home_team": "LAL",
                "away_team": "GS",
                "bookmaker_key": "book-a",
                "bookmaker_title": "Book A",
                "market_key": "h2h",
                "outcome_name": "LAL",
                "outcome_side": "home",
                "outcome_price": 140,
            },
            {
                "odds_event_id": "evt-2",
                "commence_time_utc": "2022-10-19T23:30:00Z",
                "home_team": "LAL",
                "away_team": "GS",
                "bookmaker_key": "book-a",
                "bookmaker_title": "Book A",
                "market_key": "h2h",
                "outcome_name": "GS",
                "outcome_side": "away",
                "outcome_price": -155,
            },
            {
                "odds_event_id": "evt-3",
                "commence_time_utc": "2023-10-25T23:30:00Z",
                "home_team": "DAL",
                "away_team": "DEN",
                "bookmaker_key": "book-a",
                "bookmaker_title": "Book A",
                "market_key": "h2h",
                "outcome_name": "DAL",
                "outcome_side": "home",
                "outcome_price": 125,
            },
            {
                "odds_event_id": "evt-3",
                "commence_time_utc": "2023-10-25T23:30:00Z",
                "home_team": "DAL",
                "away_team": "DEN",
                "bookmaker_key": "book-a",
                "bookmaker_title": "Book A",
                "market_key": "h2h",
                "outcome_name": "DEN",
                "outcome_side": "away",
                "outcome_price": -145,
            },
        ]
    )

    games_path = source_dir / "games.csv"
    odds_path = source_dir / "odds.csv"
    manifest_path = source_dir / "manifest.json"
    games.to_csv(games_path, index=False)
    odds.to_csv(odds_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "league": "NBA",
                "games": [
                    {
                        "path": "games.csv",
                        "source": "nba_historical_games",
                        "extracted_at_utc": "2026-03-12T00:00:00Z",
                    }
                ],
                "odds_snapshots": [
                    {
                        "path": "odds.csv",
                        "source": "nba_historical_odds",
                        "as_of_utc": "2026-03-12T00:00:00Z",
                    }
                ],
            }
        )
    )

    cfg = load_config("configs/nba.yaml")
    cfg.paths.db_path = str(tmp_path / "processed" / "nba_forecast.db")
    cfg.paths.interim_dir = str(tmp_path / "interim" / "nba")
    cfg.paths.processed_dir = str(tmp_path / "processed" / "nba")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.research.source_dir = str(source_dir.parent)

    import_historical_data(cfg, history_seasons=2)
    import_historical_data(cfg, history_seasons=2)

    db = Database(cfg.paths.db_path)
    game_count = db.query("SELECT COUNT(*) AS count FROM games")[0]["count"]
    result_count = db.query("SELECT COUNT(*) AS count FROM results")[0]["count"]
    odds_snapshot_count = db.query("SELECT COUNT(*) AS count FROM odds_snapshots")[0]["count"]
    odds_line_count = db.query("SELECT COUNT(*) AS count FROM odds_market_lines")[0]["count"]
    seasons = db.query("SELECT DISTINCT season FROM games ORDER BY season DESC")

    assert game_count == 2
    assert result_count == 2
    assert odds_snapshot_count == 1
    assert odds_line_count == 4
    assert [row["season"] for row in seasons] == [20232024, 20222023]
