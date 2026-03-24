import json

import pandas as pd

from src.common.config import load_config
from src.data_sources import odds_api
from src.services.history_import import import_historical_data
from src.storage.db import Database


class _HistoricalStubClient:
    def __init__(self, payloads: dict[tuple[str, str], dict], raw_dir):
        self.payloads = payloads
        self.raw_dir = raw_dir

    def get_json_with_headers(
        self,
        source: str,
        url: str,
        params: dict | None = None,
        key: str = "snapshot",
        headers: dict[str, str] | None = None,
    ) -> tuple[dict, str, dict[str, str], bool]:
        del url, headers
        params = params or {}
        if "dates" in params:
            payload = self.payloads[("scoreboard", str(params["dates"]))]
        elif "event" in params:
            payload = self.payloads[("summary", str(params["event"]))]
        else:
            raise AssertionError(f"Unexpected params: {params}")
        raw_path = self.save_raw(source, payload, key=key)
        return payload, raw_path, {}, False

    def save_raw(self, source: str, payload: dict, key: str = "snapshot") -> str:
        path = self.raw_dir / f"{source}_{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        return str(path)

    def snapshot_id(self, source: str, metadata: dict) -> str:
        return f"{source}_{metadata.get('league')}_{metadata.get('n_rows')}"


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


def test_import_historical_data_accepts_generated_historical_bundle(tmp_path):
    source_dir = tmp_path / "historical" / "nba"
    raw_dir = tmp_path / "raw-cache"
    games = pd.DataFrame(
        [
            {
                "game_id": 401810863,
                "season": 20252026,
                "game_date_utc": "2026-03-19",
                "start_time_utc": "2026-03-19T23:00:00Z",
                "home_team": "CHA",
                "away_team": "ORL",
                "home_score": 101,
                "away_score": 112,
                "home_win": 0,
                "status_final": 1,
                "as_of_utc": "2026-03-20T05:00:00Z",
            }
        ]
    )
    games_path = tmp_path / "research_games.parquet"
    games.to_parquet(games_path, index=False)

    scoreboard_event = {
        "id": "401810863",
        "date": "2026-03-19T23:00Z",
        "competitions": [
            {
                "id": "401810863",
                "date": "2026-03-19T23:00Z",
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {
                            "displayName": "Charlotte Hornets",
                            "shortDisplayName": "Hornets",
                            "abbreviation": "CHA",
                            "location": "Charlotte",
                            "name": "Hornets",
                        },
                    },
                    {
                        "homeAway": "away",
                        "team": {
                            "displayName": "Orlando Magic",
                            "shortDisplayName": "Magic",
                            "abbreviation": "ORL",
                            "location": "Orlando",
                            "name": "Magic",
                        },
                    },
                ],
            }
        ],
    }
    summary_payload = {
        "header": {"id": "401810863", "competitions": scoreboard_event["competitions"]},
        "pickcenter": [
            {
                "provider": {"id": "100", "name": "Draft Kings"},
                "homeTeamOdds": {"moneyLine": 455, "favorite": False},
                "awayTeamOdds": {"moneyLine": -625, "favorite": True},
                "moneyline": {
                    "home": {"close": {"odds": "+455"}},
                    "away": {"close": {"odds": "-625"}},
                },
            }
        ],
    }
    client = _HistoricalStubClient(
        {
            ("scoreboard", "20260319"): {"events": [scoreboard_event]},
            ("summary", "401810863"): summary_payload,
        },
        raw_dir=raw_dir,
    )

    bundle = odds_api.write_historical_odds_bundle(
        client,
        league="NBA",
        sport_key="basketball_nba",
        source="nba_historical_odds",
        output_dir=source_dir,
        start_date="2026-03-19",
        end_date="2026-03-19",
        games_path=games_path,
    )

    cfg = load_config("configs/nba.yaml")
    cfg.paths.db_path = str(tmp_path / "processed" / "nba_forecast.db")
    cfg.paths.interim_dir = str(tmp_path / "interim" / "nba")
    cfg.paths.processed_dir = str(tmp_path / "processed" / "nba")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.research.source_dir = str(source_dir.parent)

    import_historical_data(cfg, history_seasons=1, source_manifest=bundle["manifest_path"])

    db = Database(cfg.paths.db_path)
    game_count = db.query("SELECT COUNT(*) AS count FROM games")[0]["count"]
    odds_snapshot_count = db.query("SELECT COUNT(*) AS count FROM odds_snapshots")[0]["count"]
    odds_line_count = db.query("SELECT COUNT(*) AS count FROM odds_market_lines")[0]["count"]
    rows = db.query(
        """
        SELECT game_id, home_team, away_team, market_key, outcome_side, outcome_price
        FROM odds_market_lines
        ORDER BY market_key, outcome_side
        """
    )

    assert game_count == 1
    assert odds_snapshot_count == 1
    assert odds_line_count == 2
    assert rows == [
        {"game_id": 401810863, "home_team": "CHA", "away_team": "ORL", "market_key": "h2h", "outcome_side": "away", "outcome_price": -625.0},
        {"game_id": 401810863, "home_team": "CHA", "away_team": "ORL", "market_key": "h2h", "outcome_side": "home", "outcome_price": 455.0},
    ]
