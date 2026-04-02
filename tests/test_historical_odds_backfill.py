import json
from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.services.historical_odds_backfill import backfill_historical_odds_cache
from src.services.history_import import import_historical_data
from src.services.ingest import upsert_games
from src.storage.db import Database


class _BackfillStubClient:
    def __init__(self, payloads: dict[tuple[str, str], dict], raw_dir: Path):
        self.payloads = payloads
        self.raw_dir = raw_dir
        self.calls: list[tuple[str, str]] = []

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
            lookup = ("scoreboard", str(params["dates"]))
        elif "event" in params:
            lookup = ("summary", str(params["event"]))
        else:
            raise AssertionError(f"Unexpected params: {params}")
        self.calls.append(lookup)
        payload = self.payloads[lookup]
        raw_path = self.save_raw(source, payload, key=key)
        return payload, raw_path, {}, False

    def get_json(self, source: str, url: str, params: dict | None = None, key: str = "snapshot") -> tuple[dict, str]:
        payload, raw_path, _, _ = self.get_json_with_headers(source, url, params=params, key=key)
        return payload, raw_path

    def save_raw(self, source: str, payload: dict, key: str = "snapshot") -> str:
        path = self.raw_dir / f"{source}_{key}_{len(self.calls)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        return str(path)

    def snapshot_id(self, source: str, metadata: dict) -> str:
        return f"{source}_{metadata.get('league')}_{metadata.get('n_rows')}_{len(self.calls)}"


def _scoreboard_event(game_id: int, commence_time_utc: str, home_abbrev: str, away_abbrev: str) -> dict:
    return {
        "id": str(game_id),
        "date": commence_time_utc,
        "competitions": [
            {
                "id": str(game_id),
                "date": commence_time_utc,
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {
                            "displayName": home_abbrev,
                            "shortDisplayName": home_abbrev,
                            "abbreviation": home_abbrev,
                            "location": home_abbrev,
                            "name": home_abbrev,
                        },
                    },
                    {
                        "homeAway": "away",
                        "team": {
                            "displayName": away_abbrev,
                            "shortDisplayName": away_abbrev,
                            "abbreviation": away_abbrev,
                            "location": away_abbrev,
                            "name": away_abbrev,
                        },
                    },
                ],
            }
        ],
    }


def _summary_payload(
    game_id: int,
    commence_time_utc: str,
    home_abbrev: str,
    away_abbrev: str,
    home_price: int,
    away_price: int,
    bookmaker: str = "Draft Kings",
) -> dict:
    return {
        "header": {
            "id": str(game_id),
            "competitions": [
                {
                    "id": str(game_id),
                    "date": commence_time_utc,
                    "competitors": [
                        {
                            "homeAway": "home",
                            "team": {
                                "displayName": home_abbrev,
                                "shortDisplayName": home_abbrev,
                                "abbreviation": home_abbrev,
                                "location": home_abbrev,
                                "name": home_abbrev,
                            },
                        },
                        {
                            "homeAway": "away",
                            "team": {
                                "displayName": away_abbrev,
                                "shortDisplayName": away_abbrev,
                                "abbreviation": away_abbrev,
                                "location": away_abbrev,
                                "name": away_abbrev,
                            },
                        },
                    ],
                }
            ],
        },
        "pickcenter": [
            {
                "provider": {"id": "100", "name": bookmaker},
                "homeTeamOdds": {"moneyLine": home_price, "favorite": home_price < away_price},
                "awayTeamOdds": {"moneyLine": away_price, "favorite": away_price < home_price},
                "moneyline": {
                    "home": {"close": {"odds": f"{home_price:+d}"}},
                    "away": {"close": {"odds": f"{away_price:+d}"}},
                },
            }
        ],
    }


def _temp_cfg(tmp_path: Path):
    cfg = load_config("configs/nba.yaml")
    cfg.paths.db_path = str(tmp_path / "processed" / "nba_forecast.db")
    cfg.paths.interim_dir = str(tmp_path / "interim" / "nba")
    cfg.paths.processed_dir = str(tmp_path / "processed" / "nba")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.raw_dir = str(tmp_path / "raw")
    cfg.research.source_dir = str(tmp_path / "historical")
    return cfg


def _seed_games(db: Database) -> None:
    games = pd.DataFrame(
        [
            {
                "game_id": 401810863,
                "season": 20252026,
                "game_date_utc": "2026-03-19",
                "start_time_utc": "2026-03-19T23:00:00Z",
                "game_state": "final",
                "home_team": "CHA",
                "away_team": "ORL",
                "home_team_id": None,
                "away_team_id": None,
                "venue": None,
                "is_neutral_site": 0,
                "home_score": 101,
                "away_score": 112,
                "went_ot": 0,
                "went_so": 0,
                "home_win": 0,
                "status_final": 1,
                "as_of_utc": "2026-03-20T05:00:00Z",
            },
            {
                "game_id": 401810864,
                "season": 20252026,
                "game_date_utc": "2026-03-20",
                "start_time_utc": "2026-03-20T23:30:00Z",
                "game_state": "final",
                "home_team": "BOS",
                "away_team": "NY",
                "home_team_id": None,
                "away_team_id": None,
                "venue": None,
                "is_neutral_site": 0,
                "home_score": 110,
                "away_score": 98,
                "went_ot": 0,
                "went_so": 0,
                "home_win": 1,
                "status_final": 1,
                "as_of_utc": "2026-03-21T05:00:00Z",
            },
        ]
    )
    upsert_games(db, games)


def test_backfill_historical_odds_cache_builds_manifest_compatible_with_import_history(tmp_path) -> None:
    cfg = _temp_cfg(tmp_path)
    db = Database(cfg.paths.db_path)
    db.init_schema()
    _seed_games(db)

    payloads = {
        ("scoreboard", "20260319"): {"events": [_scoreboard_event(401810863, "2026-03-19T23:00Z", "CHA", "ORL")]},
        ("scoreboard", "20260320"): {"events": [_scoreboard_event(401810864, "2026-03-20T23:30Z", "BOS", "NY")]},
        ("summary", "401810863"): _summary_payload(401810863, "2026-03-19T23:00Z", "CHA", "ORL", 455, -625),
        ("summary", "401810864"): _summary_payload(401810864, "2026-03-20T23:30Z", "BOS", "NY", -135, 120),
    }
    client = _BackfillStubClient(payloads, raw_dir=tmp_path / "raw-cache")

    result = backfill_historical_odds_cache(
        cfg,
        start_date="2026-03-19",
        end_date="2026-03-20",
        chunk_days=1,
        history_seasons=1,
        client=client,
        teams_df=pd.DataFrame(),
    )

    manifest = json.loads(Path(result.manifest_path).read_text())

    assert result.chunk_count == 2
    assert result.fetched_chunks == 2
    assert result.skipped_chunks == 0
    assert manifest["league"] == "NBA"
    assert len(manifest["games"]) == 2
    assert len(manifest["odds_snapshots"]) == 2
    assert all(str(entry["path"]).startswith("backfill_2026-03-") for entry in manifest["games"])
    assert all(str(entry["path"]).startswith("backfill_2026-03-") for entry in manifest["odds_snapshots"])

    import_historical_data(cfg, history_seasons=1, source_manifest=str(result.manifest_path))

    imported_db = Database(cfg.paths.db_path)
    assert imported_db.query("SELECT COUNT(*) AS count FROM odds_snapshots")[0]["count"] == 2
    assert imported_db.query("SELECT COUNT(*) AS count FROM odds_market_lines")[0]["count"] == 4
    assert imported_db.query("SELECT COUNT(*) AS count FROM games")[0]["count"] == 2


def test_backfill_historical_odds_cache_skips_existing_chunk_manifests(tmp_path) -> None:
    cfg = _temp_cfg(tmp_path)
    db = Database(cfg.paths.db_path)
    db.init_schema()
    _seed_games(db)

    payloads = {
        ("scoreboard", "20260319"): {"events": [_scoreboard_event(401810863, "2026-03-19T23:00Z", "CHA", "ORL")]},
        ("scoreboard", "20260320"): {"events": [_scoreboard_event(401810864, "2026-03-20T23:30Z", "BOS", "NY")]},
        ("summary", "401810863"): _summary_payload(401810863, "2026-03-19T23:00Z", "CHA", "ORL", 455, -625),
        ("summary", "401810864"): _summary_payload(401810864, "2026-03-20T23:30Z", "BOS", "NY", -135, 120),
    }
    first_client = _BackfillStubClient(payloads, raw_dir=tmp_path / "raw-cache")
    first = backfill_historical_odds_cache(
        cfg,
        start_date="2026-03-19",
        end_date="2026-03-20",
        chunk_days=1,
        history_seasons=1,
        client=first_client,
        teams_df=pd.DataFrame(),
    )
    assert first.fetched_chunks == 2

    second_client = _BackfillStubClient({}, raw_dir=tmp_path / "raw-cache-2")
    second = backfill_historical_odds_cache(
        cfg,
        start_date="2026-03-19",
        end_date="2026-03-20",
        chunk_days=1,
        history_seasons=1,
        client=second_client,
        teams_df=pd.DataFrame(),
    )

    assert second.chunk_count == 2
    assert second.fetched_chunks == 0
    assert second.skipped_chunks == 2
    assert second_client.calls == []
