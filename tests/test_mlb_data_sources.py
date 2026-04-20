from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_sources.mlb.games import fetch_games
from src.data_sources.mlb.injuries import fetch_injuries_report
from src.data_sources.mlb.players import fetch_players
from src.data_sources.mlb.results import build_results_from_games
from src.data_sources.mlb.starting_pitchers import fetch_starter_context
from src.data_sources.mlb.team_stats import fetch_team_game_stats
from src.data_sources.mlb.teams import fetch_teams
from src.data_sources.mlb.weather import fetch_weather_context_optional
from src.data_sources import odds_api


class StubClient:
    def __init__(self, payloads: dict[tuple[str, str], dict], raw_dir: Path):
        self.payloads = payloads
        self.raw_dir = raw_dir

    def get_json(self, source: str, url: str, params: dict | None = None, key: str = "snapshot") -> tuple[dict, str]:
        del source, url
        params = params or {}
        payload = self.payloads[(key, json.dumps(params, sort_keys=True))]
        raw_path = self.save_raw(key, payload)
        return payload, raw_path

    def get_json_with_headers(
        self,
        source: str,
        url: str,
        params: dict | None = None,
        key: str = "snapshot",
        headers: dict[str, str] | None = None,
    ) -> tuple[dict, str, dict[str, str], bool]:
        del source, url, headers
        params = params or {}
        payload = self.payloads[(key, json.dumps(params, sort_keys=True))]
        raw_path = self.save_raw(key, payload)
        return payload, raw_path, {}, False

    def save_raw(self, key: str, payload: dict) -> str:
        path = self.raw_dir / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        return str(path)

    def snapshot_id(self, source: str, metadata: dict) -> str:
        return f"{source}_{metadata.get('n_games') or metadata.get('n_teams') or metadata.get('n_rows')}"


def _scoreboard_payload() -> dict:
    return {
        "events": [
            {
                "id": "401815018",
                "date": "2026-04-20T15:10Z",
                "season": {"year": 2026, "type": 2},
                "competitions": [
                    {
                        "id": "401815018",
                        "date": "2026-04-20T15:10Z",
                        "neutralSite": False,
                        "venue": {"fullName": "Fenway Park"},
                        "status": {
                            "type": {
                                "completed": False,
                                "description": "Scheduled",
                                "name": "STATUS_SCHEDULED",
                                "shortDetail": "Today, 11:10 AM CDT",
                            },
                            "period": 0,
                        },
                        "competitors": [
                            {
                                "homeAway": "home",
                                "team": {"id": "2", "abbreviation": "BOS", "displayName": "Boston Red Sox"},
                                "score": "0",
                            },
                            {
                                "homeAway": "away",
                                "team": {"id": "6", "abbreviation": "DET", "displayName": "Detroit Tigers"},
                                "score": "0",
                            },
                        ],
                    }
                ],
            },
            {
                "id": "401815019",
                "date": "2026-04-19T18:35Z",
                "season": {"year": 2026, "type": 2},
                "competitions": [
                    {
                        "id": "401815019",
                        "date": "2026-04-19T18:35Z",
                        "neutralSite": False,
                        "venue": {"fullName": "Wrigley Field"},
                        "status": {
                            "type": {
                                "completed": True,
                                "description": "Final",
                                "name": "STATUS_FINAL",
                                "shortDetail": "Final",
                            },
                            "period": 9,
                        },
                        "competitors": [
                            {
                                "homeAway": "home",
                                "team": {"id": "16", "abbreviation": "CHC", "displayName": "Chicago Cubs"},
                                "score": "5",
                            },
                            {
                                "homeAway": "away",
                                "team": {"id": "15", "abbreviation": "LAD", "displayName": "Los Angeles Dodgers"},
                                "score": "3",
                            },
                        ],
                    }
                ],
            },
        ]
    }


def _teams_payload() -> dict:
    return {
        "sports": [
            {
                "leagues": [
                    {
                        "teams": [
                            {"team": {"id": "2", "abbreviation": "BOS", "displayName": "Boston Red Sox", "location": "Boston", "name": "Red Sox"}},
                            {"team": {"id": "6", "abbreviation": "DET", "displayName": "Detroit Tigers", "location": "Detroit", "name": "Tigers"}},
                        ]
                    }
                ]
            }
        ]
    }


def _sample_scoreboard_event() -> dict:
    return _scoreboard_payload()["events"][0]


def _sample_summary_payload() -> dict:
    return {
        "header": {"id": "401815018", "competitions": _sample_scoreboard_event()["competitions"]},
        "pickcenter": [
            {
                "provider": {"id": "100", "name": "DraftKings"},
                "homeTeamOdds": {"moneyLine": -136, "spreadOdds": -102.0, "favorite": True},
                "awayTeamOdds": {"moneyLine": 113, "spreadOdds": -118.0, "favorite": False},
                "spread": -1.5,
                "overUnder": 8.0,
                "overOdds": -108.0,
                "underOdds": -112.0,
            }
        ],
    }


def _boxscore_summary_payload() -> dict:
    return {
        "header": {
            "id": "401815019",
            "competitions": _scoreboard_payload()["events"][1]["competitions"],
        },
        "boxscore": {
            "teams": [
                {
                    "homeAway": "home",
                    "team": {"id": "16", "abbreviation": "CHC", "displayName": "Chicago Cubs"},
                    "statistics": [
                        {
                            "name": "batting",
                            "displayName": "Batting",
                            "stats": [
                                {"name": "hits", "value": 9.0},
                                {"name": "walks", "value": 3.0},
                                {"name": "strikeouts", "value": 8.0},
                                {"name": "homeRuns", "value": 2.0},
                                {"name": "totalBases", "value": 17.0},
                                {"name": "stolenBases", "value": 1.0},
                            ],
                        },
                        {
                            "name": "pitching",
                            "displayName": "Pitching",
                            "stats": [
                                {"name": "walks", "value": 2.0},
                                {"name": "strikeouts", "value": 9.0},
                                {"name": "homeRuns", "value": 1.0},
                                {"name": "hits", "value": 6.0},
                                {"name": "runs", "value": 3.0},
                            ],
                        },
                    ],
                },
                {
                    "homeAway": "away",
                    "team": {"id": "15", "abbreviation": "LAD", "displayName": "Los Angeles Dodgers"},
                    "statistics": [
                        {
                            "name": "batting",
                            "displayName": "Batting",
                            "stats": [
                                {"name": "hits", "value": 6.0},
                                {"name": "walks", "value": 2.0},
                                {"name": "strikeouts", "value": 10.0},
                                {"name": "homeRuns", "value": 1.0},
                                {"name": "totalBases", "value": 10.0},
                                {"name": "stolenBases", "value": 0.0},
                            ],
                        },
                        {
                            "name": "pitching",
                            "displayName": "Pitching",
                            "stats": [
                                {"name": "walks", "value": 3.0},
                                {"name": "strikeouts", "value": 8.0},
                                {"name": "homeRuns", "value": 2.0},
                                {"name": "hits", "value": 9.0},
                                {"name": "runs", "value": 5.0},
                            ],
                        },
                    ],
                },
            ],
            "players": [
                {
                    "team": {"id": "16", "abbreviation": "CHC"},
                    "statistics": [
                        {
                            "type": "batting",
                            "athletes": [
                                {
                                    "athlete": {"id": "1001", "displayName": "Ian Happ"},
                                    "starter": True,
                                    "batOrder": 1,
                                    "position": {"abbreviation": "LF"},
                                    "stats": ["1-4", "4", "1", "1"],
                                },
                                {
                                    "athlete": {"id": "1002", "displayName": "Dansby Swanson"},
                                    "starter": True,
                                    "batOrder": 2,
                                    "position": {"abbreviation": "SS"},
                                    "stats": ["2-4", "4", "1", "2"],
                                },
                            ],
                        },
                        {
                            "type": "pitching",
                            "athletes": [
                                {
                                    "athlete": {"id": "2001", "displayName": "Shota Imanaga"},
                                    "starter": True,
                                    "position": {"abbreviation": "SP", "displayName": "Starting Pitcher"},
                                    "stats": ["6.0", "5", "2", "2", "1", "8", "1", "92-61", "3.21", "92"],
                                },
                                {
                                    "athlete": {"id": "2002", "displayName": "Adbert Alzolay"},
                                    "starter": False,
                                    "position": {"abbreviation": "RP", "displayName": "Relief Pitcher"},
                                    "stats": ["1.0", "1", "0", "0", "0", "1", "0", "12-9", "2.11", "12"],
                                },
                            ],
                        },
                    ],
                },
                {
                    "team": {"id": "15", "abbreviation": "LAD"},
                    "statistics": [
                        {
                            "type": "batting",
                            "athletes": [
                                {
                                    "athlete": {"id": "3001", "displayName": "Mookie Betts"},
                                    "starter": True,
                                    "batOrder": 1,
                                    "position": {"abbreviation": "RF"},
                                    "stats": ["1-5", "5", "1", "1"],
                                }
                            ],
                        },
                        {
                            "type": "pitching",
                            "athletes": [
                                {
                                    "athlete": {"id": "4001", "displayName": "Yoshinobu Yamamoto"},
                                    "starter": True,
                                    "position": {"abbreviation": "SP", "displayName": "Starting Pitcher"},
                                    "stats": ["5.1", "8", "4", "4", "2", "6", "2", "97-63", "2.98", "97"],
                                }
                            ],
                        },
                    ],
                },
            ],
        },
        "rosters": [
            {
                "homeAway": "home",
                "team": {"id": "16", "abbreviation": "CHC"},
                "roster": [
                    {
                        "athlete": {"id": "1001", "displayName": "Ian Happ"},
                        "starter": True,
                        "batOrder": 1,
                        "position": {"abbreviation": "LF"},
                        "active": True,
                    },
                    {
                        "athlete": {"id": "1002", "displayName": "Dansby Swanson"},
                        "starter": True,
                        "batOrder": 2,
                        "position": {"abbreviation": "SS"},
                        "active": True,
                    },
                ],
            },
            {
                "homeAway": "away",
                "team": {"id": "15", "abbreviation": "LAD"},
                "roster": [
                    {
                        "athlete": {"id": "3001", "displayName": "Mookie Betts"},
                        "starter": True,
                        "batOrder": 1,
                        "position": {"abbreviation": "RF"},
                        "active": True,
                    }
                ],
            },
        ],
        "injuries": [
            {
                "team": {"id": "16", "abbreviation": "CHC"},
                "injuries": [
                    {
                        "athlete": {"id": "9001", "displayName": "Injured Outfielder"},
                        "status": {"name": "Day-To-Day"},
                        "details": {"type": {"abbreviation": "OF"}},
                    },
                    {
                        "athlete": {"id": "9002", "displayName": "Injured Pitcher"},
                        "status": {"name": "15-Day IL"},
                        "details": {"type": {"abbreviation": "P"}},
                    },
                ],
            }
        ],
    }


def test_fetch_mlb_games_parses_schedule_and_results(tmp_path) -> None:
    client = StubClient(
        {
            ("20260419", json.dumps({"dates": "20260419"}, sort_keys=True)): _scoreboard_payload(),
            ("20260420", json.dumps({"dates": "20260420"}, sort_keys=True)): _scoreboard_payload(),
        },
        raw_dir=tmp_path,
    )

    result = fetch_games(
        client,
        start_date=pd.Timestamp("2026-04-19T00:00:00Z").to_pydatetime(),
        end_date=pd.Timestamp("2026-04-20T23:59:59Z").to_pydatetime(),
    )

    assert len(result.dataframe) == 2
    assert set(result.dataframe["home_team"]) == {"BOS", "CHC"}
    assert set(result.dataframe["season"]) == {2026}
    final_row = result.dataframe[result.dataframe["game_id"] == 401815019].iloc[0]
    assert final_row["status_final"] == 1
    assert final_row["home_win"] == 1


def test_fetch_mlb_teams_uses_espn_teams_payload(tmp_path) -> None:
    client = StubClient(
        {
            ("2026-04-20", json.dumps({}, sort_keys=True)): _teams_payload(),
        },
        raw_dir=tmp_path,
    )

    result = fetch_teams(client, as_of_date="2026-04-20")

    assert len(result.dataframe) == 2
    assert result.dataframe["team_abbrev"].tolist() == ["BOS", "DET"]
    assert result.dataframe["team_name"].tolist() == ["Boston Red Sox", "Detroit Tigers"]


def test_odds_api_supports_mlb_pickcenter_rows() -> None:
    teams_df = pd.DataFrame(
        [
            {"team_abbrev": "BOS", "team_name": "Boston Red Sox"},
            {"team_abbrev": "DET", "team_name": "Detroit Tigers"},
        ]
    )

    rows = odds_api._flatten_pickcenter_summary(
        summary_payload=_sample_summary_payload(),
        scoreboard_event=_sample_scoreboard_event(),
        league="MLB",
        sport_key="baseball_mlb",
        team_name_map=odds_api._build_team_name_map(teams_df),
        alias_map=odds_api._team_aliases_for_league("MLB"),
        as_of_utc="2026-04-20T13:00:00+00:00",
    )

    assert len(rows) == 6
    by_market_side = {(row["market_key"], row["outcome_side"]): row for row in rows}
    assert by_market_side[("h2h", "home")]["home_team"] == "BOS"
    assert by_market_side[("h2h", "away")]["away_team"] == "DET"
    assert by_market_side[("spreads", "home")]["outcome_point"] == -1.5
    assert by_market_side[("spreads", "away")]["outcome_point"] == 1.5
    assert by_market_side[("totals", "over")]["outcome_point"] == 8.0


def test_build_results_from_games_keeps_only_final_games() -> None:
    games_df = pd.DataFrame(
        [
            {
                "game_id": 401815018,
                "season": 2026,
                "game_date_utc": "2026-04-20",
                "start_time_utc": "2026-04-20T15:10:00Z",
                "home_team": "BOS",
                "away_team": "DET",
                "home_score": 0,
                "away_score": 0,
                "home_win": None,
                "status_final": 0,
                "as_of_utc": "2026-04-20T12:00:00Z",
            },
            {
                "game_id": 401815019,
                "season": 2026,
                "game_date_utc": "2026-04-19",
                "start_time_utc": "2026-04-19T18:35:00Z",
                "home_team": "CHC",
                "away_team": "LAD",
                "home_score": 5,
                "away_score": 3,
                "home_win": 1,
                "status_final": 1,
                "as_of_utc": "2026-04-19T22:00:00Z",
            },
        ]
    )

    results_df = build_results_from_games(games_df)

    assert list(results_df["game_id"]) == [401815019]
    assert list(results_df["final_utc"]) == ["2026-04-19T18:35:00Z"]
    assert list(results_df["ingested_at_utc"]) == ["2026-04-19T22:00:00Z"]


def test_mlb_module_names_are_baseball_faithful() -> None:
    mlb_dir = Path(__file__).resolve().parents[1] / "src" / "data_sources" / "mlb"
    assert not (mlb_dir / "goalies.py").exists()
    assert not (mlb_dir / "xg.py").exists()


def test_fetch_mlb_team_game_stats_parses_summary_boxscore_totals(tmp_path) -> None:
    client = StubClient(
        {
            ("401815019", json.dumps({"event": "401815019"}, sort_keys=True)): _boxscore_summary_payload(),
        },
        raw_dir=tmp_path,
    )

    result = fetch_team_game_stats(client, game_ids=[401815019], max_games=25)

    assert result.source == "mlb_team_game_stats"
    assert set(result.dataframe["team"]) == {"CHC", "LAD"}
    home = result.dataframe[result.dataframe["team"] == "CHC"].iloc[0]
    assert home["hits"] == 9.0
    assert home["walks"] == 3.0
    assert home["strikeouts"] == 8.0
    assert home["total_bases"] == 17.0


def test_fetch_mlb_starting_pitchers_parses_actual_starters_from_boxscore(tmp_path) -> None:
    client = StubClient(
        {
            ("401815019", json.dumps({"event": "401815019"}, sort_keys=True)): _boxscore_summary_payload(),
        },
        raw_dir=tmp_path,
    )

    result = fetch_starter_context(client, game_ids=[401815019], max_games=25)

    assert result.source == "mlb_starting_pitchers"
    assert set(result.dataframe["team"]) == {"CHC", "LAD"}
    home = result.dataframe[result.dataframe["team"] == "CHC"].iloc[0]
    away = result.dataframe[result.dataframe["team"] == "LAD"].iloc[0]
    assert home["starter_pitcher_id"] == "2001"
    assert home["starter_status"] == "actual"
    assert home["innings_pitched"] == 6.0
    assert away["starter_pitcher_id"] == "4001"
    assert away["innings_pitched"] == 5.333333333333333


def test_fetch_mlb_players_parses_lineup_cards_from_rosters(tmp_path) -> None:
    client = StubClient(
        {
            ("401815019", json.dumps({"event": "401815019"}, sort_keys=True)): _boxscore_summary_payload(),
        },
        raw_dir=tmp_path,
    )
    games_df = pd.DataFrame([{"game_id": 401815019}])

    result = fetch_players(client, team_abbrevs=["CHC", "LAD"], season="2026", games_df=games_df)

    assert result.source == "mlb_lineups"
    assert set(result.dataframe["team"]) == {"CHC", "LAD"}
    assert set(result.dataframe["batting_order_slot"]) == {1, 2}
    assert result.dataframe["lineup_confirmed"].eq(1).all()


def test_fetch_mlb_injuries_report_aggregates_pitcher_and_position_player_counts(tmp_path) -> None:
    client = StubClient(
        {
            ("401815019", json.dumps({"event": "401815019"}, sort_keys=True)): _boxscore_summary_payload(),
        },
        raw_dir=tmp_path,
    )
    games_df = pd.DataFrame([{"game_id": 401815019}])

    result = fetch_injuries_report(client, teams=["CHC", "LAD"], games_df=games_df)

    assert result.source == "mlb_injuries"
    assert len(result.dataframe) == 1
    row = result.dataframe.iloc[0]
    assert row["team"] == "CHC"
    assert row["position_player_out_count"] == 1
    assert row["pitcher_out_count"] == 1


def test_fetch_mlb_weather_context_stays_honest_when_provider_has_no_weather(tmp_path) -> None:
    client = StubClient({}, raw_dir=tmp_path)
    games_df = pd.DataFrame([{"game_id": 401815019, "start_time_utc": "2026-04-19T18:35:00Z", "venue": "Wrigley Field"}])

    result = fetch_weather_context_optional(client, games_df=games_df)

    assert result.source == "mlb_weather"
    assert result.dataframe.empty
    assert result.metadata["status"] == "scaffold"
