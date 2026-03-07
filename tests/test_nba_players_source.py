import pandas as pd

from src.data_sources.nba.players import fetch_players


class _FakeClient:
    def __init__(self, payload_by_source_key: dict[tuple[str, str], dict]):
        self.payload_by_source_key = payload_by_source_key

    def get_json(self, source: str, url: str, params: dict | None = None, key: str = "snapshot"):
        payload = self.payload_by_source_key[(source, key)]
        return payload, f"/tmp/{source}/{key}.json"

    def snapshot_id(self, source: str, metadata: dict):
        return f"{source}_snapshot"


def test_fetch_players_parses_boxscore_rows_and_current_status() -> None:
    client = _FakeClient(
        {
            (
                "nba_teams",
                "teams_20252026",
            ): {
                "sports": [
                    {
                        "leagues": [
                            {
                                "teams": [
                                    {"team": {"abbreviation": "TOR", "id": "28"}},
                                    {"team": {"abbreviation": "NYK", "id": "18"}},
                                ]
                            }
                        ]
                    }
                ]
            },
            (
                "nba_rosters",
                "TOR_20252026",
            ): {
                "athletes": [
                    {
                        "id": "123",
                        "fullName": "Scottie Barnes",
                        "position": {"abbreviation": "F"},
                        "status": {"type": "active"},
                        "injuries": [{"status": "Day-To-Day", "date": "2026-03-06T12:00Z"}],
                    }
                ]
            },
            (
                "nba_rosters",
                "NYK_20252026",
            ): {
                "athletes": [
                    {
                        "id": "456",
                        "fullName": "Jalen Brunson",
                        "position": {"abbreviation": "G"},
                        "status": {"type": "active"},
                        "injuries": [],
                    }
                ]
            },
            (
                "nba_players",
                "1234567",
            ): {
                "boxscore": {
                    "players": [
                        {
                            "team": {"abbreviation": "TOR"},
                            "statistics": [
                                {
                                    "labels": [
                                        "MIN",
                                        "PTS",
                                        "FG",
                                        "3PT",
                                        "FT",
                                        "REB",
                                        "AST",
                                        "TO",
                                        "STL",
                                        "BLK",
                                        "OREB",
                                        "DREB",
                                        "PF",
                                        "+/-",
                                    ],
                                    "athletes": [
                                        {
                                            "athlete": {
                                                "id": "123",
                                                "displayName": "Scottie Barnes",
                                                "position": {"abbreviation": "F"},
                                            },
                                            "starter": True,
                                            "didNotPlay": False,
                                            "stats": [
                                                "34:30",
                                                "24",
                                                "9-18",
                                                "3-7",
                                                "3-4",
                                                "10",
                                                "7",
                                                "3",
                                                "1",
                                                "1",
                                                "2",
                                                "8",
                                                "2",
                                                "+6",
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "team": {"abbreviation": "NYK"},
                            "statistics": [
                                {
                                    "labels": [
                                        "MIN",
                                        "PTS",
                                        "FG",
                                        "3PT",
                                        "FT",
                                        "REB",
                                        "AST",
                                        "TO",
                                        "STL",
                                        "BLK",
                                        "OREB",
                                        "DREB",
                                        "PF",
                                        "+/-",
                                    ],
                                    "athletes": [
                                        {
                                            "athlete": {
                                                "id": "456",
                                                "displayName": "Jalen Brunson",
                                                "position": {"abbreviation": "G"},
                                            },
                                            "starter": True,
                                            "didNotPlay": False,
                                            "stats": [
                                                "36",
                                                "27",
                                                "10-20",
                                                "3-8",
                                                "4-5",
                                                "3",
                                                "8",
                                                "2",
                                                "1",
                                                "0",
                                                "0",
                                                "3",
                                                "1",
                                                "-6",
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                    ]
                }
            },
        }
    )

    games_df = pd.DataFrame(
        {
            "game_id": [1234567],
            "status_final": [1],
            "start_time_utc": ["2026-03-01T00:00:00Z"],
            "game_date_utc": ["2026-03-01"],
            "home_team": ["TOR"],
            "away_team": ["NYK"],
        }
    )

    res = fetch_players(client, team_abbrevs=["TOR", "NYK"], season="20252026", games_df=games_df)
    assert len(res.dataframe) == 2

    tor = res.dataframe[res.dataframe["team"] == "TOR"].iloc[0]
    assert tor["player_id"] == "123"
    assert abs(float(tor["minutes"]) - 34.5) < 1e-9
    assert tor["current_injury_status"] == "Day-To-Day"
    assert tor["current_team"] == "TOR"
    assert tor["opponent"] == "NYK"
    assert tor["starter"] == 1

    nyk = res.dataframe[res.dataframe["team"] == "NYK"].iloc[0]
    assert nyk["player_id"] == "456"
    assert nyk["is_home"] == 0
    assert nyk["plus_minus_points"] == -6.0
