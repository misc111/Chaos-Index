from src.data_sources.nhl.players import fetch_players


class _FakeClient:
    def __init__(self, payload_by_team: dict[str, dict]):
        self.payload_by_team = payload_by_team

    def get_json(self, source: str, url: str, params: dict | None = None, key: str = "snapshot"):
        team = key.split("_")[0]
        return self.payload_by_team[team], f"/tmp/{source}/{key}.json"

    def snapshot_id(self, source: str, metadata: dict):
        return f"{source}_snapshot"


def test_fetch_players_maps_season_level_skater_stats():
    client = _FakeClient(
        {
            "TOR": {
                "skaters": [
                    {
                        "playerId": 1,
                        "positionCode": "C",
                        "gamesPlayed": 50,
                        "goals": 30,
                        "assists": 40,
                        "points": 70,
                        "shots": 200,
                        "shootingPctg": 15.0,
                        "faceoffWinPctg": 54.2,
                        "penaltyMinutes": 20,
                        "powerPlayGoals": 10,
                        "shorthandedGoals": 1,
                        "gameWinningGoals": 5,
                        "overtimeGoals": 2,
                        "avgShiftsPerGame": 24.1,
                        "avgTimeOnIcePerGame": "18:30",
                        "plusMinus": 12,
                    }
                ]
            }
        }
    )

    res = fetch_players(client, team_abbrevs=["TOR"], season="20252026")
    row = res.dataframe.iloc[0]
    assert row["shots"] == 200
    assert row["avg_time_on_ice_per_game"] == "18:30"
    assert row["toi_per_game"] == "18:30"
    assert abs(float(row["toi_per_game_minutes"]) - 18.5) < 1e-9


def test_fetch_players_supports_legacy_avg_toi_key():
    client = _FakeClient(
        {
            "BOS": {
                "skaters": [
                    {
                        "playerId": 2,
                        "positionCode": "D",
                        "gamesPlayed": 45,
                        "goals": 8,
                        "assists": 22,
                        "points": 30,
                        "shots": 110,
                        "avgToi": "21:15",
                        "plusMinus": 5,
                    }
                ]
            }
        }
    )

    res = fetch_players(client, team_abbrevs=["BOS"], season="20252026")
    row = res.dataframe.iloc[0]
    assert row["avg_time_on_ice_per_game"] == "21:15"
    assert row["toi_per_game"] == "21:15"
    assert abs(float(row["toi_per_game_minutes"]) - 21.25) < 1e-9
