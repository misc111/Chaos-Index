from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_sources import odds_api


class StubClient:
    def __init__(self, payloads: dict[tuple[str, str], dict], raw_dir: Path):
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
        path.write_text(json.dumps(payload, indent=2))
        return str(path)

    def snapshot_id(self, source: str, metadata: dict) -> str:
        return f"{source}_{metadata.get('league')}_{metadata.get('n_rows')}"


def _sample_scoreboard_event() -> dict:
    return {
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


def _sample_summary_payload() -> dict:
    return {
        "header": {
            "id": "401810863",
            "competitions": _sample_scoreboard_event()["competitions"],
        },
        "pickcenter": [
            {
                "provider": {"id": "100", "name": "Draft Kings"},
                "homeTeamOdds": {"moneyLine": 455, "spreadOdds": -108.0, "favorite": False},
                "awayTeamOdds": {"moneyLine": -625, "spreadOdds": -112.0, "favorite": True},
                "moneyline": {
                    "home": {"close": {"odds": "+455"}},
                    "away": {"close": {"odds": "-625"}},
                },
                "pointSpread": {
                    "home": {"close": {"line": "+11.5", "odds": "-108"}},
                    "away": {"close": {"line": "-11.5", "odds": "-112"}},
                },
                "total": {
                    "over": {"close": {"line": "o216.5", "odds": "-110"}},
                    "under": {"close": {"line": "u216.5", "odds": "-110"}},
                },
                "overUnder": 216.5,
                "overOdds": -110.0,
                "underOdds": -110.0,
            }
        ],
    }


def test_flatten_pickcenter_summary_generates_standard_market_rows() -> None:
    teams_df = pd.DataFrame(
        [
            {"team_abbrev": "CHA", "team_name": "Charlotte"},
            {"team_abbrev": "ORL", "team_name": "Orlando"},
        ]
    )

    rows = odds_api._flatten_pickcenter_summary(
        summary_payload=_sample_summary_payload(),
        scoreboard_event=_sample_scoreboard_event(),
        league="NBA",
        sport_key="basketball_nba",
        team_name_map=odds_api._build_team_name_map(teams_df),
        alias_map=odds_api._team_aliases_for_league("NBA"),
        as_of_utc="2026-03-19T13:00:00+00:00",
    )

    assert len(rows) == 6
    by_market_side = {(row["market_key"], row["outcome_side"]): row for row in rows}

    assert by_market_side[("h2h", "home")]["outcome_price"] == 455
    assert by_market_side[("h2h", "home")]["home_team"] == "CHA"
    assert by_market_side[("h2h", "away")]["outcome_price"] == -625
    assert by_market_side[("spreads", "home")]["outcome_point"] == 11.5
    assert by_market_side[("spreads", "away")]["outcome_point"] == -11.5
    assert by_market_side[("totals", "over")]["outcome_point"] == 216.5
    assert by_market_side[("totals", "under")]["outcome_name"] == "Under"


def test_fetch_public_odds_aggregates_scoreboard_and_summary_payloads(tmp_path, monkeypatch) -> None:
    scoreboard_payload = {"events": [_sample_scoreboard_event()]}
    summary_payload = _sample_summary_payload()
    client = StubClient(
        {
            ("scoreboard", "20260319"): scoreboard_payload,
            ("summary", "401810863"): summary_payload,
        },
        raw_dir=tmp_path,
    )
    teams_df = pd.DataFrame(
        [
            {"team_abbrev": "OSU", "team_name": "Ohio State"},
            {"team_abbrev": "TCU", "team_name": "TCU"},
        ]
    )

    ncaam_event = {
        "id": "401856479",
        "date": "2026-03-19T01:15Z",
        "competitions": [
            {
                "id": "401856479",
                "date": "2026-03-19T01:15Z",
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {
                            "displayName": "Ohio State Buckeyes",
                            "shortDisplayName": "Ohio State",
                            "abbreviation": "OSU",
                            "location": "Ohio State",
                            "name": "Buckeyes",
                        },
                    },
                    {
                        "homeAway": "away",
                        "team": {
                            "displayName": "TCU Horned Frogs",
                            "shortDisplayName": "TCU",
                            "abbreviation": "TCU",
                            "location": "TCU",
                            "name": "Horned Frogs",
                        },
                    },
                ],
            }
        ],
    }
    ncaam_summary = {
        "header": {"id": "401856479", "competitions": ncaam_event["competitions"]},
        "pickcenter": [
            {
                "provider": {"id": "100", "name": "Draft Kings"},
                "homeTeamOdds": {"moneyLine": -310, "spreadOdds": -115.0, "favorite": True},
                "awayTeamOdds": {"moneyLine": 250, "spreadOdds": -105.0, "favorite": False},
                "pointSpread": {
                    "home": {"close": {"line": "-6.5", "odds": "-115"}},
                    "away": {"close": {"line": "+6.5", "odds": "-105"}},
                },
                "total": {
                    "over": {"close": {"line": "o162.5", "odds": "-105"}},
                    "under": {"close": {"line": "u162.5", "odds": "-115"}},
                },
            }
        ],
    }
    client = StubClient(
        {
            ("scoreboard", "20260319"): {"events": [ncaam_event]},
            ("summary", "401856479"): ncaam_summary,
        },
        raw_dir=tmp_path,
    )

    monkeypatch.setattr(odds_api, "_date_keys_for_window", lambda days_ahead: ["20260319"])

    result = odds_api.fetch_public_odds(
        client,
        league="NCAAM",
        sport_key="basketball_ncaab",
        source="ncaam_odds",
        teams_df=teams_df,
        upcoming_days=14,
    )

    assert len(result.dataframe) == 6
    assert result.metadata["provider"] == "espn_pickcenter"
    assert result.metadata["n_events"] == 1
    assert result.dataframe["home_team"].dropna().unique().tolist() == ["OSU"]
    assert result.dataframe["away_team"].dropna().unique().tolist() == ["TCU"]
    assert Path(result.raw_path).exists()
