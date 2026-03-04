from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.common.time import utc_now_iso
from src.data_sources.base import HttpClient, SourceFetchResult

ESPN_NBA_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"



def fetch_teams(client: HttpClient, as_of_date: str | None = None) -> SourceFetchResult:
    if as_of_date is None:
        as_of_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    as_of_utc = utc_now_iso()

    payload, raw_path = client.get_json("nba_teams", ESPN_NBA_TEAMS_URL, key=as_of_date)

    rows = []
    leagues = (((payload.get("sports") or [{}])[0].get("leagues") or [{}]))
    teams = (leagues[0].get("teams") if leagues else []) or []
    for row in teams:
        team = row.get("team") or {}
        rows.append(
            {
                "as_of_date": as_of_date,
                "team_id": team.get("id"),
                "team_abbrev": team.get("abbreviation"),
                "team_name": team.get("displayName"),
                "conference": None,
                "division": None,
                "wins": None,
                "losses": None,
                "ot_losses": 0,
                "points": None,
                "goal_diff": None,
                "goal_pct": None,
                "x_streak": None,
            }
        )

    df = pd.DataFrame(rows)
    metadata = {
        "as_of_date": as_of_date,
        "n_teams": int(len(df)),
        "fetched_at_utc": as_of_utc,
    }
    snapshot_id = client.snapshot_id("nba_teams", metadata)
    return SourceFetchResult(
        source="nba_teams",
        snapshot_id=snapshot_id,
        extracted_at_utc=as_of_utc,
        raw_path=raw_path,
        metadata=metadata,
        dataframe=df,
    )
