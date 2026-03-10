from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.common.time import utc_now_iso
from src.data_sources.base import HttpClient, SourceFetchResult

ESPN_NCAAM_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams"


def fetch_teams(client: HttpClient, as_of_date: str | None = None) -> SourceFetchResult:
    if as_of_date is None:
        as_of_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    as_of_utc = utc_now_iso()

    payload, raw_path = client.get_json("ncaam_teams", ESPN_NCAAM_TEAMS_URL, params={"limit": 500}, key=as_of_date)

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
                "team_display_name": team.get("displayName"),
                "team_short_name": team.get("shortDisplayName"),
                "team_location": team.get("location"),
                "team_nickname": team.get("name"),
                "conference": None,
                "division": None,
                "wins": None,
                "losses": None,
                "ot_losses": 0,
                "points": None,
                "goal_diff": None,
                "goal_pct": None,
                "x_streak": None,
                "logo_url": ((team.get("logos") or [{}])[0] or {}).get("href"),
            }
        )

    df = pd.DataFrame(rows)
    metadata = {
        "as_of_date": as_of_date,
        "n_teams": int(len(df)),
        "fetched_at_utc": as_of_utc,
    }
    snapshot_id = client.snapshot_id("ncaam_teams", metadata)
    return SourceFetchResult(
        source="ncaam_teams",
        snapshot_id=snapshot_id,
        extracted_at_utc=as_of_utc,
        raw_path=raw_path,
        metadata=metadata,
        dataframe=df,
    )
