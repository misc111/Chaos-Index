from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.common.time import utc_now_iso
from src.data_sources.base import HttpClient, SourceFetchResult

NHL_API_BASE = "https://api-web.nhle.com/v1"


def fetch_teams(client: HttpClient, as_of_date: str | None = None) -> SourceFetchResult:
    if as_of_date is None:
        as_of_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    as_of_utc = utc_now_iso()

    url = f"{NHL_API_BASE}/standings/{as_of_date}"
    payload, raw_path = client.get_json("nhl_teams", url, key=as_of_date)

    rows = []
    for row in payload.get("standings", []):
        rows.append(
            {
                "as_of_date": as_of_date,
                "team_id": row.get("teamAbbrev", {}).get("id") or row.get("teamLogo") or row.get("teamName", {}).get("default"),
                "team_abbrev": row.get("teamAbbrev", {}).get("default"),
                "team_name": row.get("teamName", {}).get("default"),
                "conference": row.get("conferenceAbbrev"),
                "division": row.get("divisionAbbrev"),
                "wins": row.get("wins"),
                "losses": row.get("losses"),
                "ot_losses": row.get("otLosses"),
                "points": row.get("points"),
                "goal_diff": row.get("goalDifferential"),
                "goal_pct": row.get("goalForPctg"),
                "x_streak": row.get("streakCode"),
            }
        )

    df = pd.DataFrame(rows)
    metadata = {
        "as_of_date": as_of_date,
        "n_teams": int(len(df)),
        "fetched_at_utc": as_of_utc,
    }
    snapshot_id = client.snapshot_id("nhl_teams", metadata)
    return SourceFetchResult(
        source="nhl_teams",
        snapshot_id=snapshot_id,
        extracted_at_utc=as_of_utc,
        raw_path=raw_path,
        metadata=metadata,
        dataframe=df,
    )
