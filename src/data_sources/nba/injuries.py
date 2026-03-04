from __future__ import annotations

import pandas as pd

from src.common.time import utc_now_iso
from src.data_sources.base import HttpClient, SourceFetchResult



def fetch_injuries_proxy(client: HttpClient, teams: list[str]) -> SourceFetchResult:
    # Public no-auth injury feeds are unstable; return explicit fallback payload.
    as_of_utc = utc_now_iso()
    rows = [{"team": t, "man_games_lost_proxy": None, "lineup_uncertainty": 1} for t in sorted(set(teams))]
    df = pd.DataFrame(rows)
    metadata = {
        "n_teams": int(len(rows)),
        "fetched_at_utc": as_of_utc,
        "fallback_used": 1,
        "reason": "no_stable_public_injury_api_without_credentials",
    }
    snapshot_id = client.snapshot_id("nba_injuries", metadata)
    return SourceFetchResult(
        source="nba_injuries",
        snapshot_id=snapshot_id,
        extracted_at_utc=as_of_utc,
        raw_path="",
        metadata=metadata,
        dataframe=df,
    )
