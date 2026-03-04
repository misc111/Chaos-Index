from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.data_sources.base import HttpClient, SourceFetchResult
from src.data_sources.nhl.games import fetch_games



def fetch_upcoming_schedule(client: HttpClient, days_ahead: int = 14) -> SourceFetchResult:
    start = datetime.now(timezone.utc)
    end = start + timedelta(days=days_ahead)
    res = fetch_games(client, start_date=start, end_date=end)
    if not res.dataframe.empty:
        res.dataframe = res.dataframe[res.dataframe["status_final"] == 0].copy()
        res.metadata["n_upcoming"] = int(len(res.dataframe))
    res.source = "nhl_schedule"
    return res
