from __future__ import annotations

import pandas as pd

from src.common.time import utc_now_iso
from src.data_sources.base import HttpClient, SourceFetchResult



def fetch_public_odds_optional(client: HttpClient) -> SourceFetchResult:
    # No API key configured by default; persist explicit fallback metadata.
    as_of_utc = utc_now_iso()
    df = pd.DataFrame(columns=["game_id", "book", "home_price", "away_price", "implied_home_win"])
    metadata = {
        "fetched_at_utc": as_of_utc,
        "fallback_used": 1,
        "reason": "odds_api_key_not_configured",
    }
    snapshot_id = client.snapshot_id("nba_odds", metadata)
    return SourceFetchResult(
        source="nba_odds",
        snapshot_id=snapshot_id,
        extracted_at_utc=as_of_utc,
        raw_path="",
        metadata=metadata,
        dataframe=df,
    )
