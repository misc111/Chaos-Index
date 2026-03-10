from __future__ import annotations

import pandas as pd

from src.common.time import utc_now_iso
from src.data_sources.base import HttpClient, SourceFetchResult


def fetch_xg_optional(client: HttpClient) -> SourceFetchResult:
    as_of_utc = utc_now_iso()
    df = pd.DataFrame(columns=["game_id", "team", "xg_for", "xg_against", "hdc_for", "hdc_against"])
    metadata = {
        "fetched_at_utc": as_of_utc,
        "fallback_used": 1,
        "reason": "public_expected_value_source_unavailable_or_schema_unstable",
    }
    snapshot_id = client.snapshot_id("ncaam_xg", metadata)
    return SourceFetchResult(
        source="ncaam_xg",
        snapshot_id=snapshot_id,
        extracted_at_utc=as_of_utc,
        raw_path="",
        metadata=metadata,
        dataframe=df,
    )
