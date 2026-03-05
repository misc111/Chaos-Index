from __future__ import annotations

import pandas as pd

from src.data_sources.base import HttpClient, SourceFetchResult
from src.data_sources.odds_api import fetch_public_odds


def fetch_public_odds_optional(client: HttpClient, teams_df: pd.DataFrame | None = None, **_: object) -> SourceFetchResult:
    return fetch_public_odds(
        client,
        league="NHL",
        sport_key="icehockey_nhl",
        source="nhl_odds",
        teams_df=teams_df,
    )
