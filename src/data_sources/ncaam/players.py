from __future__ import annotations

import pandas as pd

from src.common.time import utc_now_iso
from src.data_sources.base import HttpClient, SourceFetchResult


def fetch_players(
    client: HttpClient,
    team_abbrevs: list[str],
    season: int | str,
    games_df: pd.DataFrame | None = None,
) -> SourceFetchResult:
    del games_df

    as_of_utc = utc_now_iso()
    season_str = str(season)
    df = pd.DataFrame(
        columns=[
            "season",
            "team",
            "player_id",
            "player_name",
            "position",
            "status",
            "injury_status",
            "injury_date",
        ]
    )
    metadata = {
        "season": season_str,
        "n_rows": 0,
        "n_teams": int(len(set(team_abbrevs))),
        "fetched_at_utc": as_of_utc,
        "fallback_used": 1,
        "mode": "disabled_for_scale_until_ncaam_player_features_exist",
    }
    snapshot_id = client.snapshot_id("ncaam_players", metadata)
    return SourceFetchResult(
        source="ncaam_players",
        snapshot_id=snapshot_id,
        extracted_at_utc=as_of_utc,
        raw_path="",
        metadata=metadata,
        dataframe=df,
    )
