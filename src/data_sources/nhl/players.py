from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.common.time import utc_now_iso
from src.data_sources.base import HttpClient, SourceFetchResult

NHL_API_BASE = "https://api-web.nhle.com/v1"


def fetch_players(client: HttpClient, team_abbrevs: list[str], season: int | str) -> SourceFetchResult:
    as_of_utc = utc_now_iso()
    season_str = str(season)
    rows: list[dict] = []
    raw_paths: list[str] = []

    for team in sorted(set(team_abbrevs)):
        url = f"{NHL_API_BASE}/club-stats/{team}/{season_str}/2"
        try:
            payload, raw_path = client.get_json("nhl_players", url, key=f"{team}_{season_str}")
            raw_paths.append(raw_path)
        except Exception:
            continue

        for sk in payload.get("skaters", []):
            rows.append(
                {
                    "season": season_str,
                    "team": team,
                    "player_id": sk.get("playerId"),
                    "position": sk.get("positionCode"),
                    "games_played": sk.get("gamesPlayed"),
                    "goals": sk.get("goals"),
                    "assists": sk.get("assists"),
                    "points": sk.get("points"),
                    "toi_per_game": sk.get("avgToi"),
                    "plus_minus": sk.get("plusMinus"),
                }
            )

    df = pd.DataFrame(rows)
    metadata = {
        "season": season_str,
        "n_rows": int(len(df)),
        "n_teams": int(len(set(team_abbrevs))),
        "fetched_at_utc": as_of_utc,
        "fallback_used": int(df.empty),
    }
    snapshot_id = client.snapshot_id("nhl_players", metadata)
    return SourceFetchResult(
        source="nhl_players",
        snapshot_id=snapshot_id,
        extracted_at_utc=as_of_utc,
        raw_path=raw_paths[-1] if raw_paths else "",
        metadata=metadata,
        dataframe=df,
    )
