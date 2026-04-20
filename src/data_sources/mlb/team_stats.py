from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.common.time import utc_now_iso
from src.data_sources.base import HttpClient, SourceFetchResult
from src.data_sources.mlb._helpers import fetch_game_summary, stat_value_map


def fetch_team_game_stats(
    client: HttpClient,
    game_ids: Iterable[int],
    max_games: int = 350,
) -> SourceFetchResult:
    as_of_utc = utc_now_iso()
    raw_paths: list[str] = []
    rows: list[dict[str, object]] = []

    requested_game_ids = [int(game_id) for game_id in game_ids]
    selected_game_ids = requested_game_ids[-max_games:]

    for game_id in selected_game_ids:
        try:
            payload, raw_path = fetch_game_summary(client, game_id)
            raw_paths.append(raw_path)
        except Exception:
            continue

        teams = (payload.get("boxscore") or {}).get("teams") or []
        for team_block in teams:
            team = (team_block.get("team") or {}).get("abbreviation")
            if not team:
                continue
            stat_blocks = {str(block.get("name") or ""): stat_value_map(block.get("stats")) for block in team_block.get("statistics") or []}
            batting = stat_blocks.get("batting", {})
            pitching = stat_blocks.get("pitching", {})
            rows.append(
                {
                    "game_id": int(game_id),
                    "team": team,
                    "hits": batting.get("hits"),
                    "walks": batting.get("walks"),
                    "strikeouts": batting.get("strikeouts"),
                    "home_runs": batting.get("homeRuns"),
                    "total_bases": batting.get("totalBases"),
                    "stolen_bases": batting.get("stolenBases"),
                    "pitching_hits_allowed": pitching.get("hits"),
                    "pitching_walks_allowed": pitching.get("walks"),
                    "pitching_strikeouts": pitching.get("strikeouts"),
                    "pitching_home_runs_allowed": pitching.get("homeRuns"),
                    "pitching_runs_allowed": pitching.get("runs"),
                }
            )

    df = pd.DataFrame(rows)
    metadata = {
        "n_games_requested": int(len(requested_game_ids)),
        "n_games_fetched": int(len(selected_game_ids)),
        "n_rows": int(len(df)),
        "max_games": int(max_games),
        "fetched_at_utc": as_of_utc,
        "provider": "espn_summary_boxscore",
    }
    snapshot_id = client.snapshot_id("mlb_team_game_stats", metadata)
    return SourceFetchResult(
        source="mlb_team_game_stats",
        snapshot_id=snapshot_id,
        extracted_at_utc=as_of_utc,
        raw_path=raw_paths[-1] if raw_paths else "",
        metadata=metadata,
        dataframe=df,
    )
