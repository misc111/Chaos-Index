from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from src.common.time import utc_now_iso
from src.data_sources.base import HttpClient, SourceFetchResult
from src.data_sources.mlb._helpers import athlete_stat_map, fetch_game_summary, parse_baseball_innings


def _pick_starter(stat_block: dict[str, Any]) -> dict[str, Any] | None:
    athletes = list(stat_block.get("athletes") or [])
    if not athletes:
        return None

    flagged = [ath for ath in athletes if bool(ath.get("starter"))]
    if flagged:
        return flagged[0]

    sp_position = [
        ath
        for ath in athletes
        if str(((ath.get("position") or {}).get("abbreviation") or "")).upper() == "SP"
        or "STARTING PITCHER" in str(((ath.get("position") or {}).get("displayName") or "")).upper()
    ]
    if sp_position:
        return sp_position[0]

    def innings_key(ath: dict[str, Any]) -> float:
        parsed = parse_baseball_innings(athlete_stat_map(stat_block, ath).get("fullInnings.partInnings"))
        return float(parsed or 0.0)

    return max(athletes, key=innings_key)


def fetch_starter_context(
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

        player_groups = (payload.get("boxscore") or {}).get("players") or []
        for group in player_groups:
            team = (group.get("team") or {}).get("abbreviation")
            if not team:
                continue
            pitching_block = next((block for block in group.get("statistics") or [] if str(block.get("type") or "").lower() == "pitching"), None)
            if not pitching_block:
                continue
            starter = _pick_starter(pitching_block)
            if not starter:
                continue
            stats = athlete_stat_map(pitching_block, starter)
            athlete = starter.get("athlete") or {}
            rows.append(
                {
                    "game_id": int(game_id),
                    "team": team,
                    "starter_pitcher_id": athlete.get("id"),
                    "starter_pitcher_name": athlete.get("displayName"),
                    "starter_status": "actual",
                    "innings_pitched": parse_baseball_innings(stats.get("fullInnings.partInnings")),
                    "starter_era": pd.to_numeric(stats.get("ERA"), errors="coerce"),
                    "starter_whip": None,
                    "pitcher_strikeouts": pd.to_numeric(stats.get("strikeouts"), errors="coerce"),
                    "pitcher_walks": pd.to_numeric(stats.get("walks"), errors="coerce"),
                    "runs_allowed": pd.to_numeric(stats.get("runs"), errors="coerce"),
                    "hits_allowed": pd.to_numeric(stats.get("hits"), errors="coerce"),
                    "home_runs_allowed": pd.to_numeric(stats.get("homeRuns"), errors="coerce"),
                }
            )

    df = pd.DataFrame(rows)
    metadata = {
        "n_games_requested": int(len(requested_game_ids)),
        "n_games_fetched": int(len(selected_game_ids)),
        "n_rows": int(len(df)),
        "max_games": int(max_games),
        "fetched_at_utc": as_of_utc,
        "provider": "espn_summary_boxscore_pitching",
    }
    snapshot_id = client.snapshot_id("mlb_starting_pitchers", metadata)
    return SourceFetchResult(
        source="mlb_starting_pitchers",
        snapshot_id=snapshot_id,
        extracted_at_utc=as_of_utc,
        raw_path=raw_paths[-1] if raw_paths else "",
        metadata=metadata,
        dataframe=df,
    )
