from __future__ import annotations

import pandas as pd

from src.common.time import utc_now_iso
from src.data_sources.base import HttpClient, SourceFetchResult
from src.data_sources.mlb._helpers import fetch_game_summary


def fetch_injuries_report(
    client: HttpClient,
    teams: list[str] | None = None,
    games_df: pd.DataFrame | None = None,
) -> SourceFetchResult:
    as_of_utc = utc_now_iso()
    team_filter = set(str(team) for team in (teams or []))
    raw_paths: list[str] = []
    rows: list[dict[str, object]] = []

    game_ids = []
    if games_df is not None and not games_df.empty and "game_id" in games_df.columns:
        game_ids = [int(game_id) for game_id in games_df["game_id"].dropna().astype(int).tolist()]

    for game_id in game_ids:
        try:
            payload, raw_path = fetch_game_summary(client, game_id)
            raw_paths.append(raw_path)
        except Exception:
            continue

        for team_block in payload.get("injuries") or []:
            team = (team_block.get("team") or {}).get("abbreviation")
            if not team or (team_filter and team not in team_filter):
                continue
            position_player_out_count = 0
            pitcher_out_count = 0
            for injury in team_block.get("injuries") or []:
                pos = str((((injury.get("details") or {}).get("type") or {}).get("abbreviation")) or "").upper()
                if pos in {"P", "SP", "RP"}:
                    pitcher_out_count += 1
                else:
                    position_player_out_count += 1
            rows.append(
                {
                    "game_id": int(game_id),
                    "team": team,
                    "position_player_out_count": position_player_out_count,
                    "pitcher_out_count": pitcher_out_count,
                }
            )

    df = pd.DataFrame(rows)
    metadata = {
        "n_games": int(len(game_ids)),
        "n_rows": int(len(df)),
        "fetched_at_utc": as_of_utc,
        "provider": "espn_summary_injuries",
    }
    snapshot_id = client.snapshot_id("mlb_injuries", metadata)
    return SourceFetchResult(
        source="mlb_injuries",
        snapshot_id=snapshot_id,
        extracted_at_utc=as_of_utc,
        raw_path=raw_paths[-1] if raw_paths else "",
        metadata=metadata,
        dataframe=df,
    )
