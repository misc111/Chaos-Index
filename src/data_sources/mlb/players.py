from __future__ import annotations

from src.data_sources.base import HttpClient, SourceFetchResult
from src.data_sources.mlb._helpers import fetch_game_summary
from src.common.time import utc_now_iso

import pandas as pd


def fetch_players(
    client: HttpClient,
    team_abbrevs: list[str] | None = None,
    season: str | None = None,
    games_df: pd.DataFrame | None = None,
) -> SourceFetchResult:
    del team_abbrevs, season
    as_of_utc = utc_now_iso()
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

        rosters = payload.get("rosters") or []
        for roster in rosters:
            team = (roster.get("team") or {}).get("abbreviation")
            if not team:
                continue
            for athlete in roster.get("roster") or []:
                slot = pd.to_numeric(athlete.get("batOrder"), errors="coerce")
                if pd.isna(slot):
                    continue
                player = athlete.get("athlete") or {}
                rows.append(
                    {
                        "game_id": int(game_id),
                        "team": team,
                        "player_id": player.get("id"),
                        "player_name": player.get("displayName"),
                        "batting_order_slot": int(slot),
                        "lineup_confirmed": 1,
                        "starter": int(bool(athlete.get("starter"))),
                        "position": (athlete.get("position") or {}).get("abbreviation"),
                        "active": int(bool(athlete.get("active", True))),
                    }
                )

    df = pd.DataFrame(rows)
    metadata = {
        "n_games": int(len(game_ids)),
        "n_rows": int(len(df)),
        "fetched_at_utc": as_of_utc,
        "provider": "espn_summary_rosters",
    }
    snapshot_id = client.snapshot_id("mlb_lineups", metadata)
    return SourceFetchResult(
        source="mlb_lineups",
        snapshot_id=snapshot_id,
        extracted_at_utc=as_of_utc,
        raw_path=raw_paths[-1] if raw_paths else "",
        metadata=metadata,
        dataframe=df,
    )
