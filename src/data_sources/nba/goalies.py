from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.common.time import utc_now_iso
from src.data_sources.base import HttpClient, SourceFetchResult

ESPN_NBA_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"



def _safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except Exception:
        return None



def _parse_made_attempted(text: str | None) -> tuple[int | None, int | None]:
    if not text or "-" not in str(text):
        return None, None
    left, right = str(text).split("-", 1)
    return _safe_int(left), _safe_int(right)



def _team_stats_map(team_block: dict) -> dict[str, int | None]:
    stats = {str(s.get("name")): s.get("displayValue") or s.get("value") for s in team_block.get("statistics", [])}
    ft_made, _ = _parse_made_attempted(stats.get("freeThrowsMade-freeThrowsAttempted"))
    _, fg_attempted = _parse_made_attempted(stats.get("fieldGoalsMade-fieldGoalsAttempted"))
    return {
        "fga": fg_attempted,
        "fouls": _safe_int(stats.get("fouls")),
        "ftm": ft_made,
    }



def fetch_goalie_game_stats(
    client: HttpClient,
    game_ids: Iterable[int],
    max_games: int = 300,
) -> SourceFetchResult:
    as_of_utc = utc_now_iso()
    rows: list[dict] = []
    raw_paths: list[str] = []

    requested_game_ids = list(game_ids)
    selected_game_ids = requested_game_ids[-max_games:]
    for game_id in selected_game_ids:
        try:
            payload, raw_path = client.get_json("nba_goalies", ESPN_NBA_SUMMARY_URL, params={"event": int(game_id)}, key=str(game_id))
            raw_paths.append(raw_path)
        except Exception:
            continue

        teams = (payload.get("boxscore", {}).get("teams") or [])
        if len(teams) != 2:
            continue

        header_comp = ((payload.get("header") or {}).get("competitions") or [{}])[0]
        game_date_utc = (header_comp.get("date") or "")[:10] or None

        parsed_teams = []
        for team_block in teams:
            team_obj = team_block.get("team") or {}
            parsed_teams.append(
                {
                    "team": team_obj.get("abbreviation"),
                    "is_home": 1 if team_block.get("homeAway") == "home" else 0,
                    **_team_stats_map(team_block),
                }
            )

        for t in parsed_teams:
            team = t.get("team")
            if not team:
                continue
            opp = next((o for o in parsed_teams if o.get("team") != team), None)
            if opp is None:
                continue

            rows.append(
                {
                    "game_id": int(game_id),
                    "game_date_utc": game_date_utc,
                    "team": team,
                    "opponent": opp.get("team"),
                    "is_home": t.get("is_home"),
                    "goalie_id": None,
                    "goalie_name": None,
                    "starter_status": "unknown",
                    "save_pct": None,
                    "goals_against": None,
                    "shots_against": opp.get("fga"),
                    "saves": None,
                    "toi_minutes": None,
                    "decision": None,
                    "shots_for": t.get("fga"),
                    "penalties_taken": t.get("fouls"),
                    "penalties_drawn": opp.get("fouls"),
                    "pp_goals": t.get("ftm") or 0,
                }
            )

    df = pd.DataFrame(rows)
    metadata = {
        "n_games_requested": int(len(requested_game_ids)),
        "n_games_fetched": int(len(selected_game_ids)),
        "n_rows": int(len(df)),
        "max_games": int(max_games),
        "fetched_at_utc": as_of_utc,
        "fallback_used": int(df.empty or len(selected_game_ids) < len(requested_game_ids)),
    }
    snapshot_id = client.snapshot_id("nba_goalies", metadata)
    return SourceFetchResult(
        source="nba_goalies",
        snapshot_id=snapshot_id,
        extracted_at_utc=as_of_utc,
        raw_path=raw_paths[-1] if raw_paths else "",
        metadata=metadata,
        dataframe=df,
    )
