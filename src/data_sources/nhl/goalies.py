from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.common.time import utc_now_iso
from src.data_sources.base import HttpClient, SourceFetchResult

NHL_API_BASE = "https://api-web.nhle.com/v1"


def _toi_to_minutes(toi: str | None) -> float | None:
    if not toi or ":" not in str(toi):
        return None
    mm, ss = str(toi).split(":")
    return int(mm) + int(ss) / 60.0


def fetch_goalie_game_stats(
    client: HttpClient,
    game_ids: Iterable[int],
    max_games: int = 300,
) -> SourceFetchResult:
    as_of_utc = utc_now_iso()
    rows: list[dict] = []
    raw_paths: list[str] = []
    penalty_rows: dict[tuple[int, str], dict[str, int]] = {}
    pp_goal_rows: dict[tuple[int, str], int] = {}

    requested_game_ids = list(game_ids)
    selected_game_ids = requested_game_ids[-max_games:]
    for game_id in selected_game_ids:
        try:
            url = f"{NHL_API_BASE}/gamecenter/{int(game_id)}/boxscore"
            payload, raw_path = client.get_json("nhl_goalies", url, key=str(game_id))
            raw_paths.append(raw_path)
        except Exception:
            continue

        stats = payload.get("playerByGameStats", {})
        away_team = payload.get("awayTeam", {}).get("abbrev")
        home_team = payload.get("homeTeam", {}).get("abbrev")
        game_date = payload.get("gameDate")

        try:
            landing_url = f"{NHL_API_BASE}/gamecenter/{int(game_id)}/landing"
            landing, _ = client.get_json("nhl_goalies", landing_url, key=f"{game_id}_landing")
            for period_block in landing.get("summary", {}).get("penalties", []):
                for pen in period_block.get("penalties", []):
                    team = pen.get("teamAbbrev", {}).get("default")
                    if not team:
                        continue
                    key = (int(game_id), team)
                    penalty_rows.setdefault(key, {"penalties_taken": 0, "penalties_drawn": 0})
                    penalty_rows[key]["penalties_taken"] += 1
                    opp = home_team if team == away_team else away_team
                    opp_key = (int(game_id), opp)
                    penalty_rows.setdefault(opp_key, {"penalties_taken": 0, "penalties_drawn": 0})
                    penalty_rows[opp_key]["penalties_drawn"] += 1

            for period_block in landing.get("summary", {}).get("scoring", []):
                for goal in period_block.get("goals", []):
                    strength = str(goal.get("strength", "")).lower()
                    team = goal.get("teamAbbrev", {}).get("default")
                    if strength == "pp" and team:
                        key = (int(game_id), team)
                        pp_goal_rows[key] = pp_goal_rows.get(key, 0) + 1
        except Exception:
            pass

        for side, team, opp, is_home in [
            ("awayTeam", away_team, home_team, 0),
            ("homeTeam", home_team, away_team, 1),
        ]:
            for goalie in stats.get(side, {}).get("goalies", []):
                rows.append(
                    {
                        "game_id": int(game_id),
                        "game_date_utc": game_date,
                        "team": team,
                        "opponent": opp,
                        "is_home": is_home,
                        "goalie_id": goalie.get("playerId"),
                        "goalie_name": goalie.get("name", {}).get("default"),
                        "starter_status": "confirmed" if goalie.get("starter") else "probable",
                        "save_pct": goalie.get("savePctg"),
                        "goals_against": goalie.get("goalsAgainst"),
                        "shots_against": goalie.get("shotsAgainst"),
                        "saves": goalie.get("saves"),
                        "toi_minutes": _toi_to_minutes(goalie.get("toi")),
                        "decision": goalie.get("decision"),
                        "penalties_taken": penalty_rows.get((int(game_id), team), {}).get("penalties_taken"),
                        "penalties_drawn": penalty_rows.get((int(game_id), team), {}).get("penalties_drawn"),
                        "pp_goals": pp_goal_rows.get((int(game_id), team), 0),
                    }
                )

            shots_for = payload.get("homeTeam", {}).get("sog") if is_home == 1 else payload.get("awayTeam", {}).get("sog")
            shots_against = payload.get("awayTeam", {}).get("sog") if is_home == 1 else payload.get("homeTeam", {}).get("sog")
            rows.append(
                {
                    "game_id": int(game_id),
                    "game_date_utc": game_date,
                    "team": team,
                    "opponent": opp,
                    "is_home": is_home,
                    "goalie_id": None,
                    "goalie_name": None,
                    "starter_status": "unknown",
                    "save_pct": None,
                    "goals_against": None,
                    "shots_against": shots_against,
                    "saves": None,
                    "toi_minutes": None,
                    "decision": None,
                    "shots_for": shots_for,
                    "penalties_taken": penalty_rows.get((int(game_id), team), {}).get("penalties_taken"),
                    "penalties_drawn": penalty_rows.get((int(game_id), team), {}).get("penalties_drawn"),
                    "pp_goals": pp_goal_rows.get((int(game_id), team), 0),
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
    snapshot_id = client.snapshot_id("nhl_goalies", metadata)
    return SourceFetchResult(
        source="nhl_goalies",
        snapshot_id=snapshot_id,
        extracted_at_utc=as_of_utc,
        raw_path=raw_paths[-1] if raw_paths else "",
        metadata=metadata,
        dataframe=df,
    )
