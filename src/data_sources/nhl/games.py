from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from src.common.logging import get_logger
from src.common.time import utc_now_iso
from src.data_sources.base import HttpClient, SourceFetchResult

logger = get_logger(__name__)

NHL_API_BASE = "https://api-web.nhle.com/v1"
FINAL_STATES = {"OFF", "FINAL"}


def _parse_game(game: dict, as_of_utc: str, day_date: str | None = None) -> dict:
    away = game.get("awayTeam", {})
    home = game.get("homeTeam", {})
    outcome = game.get("gameOutcome", {})
    period_type = (outcome.get("lastPeriodType") or "REG").upper()

    home_score = home.get("score")
    away_score = away.get("score")
    status_final = int(game.get("gameState") in FINAL_STATES and home_score is not None and away_score is not None)
    home_win = None
    if status_final:
        home_win = int(home_score > away_score)

    return {
        "game_id": int(game["id"]),
        "season": int(game.get("season", 0) or 0),
        "game_type": int(game.get("gameType", 0) or 0),
        "game_date_utc": game.get("gameDate") or day_date,
        "start_time_utc": game.get("startTimeUTC"),
        "game_state": game.get("gameState"),
        "home_team": home.get("abbrev"),
        "away_team": away.get("abbrev"),
        "home_team_id": home.get("id"),
        "away_team_id": away.get("id"),
        "venue": game.get("venue", {}).get("default"),
        "is_neutral_site": int(bool(game.get("neutralSite", False))),
        "home_score": home_score,
        "away_score": away_score,
        "went_ot": int(period_type == "OT"),
        "went_so": int(period_type == "SO"),
        "home_win": home_win,
        "status_final": status_final,
        "as_of_utc": as_of_utc,
    }


def fetch_games(client: HttpClient, start_date: datetime, end_date: datetime) -> SourceFetchResult:
    as_of_utc = utc_now_iso()
    all_games: dict[int, dict] = {}
    raw_paths: list[str] = []

    cursor = start_date
    while cursor <= end_date:
        date_str = cursor.strftime("%Y-%m-%d")
        url = f"{NHL_API_BASE}/schedule/{date_str}"
        payload, raw_path = client.get_json("nhl_games", url, key=date_str)
        raw_paths.append(raw_path)

        for day in payload.get("gameWeek", []):
            day_date = datetime.strptime(day["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if day_date < start_date.replace(tzinfo=timezone.utc) or day_date > end_date.replace(tzinfo=timezone.utc):
                continue
            for game in day.get("games", []):
                parsed = _parse_game(game, as_of_utc, day_date=day.get("date"))
                all_games[parsed["game_id"]] = parsed

        cursor += timedelta(days=7)

    df = pd.DataFrame(list(all_games.values())).sort_values("start_time_utc").reset_index(drop=True)
    metadata = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "n_games": int(len(df)),
        "n_final": int(df["status_final"].sum()) if not df.empty else 0,
        "raw_paths": raw_paths,
        "fetched_at_utc": as_of_utc,
    }
    snapshot_id = client.snapshot_id("nhl_games", metadata)
    return SourceFetchResult(
        source="nhl_games",
        snapshot_id=snapshot_id,
        extracted_at_utc=as_of_utc,
        raw_path=raw_paths[-1] if raw_paths else "",
        metadata=metadata,
        dataframe=df,
    )


def fetch_recent_games(client: HttpClient, history_days: int = 220) -> SourceFetchResult:
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=history_days)
    return fetch_games(client, start_date=start_date, end_date=end_date)
