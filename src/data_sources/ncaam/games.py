from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from src.common.time import utc_now_iso
from src.data_sources.base import HttpClient, SourceFetchResult

ESPN_NCAAM_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"


def _safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except Exception:
        return None


def _season_code(game_date_utc: str | None, season_year: int | None) -> int | None:
    if season_year is not None and season_year >= 1900:
        return int(f"{season_year - 1}{season_year}")
    if not game_date_utc:
        return None
    try:
        dt = datetime.fromisoformat(game_date_utc.replace("Z", "+00:00"))
    except Exception:
        return None
    end_year = dt.year + (1 if dt.month >= 7 else 0)
    return int(f"{end_year - 1}{end_year}")


def _parse_game(event: dict, as_of_utc: str) -> dict | None:
    competition = (event.get("competitions") or [{}])[0]
    competitors = competition.get("competitors") or []
    if len(competitors) < 2:
        return None

    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home or not away:
        return None

    status = competition.get("status", {}).get("type", {})
    completed = bool(status.get("completed"))
    period = _safe_int(competition.get("status", {}).get("period")) or 0

    home_score = _safe_int(home.get("score"))
    away_score = _safe_int(away.get("score"))
    status_final = int(completed and home_score is not None and away_score is not None)
    home_win = None
    if status_final:
        home_win = int(home_score > away_score)

    date_iso = competition.get("date") or event.get("date")
    game_date_utc = date_iso[:10] if isinstance(date_iso, str) and len(date_iso) >= 10 else None
    season_year = _safe_int((event.get("season") or {}).get("year"))

    return {
        "game_id": _safe_int(event.get("id")),
        "season": _season_code(game_date_utc, season_year),
        "game_type": _safe_int((competition.get("type") or {}).get("id")),
        "game_date_utc": game_date_utc,
        "start_time_utc": date_iso,
        "game_state": status.get("shortDetail") or status.get("detail") or status.get("description") or status.get("name"),
        "home_team": (home.get("team") or {}).get("abbreviation"),
        "away_team": (away.get("team") or {}).get("abbreviation"),
        "home_team_id": _safe_int((home.get("team") or {}).get("id")),
        "away_team_id": _safe_int((away.get("team") or {}).get("id")),
        "venue": (competition.get("venue") or {}).get("fullName") or (competition.get("venue") or {}).get("name"),
        "is_neutral_site": int(bool(competition.get("neutralSite", False))),
        "home_score": home_score,
        "away_score": away_score,
        "went_ot": int(period > 2),
        "went_so": 0,
        "home_win": home_win,
        "status_final": status_final,
        "as_of_utc": as_of_utc,
    }


def fetch_games(client: HttpClient, start_date: datetime, end_date: datetime) -> SourceFetchResult:
    as_of_utc = utc_now_iso()
    all_games: dict[int, dict] = {}
    raw_paths: list[str] = []

    cursor = start_date.date()
    end = end_date.date()
    while cursor <= end:
        date_str = cursor.strftime("%Y%m%d")
        payload, raw_path = client.get_json("ncaam_games", ESPN_NCAAM_SCOREBOARD_URL, params={"dates": date_str}, key=date_str)
        raw_paths.append(raw_path)

        for event in payload.get("events", []):
            parsed = _parse_game(event, as_of_utc=as_of_utc)
            if not parsed or parsed.get("game_id") is None:
                continue
            all_games[int(parsed["game_id"])] = parsed

        cursor += timedelta(days=1)

    df = pd.DataFrame(list(all_games.values()))
    if not df.empty:
        df = df.sort_values("start_time_utc").reset_index(drop=True)

    metadata = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "n_games": int(len(df)),
        "n_final": int(df["status_final"].sum()) if not df.empty else 0,
        "raw_paths": raw_paths,
        "fetched_at_utc": as_of_utc,
    }
    snapshot_id = client.snapshot_id("ncaam_games", metadata)
    return SourceFetchResult(
        source="ncaam_games",
        snapshot_id=snapshot_id,
        extracted_at_utc=as_of_utc,
        raw_path=raw_paths[-1] if raw_paths else "",
        metadata=metadata,
        dataframe=df,
    )


def fetch_recent_games(client: HttpClient, history_days: int = 160) -> SourceFetchResult:
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=history_days)
    return fetch_games(client, start_date=start_date, end_date=end_date)
