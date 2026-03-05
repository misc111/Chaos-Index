from __future__ import annotations

import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.common.time import utc_now_iso
from src.data_sources.base import HttpClient, SourceFetchResult

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
CENTRAL_TZ = ZoneInfo("America/Chicago")

ODDS_COLUMNS = [
    "league",
    "sport_key",
    "odds_event_id",
    "commence_time_utc",
    "commence_date_central",
    "api_home_team",
    "api_away_team",
    "home_team",
    "away_team",
    "bookmaker_key",
    "bookmaker_title",
    "bookmaker_last_update_utc",
    "market_key",
    "outcome_name",
    "outcome_side",
    "outcome_team",
    "outcome_price",
    "outcome_point",
    "implied_probability",
]

NBA_NAME_ALIASES = {
    "los angeles clippers": "LAC",
    "la clippers": "LAC",
    "los angeles lakers": "LAL",
    "new york knicks": "NY",
    "golden state warriors": "GS",
    "new orleans pelicans": "NO",
    "san antonio spurs": "SA",
    "utah jazz": "UTAH",
    "phoenix suns": "PHX",
    "brooklyn nets": "BKN",
}

NHL_NAME_ALIASES = {
    "utah mammoth": "UTA",
    "utah hockey club": "UTA",
    "new york islanders": "NYI",
    "new york rangers": "NYR",
    "st louis blues": "STL",
    "montreal canadiens": "MTL",
    "los angeles kings": "LAK",
    "new jersey devils": "NJD",
    "san jose sharks": "SJS",
    "columbus blue jackets": "CBJ",
    "tampa bay lightning": "TBL",
    "vegas golden knights": "VGK",
}


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def _parse_env_int(name: str, default: int = 0) -> int:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except Exception:
        return default
    return parsed if parsed >= 0 else default


def _parse_header_int(headers: dict[str, str], key: str) -> int | None:
    lowered = {str(k).lower(): str(v) for k, v in headers.items()}
    raw = headers.get(key) or headers.get(key.lower()) or lowered.get(key.lower())
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _american_to_implied_probability(price: Any) -> float | None:
    try:
        p = float(price)
    except Exception:
        return None
    if p == 0:
        return None
    if p > 0:
        return 100.0 / (p + 100.0)
    return (-p) / ((-p) + 100.0)


def _central_date_key(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    return dt.astimezone(CENTRAL_TZ).date().isoformat()


def _team_aliases_for_league(league: str) -> dict[str, str]:
    if league.upper() == "NBA":
        return NBA_NAME_ALIASES
    return NHL_NAME_ALIASES


def _build_team_name_map(teams_df: pd.DataFrame | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if teams_df is None or teams_df.empty:
        return mapping
    for row in teams_df.itertuples(index=False):
        team_abbrev = str(getattr(row, "team_abbrev", "") or "").strip()
        team_name = str(getattr(row, "team_name", "") or "").strip()
        if not team_abbrev:
            continue
        if team_name:
            mapping[_normalize_text(team_name)] = team_abbrev
        mapping[_normalize_text(team_abbrev)] = team_abbrev
    return mapping


def _resolve_abbrev(
    raw_team_name: str,
    team_name_map: dict[str, str],
    alias_map: dict[str, str],
) -> str | None:
    normalized = _normalize_text(raw_team_name)
    if not normalized:
        return None
    if normalized in team_name_map:
        return team_name_map[normalized]
    if normalized in alias_map:
        return alias_map[normalized]
    return None


def _resolve_outcome_side(outcome_name: str, api_home_team: str, api_away_team: str) -> str | None:
    normalized_outcome = _normalize_text(outcome_name)
    if normalized_outcome in {"over", "under"}:
        return normalized_outcome
    if normalized_outcome == _normalize_text(api_home_team):
        return "home"
    if normalized_outcome == _normalize_text(api_away_team):
        return "away"
    return None


def _maybe_cached_payload_for_throttle(
    client: HttpClient,
    source: str,
    min_interval_seconds: int,
) -> tuple[Any, str] | None:
    if min_interval_seconds <= 0:
        return None
    latest_cached: Path | None = client.latest_cached_file(source)
    if latest_cached is None:
        return None

    age_seconds = max(0.0, datetime.now(timezone.utc).timestamp() - latest_cached.stat().st_mtime)
    if age_seconds >= float(min_interval_seconds):
        return None

    payload = client.load_latest_cached(source)
    if payload is None:
        return None
    return payload, str(latest_cached)


def fetch_public_odds(
    client: HttpClient,
    *,
    league: str,
    sport_key: str,
    source: str,
    teams_df: pd.DataFrame | None = None,
) -> SourceFetchResult:
    as_of_utc = utc_now_iso()
    api_key = str(os.getenv("ODDS_API_KEY", "")).strip()
    regions = str(os.getenv("ODDS_API_REGIONS", "us")).strip() or "us"
    markets = str(os.getenv("ODDS_API_MARKETS", "h2h,spreads,totals")).strip() or "h2h,spreads,totals"
    odds_format = str(os.getenv("ODDS_API_ODDS_FORMAT", "american")).strip() or "american"
    date_format = str(os.getenv("ODDS_API_DATE_FORMAT", "iso")).strip() or "iso"
    throttle_seconds = _parse_env_int("ODDS_API_THROTTLE_SECONDS", default=0)

    if not api_key:
        df = pd.DataFrame(columns=ODDS_COLUMNS)
        metadata = {
            "fetched_at_utc": as_of_utc,
            "league": league,
            "sport_key": sport_key,
            "regions": regions,
            "markets": markets,
            "odds_format": odds_format,
            "date_format": date_format,
            "fallback_used": 1,
            "reason": "odds_api_key_not_configured",
            "api_call_performed": 0,
            "from_cache": 0,
            "throttle_seconds": throttle_seconds,
            "throttle_applied": 0,
            "n_events": 0,
            "n_rows": 0,
        }
        snapshot_id = client.snapshot_id(source, metadata)
        return SourceFetchResult(
            source=source,
            snapshot_id=snapshot_id,
            extracted_at_utc=as_of_utc,
            raw_path="",
            metadata=metadata,
            dataframe=df,
        )

    endpoint = f"{ODDS_API_BASE}/sports/{sport_key}/odds/"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }

    throttled = _maybe_cached_payload_for_throttle(client, source, throttle_seconds)
    if throttled is not None:
        payload, raw_path = throttled
        headers: dict[str, str] = {}
        from_cache = True
        throttle_applied = True
    else:
        payload, raw_path, headers, from_cache = client.get_json_with_headers(
            source=source,
            url=endpoint,
            params=params,
            key=as_of_utc.replace(":", "-"),
        )
        throttle_applied = False

    events = payload if isinstance(payload, list) else []
    team_name_map = _build_team_name_map(teams_df)
    alias_map = _team_aliases_for_league(league)

    rows: list[dict[str, Any]] = []
    for event in events:
        api_home_team = str(event.get("home_team") or "")
        api_away_team = str(event.get("away_team") or "")
        home_team = _resolve_abbrev(api_home_team, team_name_map, alias_map)
        away_team = _resolve_abbrev(api_away_team, team_name_map, alias_map)
        commence_time = str(event.get("commence_time") or "")
        commence_date_central = _central_date_key(commence_time)

        for bookmaker in event.get("bookmakers") or []:
            for market in bookmaker.get("markets") or []:
                market_key = str(market.get("key") or "")
                for outcome in market.get("outcomes") or []:
                    outcome_name = str(outcome.get("name") or "")
                    outcome_side = _resolve_outcome_side(outcome_name, api_home_team, api_away_team)
                    if outcome_side == "home":
                        outcome_team = home_team
                    elif outcome_side == "away":
                        outcome_team = away_team
                    else:
                        outcome_team = None

                    rows.append(
                        {
                            "league": league,
                            "sport_key": str(event.get("sport_key") or sport_key),
                            "odds_event_id": str(event.get("id") or ""),
                            "commence_time_utc": commence_time or None,
                            "commence_date_central": commence_date_central,
                            "api_home_team": api_home_team or None,
                            "api_away_team": api_away_team or None,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker_key": str(bookmaker.get("key") or ""),
                            "bookmaker_title": str(bookmaker.get("title") or ""),
                            "bookmaker_last_update_utc": str(bookmaker.get("last_update") or ""),
                            "market_key": market_key,
                            "outcome_name": outcome_name,
                            "outcome_side": outcome_side,
                            "outcome_team": outcome_team,
                            "outcome_price": pd.to_numeric(outcome.get("price"), errors="coerce"),
                            "outcome_point": pd.to_numeric(outcome.get("point"), errors="coerce"),
                            "implied_probability": _american_to_implied_probability(outcome.get("price")),
                        }
                    )

    df = pd.DataFrame(rows, columns=ODDS_COLUMNS)
    requests_last = _parse_header_int(headers, "x-requests-last")
    requests_used = _parse_header_int(headers, "x-requests-used")
    requests_remaining = _parse_header_int(headers, "x-requests-remaining")

    metadata = {
        "fetched_at_utc": as_of_utc,
        "league": league,
        "sport_key": sport_key,
        "endpoint": endpoint,
        "regions": regions,
        "markets": markets,
        "odds_format": odds_format,
        "date_format": date_format,
        "api_call_performed": int(not from_cache),
        "from_cache": int(from_cache),
        "throttle_seconds": throttle_seconds,
        "throttle_applied": int(throttle_applied),
        "n_events": int(len(events)),
        "n_rows": int(len(df)),
        "n_unique_books": int(df["bookmaker_key"].nunique()) if not df.empty else 0,
        "n_events_with_team_mapping": int(df["odds_event_id"][df["home_team"].notna() & df["away_team"].notna()].nunique())
        if not df.empty
        else 0,
        "requests_last": requests_last,
        "requests_used": requests_used,
        "requests_remaining": requests_remaining,
    }
    snapshot_id = client.snapshot_id(source, metadata)

    return SourceFetchResult(
        source=source,
        snapshot_id=snapshot_id,
        extracted_at_utc=as_of_utc,
        raw_path=raw_path,
        metadata=metadata,
        dataframe=df,
    )
