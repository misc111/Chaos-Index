"""Legacy module name for the repo-owned public odds fetcher.

This implementation no longer depends on The Odds API. It builds odds snapshots
from ESPN's public scoreboard + summary payloads, specifically the `pickcenter`
markets exposed on each event summary.
"""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.common.time import utc_now_iso
from src.data_sources.base import HttpClient, SourceFetchResult

CENTRAL_TZ = ZoneInfo("America/Chicago")
DEFAULT_MARKETS = "h2h,spreads,totals"
DEFAULT_ODDS_FORMAT = "american"
DEFAULT_DATE_FORMAT = "iso"

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

ESPN_ENDPOINTS = {
    "NBA": {
        "scoreboard": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
        "summary": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary",
    },
    "NCAAM": {
        "scoreboard": "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard",
        "summary": "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary",
    },
    "NHL": {
        "scoreboard": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
        "summary": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary",
    },
}

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

NCAAM_NAME_ALIASES = {
    "uconn": "UCONN",
    "uconn huskies": "UCONN",
    "connecticut": "CONN",
    "connecticut huskies": "CONN",
    "unc": "UNC",
    "north carolina": "UNC",
    "north carolina tar heels": "UNC",
    "ole miss": "MISS",
    "saint johns": "SJU",
    "st johns": "SJU",
    "saint johns red storm": "SJU",
    "st johns red storm": "SJU",
    "byu": "BYU",
    "lsu": "LSU",
    "smu": "SMU",
    "tcu": "TCU",
    "ucf": "UCF",
    "ucla": "UCLA",
    "usc": "USC",
    "utah state": "USU",
}


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def _slugify(value: Any) -> str:
    text = _normalize_text(value).replace(" ", "_")
    return text or "espn_pickcenter"


def _parse_float(value: Any) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    cleaned = raw.replace(",", "")
    cleaned = re.sub(r"^[ou]", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.lstrip("+")
    try:
        return float(cleaned)
    except Exception:
        return None


def _parse_american_odds(value: Any) -> float | None:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    return float(int(parsed)) if float(parsed).is_integer() else float(parsed)


def _american_to_implied_probability(price: Any) -> float | None:
    value = _parse_american_odds(price)
    if value is None or value == 0:
        return None
    if value > 0:
        return 100.0 / (value + 100.0)
    return (-value) / ((-value) + 100.0)


def _central_date_key(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    return dt.astimezone(CENTRAL_TZ).date().isoformat()


def _current_central_date() -> date:
    return datetime.now(timezone.utc).astimezone(CENTRAL_TZ).date()


def _date_keys_for_window(days_ahead: int) -> list[str]:
    normalized_days = max(0, int(days_ahead))
    start = _current_central_date()
    return [(start + timedelta(days=offset)).strftime("%Y%m%d") for offset in range(normalized_days + 1)]


def _coerce_date(value: date | datetime | str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Historical odds date values must be non-empty")
    if len(raw) == 8 and raw.isdigit():
        return datetime.strptime(raw, "%Y%m%d").date()
    return date.fromisoformat(raw)


def _date_keys_for_range(start_date: date | datetime | str, end_date: date | datetime | str) -> list[str]:
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    if end < start:
        raise ValueError(f"Historical odds date range must be ascending: start={start} end={end}")
    span = (end - start).days
    return [(start + timedelta(days=offset)).strftime("%Y%m%d") for offset in range(span + 1)]


def _team_aliases_for_league(league: str) -> dict[str, str]:
    league_code = str(league or "").upper()
    if league_code == "NBA":
        return NBA_NAME_ALIASES
    if league_code == "NCAAM":
        return NCAAM_NAME_ALIASES
    return NHL_NAME_ALIASES


def _build_team_name_map(teams_df: pd.DataFrame | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if teams_df is None or teams_df.empty:
        return mapping
    for row in teams_df.itertuples(index=False):
        team_abbrev = str(getattr(row, "team_abbrev", "") or "").strip()
        if not team_abbrev:
            continue
        candidate_names = [
            getattr(row, "team_name", ""),
            getattr(row, "team_short_name", ""),
            getattr(row, "team_location", ""),
            getattr(row, "team_nickname", ""),
            getattr(row, "team_display_name", ""),
            team_abbrev,
        ]
        for candidate in candidate_names:
            normalized = _normalize_text(candidate)
            if normalized:
                mapping[normalized] = team_abbrev
    return mapping


def _resolve_abbrev(raw_team_name: Any, team_name_map: dict[str, str], alias_map: dict[str, str]) -> str | None:
    normalized = _normalize_text(raw_team_name)
    if not normalized:
        return None
    if normalized in team_name_map:
        return team_name_map[normalized]
    if normalized in alias_map:
        return alias_map[normalized]
    return None


def _resolve_competitor_abbrev(
    competitor: dict[str, Any] | None,
    team_name_map: dict[str, str],
    alias_map: dict[str, str],
) -> str | None:
    team = (competitor or {}).get("team") or {}
    candidates = [
        team.get("abbreviation"),
        team.get("shortDisplayName"),
        team.get("displayName"),
        team.get("location"),
        team.get("name"),
    ]
    for candidate in candidates:
        resolved = _resolve_abbrev(candidate, team_name_map, alias_map)
        if resolved:
            return resolved

    raw_abbrev = str(team.get("abbreviation") or "").strip()
    return raw_abbrev or None


def _competition_from_summary(summary_payload: dict[str, Any], scoreboard_event: dict[str, Any]) -> dict[str, Any]:
    summary_header = summary_payload.get("header") or {}
    competitions = summary_header.get("competitions") or summary_payload.get("competitions") or scoreboard_event.get("competitions") or []
    return competitions[0] if competitions else {}


def _competition_time_utc(competition: dict[str, Any], scoreboard_event: dict[str, Any]) -> str | None:
    for value in (
        competition.get("date"),
        competition.get("startDate"),
        scoreboard_event.get("date"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return None


def _preferred_event_team_name(competitor: dict[str, Any] | None) -> str | None:
    team = (competitor or {}).get("team") or {}
    for value in (
        team.get("displayName"),
        team.get("shortDisplayName"),
        team.get("name"),
        team.get("location"),
        team.get("abbreviation"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return None


def _bookmaker_fields(entry: dict[str, Any]) -> tuple[str, str]:
    provider = entry.get("provider") or {}
    title = str(provider.get("name") or "ESPN PickCenter").strip() or "ESPN PickCenter"
    key = str(provider.get("id") or "").strip() or _slugify(title)
    return key, title


def _pick_nested_value(node: dict[str, Any], *path: str) -> Any:
    current: Any = node
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _pick_moneyline_price(entry: dict[str, Any], side: str) -> float | None:
    prices = [
        _pick_nested_value(entry, "moneyline", side, "close", "odds"),
        _pick_nested_value(entry, "moneyline", side, "open", "odds"),
        _pick_nested_value(entry, f"{side}TeamOdds", "moneyLine"),
    ]
    for candidate in prices:
        parsed = _parse_american_odds(candidate)
        if parsed is not None:
            return parsed
    return None


def _pick_spread_price(entry: dict[str, Any], side: str) -> float | None:
    prices = [
        _pick_nested_value(entry, "pointSpread", side, "close", "odds"),
        _pick_nested_value(entry, "pointSpread", side, "open", "odds"),
        _pick_nested_value(entry, f"{side}TeamOdds", "spreadOdds"),
    ]
    for candidate in prices:
        parsed = _parse_american_odds(candidate)
        if parsed is not None:
            return parsed
    return None


def _pick_spread_point(entry: dict[str, Any], side: str) -> float | None:
    for candidate in (
        _pick_nested_value(entry, "pointSpread", side, "close", "line"),
        _pick_nested_value(entry, "pointSpread", side, "open", "line"),
    ):
        parsed = _parse_float(candidate)
        if parsed is not None:
            return parsed

    base_spread = _parse_float(entry.get("spread"))
    if base_spread is None:
        return None

    absolute_spread = abs(base_spread)
    side_odds = entry.get(f"{side}TeamOdds") or {}
    is_favorite = bool(side_odds.get("favorite"))
    return -absolute_spread if is_favorite else absolute_spread


def _pick_total_price(entry: dict[str, Any], side: str) -> float | None:
    prices = [
        _pick_nested_value(entry, "total", side, "close", "odds"),
        _pick_nested_value(entry, "total", side, "open", "odds"),
        entry.get("overOdds") if side == "over" else entry.get("underOdds"),
    ]
    for candidate in prices:
        parsed = _parse_american_odds(candidate)
        if parsed is not None:
            return parsed
    return None


def _pick_total_point(entry: dict[str, Any], side: str) -> float | None:
    for candidate in (
        _pick_nested_value(entry, "total", side, "close", "line"),
        _pick_nested_value(entry, "total", side, "open", "line"),
        entry.get("overUnder"),
    ):
        parsed = _parse_float(candidate)
        if parsed is not None:
            return parsed
    return None


def _row_payload(
    *,
    league: str,
    sport_key: str,
    event_id: str,
    commence_time_utc: str | None,
    api_home_team: str | None,
    api_away_team: str | None,
    home_team: str | None,
    away_team: str | None,
    bookmaker_key: str,
    bookmaker_title: str,
    bookmaker_last_update_utc: str,
    market_key: str,
    outcome_name: str,
    outcome_side: str,
    outcome_team: str | None,
    outcome_price: float | None,
    outcome_point: float | None,
) -> dict[str, Any]:
    return {
        "league": league,
        "sport_key": sport_key,
        "odds_event_id": event_id,
        "commence_time_utc": commence_time_utc,
        "commence_date_central": _central_date_key(commence_time_utc),
        "api_home_team": api_home_team,
        "api_away_team": api_away_team,
        "home_team": home_team,
        "away_team": away_team,
        "bookmaker_key": bookmaker_key,
        "bookmaker_title": bookmaker_title,
        "bookmaker_last_update_utc": bookmaker_last_update_utc,
        "market_key": market_key,
        "outcome_name": outcome_name,
        "outcome_side": outcome_side,
        "outcome_team": outcome_team,
        "outcome_price": outcome_price,
        "outcome_point": outcome_point,
        "implied_probability": _american_to_implied_probability(outcome_price),
    }


def _flatten_pickcenter_summary(
    *,
    summary_payload: dict[str, Any],
    scoreboard_event: dict[str, Any],
    league: str,
    sport_key: str,
    team_name_map: dict[str, str],
    alias_map: dict[str, str],
    as_of_utc: str,
) -> list[dict[str, Any]]:
    competition = _competition_from_summary(summary_payload, scoreboard_event)
    competitors = competition.get("competitors") or []
    home_competitor = next((c for c in competitors if str(c.get("homeAway") or "").lower() == "home"), None)
    away_competitor = next((c for c in competitors if str(c.get("homeAway") or "").lower() == "away"), None)
    if home_competitor is None or away_competitor is None:
        return []

    api_home_team = _preferred_event_team_name(home_competitor)
    api_away_team = _preferred_event_team_name(away_competitor)
    home_team = _resolve_competitor_abbrev(home_competitor, team_name_map, alias_map)
    away_team = _resolve_competitor_abbrev(away_competitor, team_name_map, alias_map)
    event_id = str(competition.get("id") or summary_payload.get("id") or scoreboard_event.get("id") or "").strip()
    commence_time_utc = _competition_time_utc(competition, scoreboard_event)
    if not event_id:
        return []

    rows: list[dict[str, Any]] = []
    for entry in summary_payload.get("pickcenter") or []:
        bookmaker_key, bookmaker_title = _bookmaker_fields(entry)
        bookmaker_last_update_utc = as_of_utc

        home_moneyline = _pick_moneyline_price(entry, "home")
        away_moneyline = _pick_moneyline_price(entry, "away")
        if home_moneyline is not None:
            rows.append(
                _row_payload(
                    league=league,
                    sport_key=sport_key,
                    event_id=event_id,
                    commence_time_utc=commence_time_utc,
                    api_home_team=api_home_team,
                    api_away_team=api_away_team,
                    home_team=home_team,
                    away_team=away_team,
                    bookmaker_key=bookmaker_key,
                    bookmaker_title=bookmaker_title,
                    bookmaker_last_update_utc=bookmaker_last_update_utc,
                    market_key="h2h",
                    outcome_name=api_home_team or "Home",
                    outcome_side="home",
                    outcome_team=home_team,
                    outcome_price=home_moneyline,
                    outcome_point=None,
                )
            )
        if away_moneyline is not None:
            rows.append(
                _row_payload(
                    league=league,
                    sport_key=sport_key,
                    event_id=event_id,
                    commence_time_utc=commence_time_utc,
                    api_home_team=api_home_team,
                    api_away_team=api_away_team,
                    home_team=home_team,
                    away_team=away_team,
                    bookmaker_key=bookmaker_key,
                    bookmaker_title=bookmaker_title,
                    bookmaker_last_update_utc=bookmaker_last_update_utc,
                    market_key="h2h",
                    outcome_name=api_away_team or "Away",
                    outcome_side="away",
                    outcome_team=away_team,
                    outcome_price=away_moneyline,
                    outcome_point=None,
                )
            )

        home_spread_point = _pick_spread_point(entry, "home")
        home_spread_price = _pick_spread_price(entry, "home")
        away_spread_point = _pick_spread_point(entry, "away")
        away_spread_price = _pick_spread_price(entry, "away")
        if home_spread_price is not None or home_spread_point is not None:
            rows.append(
                _row_payload(
                    league=league,
                    sport_key=sport_key,
                    event_id=event_id,
                    commence_time_utc=commence_time_utc,
                    api_home_team=api_home_team,
                    api_away_team=api_away_team,
                    home_team=home_team,
                    away_team=away_team,
                    bookmaker_key=bookmaker_key,
                    bookmaker_title=bookmaker_title,
                    bookmaker_last_update_utc=bookmaker_last_update_utc,
                    market_key="spreads",
                    outcome_name=api_home_team or "Home",
                    outcome_side="home",
                    outcome_team=home_team,
                    outcome_price=home_spread_price,
                    outcome_point=home_spread_point,
                )
            )
        if away_spread_price is not None or away_spread_point is not None:
            rows.append(
                _row_payload(
                    league=league,
                    sport_key=sport_key,
                    event_id=event_id,
                    commence_time_utc=commence_time_utc,
                    api_home_team=api_home_team,
                    api_away_team=api_away_team,
                    home_team=home_team,
                    away_team=away_team,
                    bookmaker_key=bookmaker_key,
                    bookmaker_title=bookmaker_title,
                    bookmaker_last_update_utc=bookmaker_last_update_utc,
                    market_key="spreads",
                    outcome_name=api_away_team or "Away",
                    outcome_side="away",
                    outcome_team=away_team,
                    outcome_price=away_spread_price,
                    outcome_point=away_spread_point,
                )
            )

        over_total_price = _pick_total_price(entry, "over")
        over_total_point = _pick_total_point(entry, "over")
        under_total_price = _pick_total_price(entry, "under")
        under_total_point = _pick_total_point(entry, "under")
        if over_total_price is not None or over_total_point is not None:
            rows.append(
                _row_payload(
                    league=league,
                    sport_key=sport_key,
                    event_id=event_id,
                    commence_time_utc=commence_time_utc,
                    api_home_team=api_home_team,
                    api_away_team=api_away_team,
                    home_team=home_team,
                    away_team=away_team,
                    bookmaker_key=bookmaker_key,
                    bookmaker_title=bookmaker_title,
                    bookmaker_last_update_utc=bookmaker_last_update_utc,
                    market_key="totals",
                    outcome_name="Over",
                    outcome_side="over",
                    outcome_team=None,
                    outcome_price=over_total_price,
                    outcome_point=over_total_point,
                )
            )
        if under_total_price is not None or under_total_point is not None:
            rows.append(
                _row_payload(
                    league=league,
                    sport_key=sport_key,
                    event_id=event_id,
                    commence_time_utc=commence_time_utc,
                    api_home_team=api_home_team,
                    api_away_team=api_away_team,
                    home_team=home_team,
                    away_team=away_team,
                    bookmaker_key=bookmaker_key,
                    bookmaker_title=bookmaker_title,
                    bookmaker_last_update_utc=bookmaker_last_update_utc,
                    market_key="totals",
                    outcome_name="Under",
                    outcome_side="under",
                    outcome_team=None,
                    outcome_price=under_total_price,
                    outcome_point=under_total_point,
                )
            )

    return rows


def _fetch_public_odds_for_date_keys(
    client: HttpClient,
    *,
    league: str,
    sport_key: str,
    source: str,
    date_keys: list[str],
    teams_df: pd.DataFrame | None = None,
) -> SourceFetchResult:
    as_of_utc = utc_now_iso()
    normalized_league = str(league or "").upper()
    endpoints = ESPN_ENDPOINTS[normalized_league]
    normalized_date_keys = [str(date_key).strip() for date_key in date_keys if str(date_key).strip()]
    if not normalized_date_keys:
        raise ValueError("Historical/public odds fetch requires at least one date key")

    events_by_id: dict[str, dict[str, Any]] = {}
    scoreboard_raw_paths: list[str] = []
    summary_raw_paths: list[str] = []
    scoreboard_failures: list[str] = []
    summary_failures: list[str] = []
    scoreboard_cache_hits = 0
    summary_cache_hits = 0

    for date_key in normalized_date_keys:
        try:
            payload, raw_path, _, from_cache = client.get_json_with_headers(
                source=f"{source}_scoreboard_{date_key}",
                url=endpoints["scoreboard"],
                params={"dates": date_key},
                key="scoreboard",
            )
        except Exception as exc:
            scoreboard_failures.append(f"{date_key}: {exc}")
            continue

        if raw_path:
            scoreboard_raw_paths.append(raw_path)
        if from_cache:
            scoreboard_cache_hits += 1

        for event in (payload or {}).get("events") or []:
            event_id = str(event.get("id") or "").strip()
            if event_id:
                events_by_id[event_id] = event

    team_name_map = _build_team_name_map(teams_df)
    alias_map = _team_aliases_for_league(normalized_league)
    rows: list[dict[str, Any]] = []
    event_ids_with_pickcenter: list[str] = []

    for event_id in sorted(events_by_id):
        try:
            summary_payload, raw_path, _, from_cache = client.get_json_with_headers(
                source=f"{source}_summary_{event_id}",
                url=endpoints["summary"],
                params={"event": event_id},
                key="summary",
            )
        except Exception as exc:
            summary_failures.append(f"{event_id}: {exc}")
            continue

        if raw_path:
            summary_raw_paths.append(raw_path)
        if from_cache:
            summary_cache_hits += 1

        event_rows = _flatten_pickcenter_summary(
            summary_payload=summary_payload or {},
            scoreboard_event=events_by_id[event_id],
            league=normalized_league,
            sport_key=sport_key,
            team_name_map=team_name_map,
            alias_map=alias_map,
            as_of_utc=as_of_utc,
        )
        if event_rows:
            event_ids_with_pickcenter.append(event_id)
            rows.extend(event_rows)

    df = pd.DataFrame(rows, columns=ODDS_COLUMNS)
    manifest_payload = {
        "provider": "espn_pickcenter",
        "league": normalized_league,
        "fetched_at_utc": as_of_utc,
        "date_keys": normalized_date_keys,
        "event_ids_seen": sorted(events_by_id),
        "event_ids_with_pickcenter": event_ids_with_pickcenter,
        "scoreboard_raw_paths": scoreboard_raw_paths,
        "summary_raw_paths": summary_raw_paths,
        "scoreboard_failures": scoreboard_failures,
        "summary_failures": summary_failures,
    }
    manifest_raw_path = client.save_raw(source, manifest_payload, key=as_of_utc.replace(":", "-"))

    total_requests = len(scoreboard_raw_paths) + len(summary_raw_paths) + len(scoreboard_failures) + len(summary_failures)
    total_cache_hits = scoreboard_cache_hits + summary_cache_hits
    metadata = {
        "fetched_at_utc": as_of_utc,
        "league": normalized_league,
        "sport_key": sport_key,
        "provider": "espn_pickcenter",
        "regions": "us",
        "markets": DEFAULT_MARKETS,
        "odds_format": DEFAULT_ODDS_FORMAT,
        "date_format": DEFAULT_DATE_FORMAT,
        "scoreboard_endpoint": endpoints["scoreboard"],
        "summary_endpoint": endpoints["summary"],
        "date_keys": normalized_date_keys,
        "scoreboard_request_count": len(scoreboard_raw_paths) + len(scoreboard_failures),
        "summary_request_count": len(summary_raw_paths) + len(summary_failures),
        "scoreboard_cache_hits": scoreboard_cache_hits,
        "summary_cache_hits": summary_cache_hits,
        "from_cache": int(total_requests > 0 and total_cache_hits == total_requests),
        "n_events_seen": int(len(events_by_id)),
        "n_events": int(df["odds_event_id"].nunique()) if not df.empty else 0,
        "n_rows": int(len(df)),
        "n_unique_books": int(df["bookmaker_key"].nunique()) if not df.empty else 0,
        "n_events_with_team_mapping": int(df["odds_event_id"][df["home_team"].notna() & df["away_team"].notna()].nunique())
        if not df.empty
        else 0,
        "scoreboard_failures": scoreboard_failures,
        "summary_failures": summary_failures,
        "raw_paths": {
            "manifest": manifest_raw_path,
            "scoreboards": scoreboard_raw_paths,
            "summaries": summary_raw_paths,
        },
    }
    snapshot_id = client.snapshot_id(source, metadata)

    return SourceFetchResult(
        source=source,
        snapshot_id=snapshot_id,
        extracted_at_utc=as_of_utc,
        raw_path=manifest_raw_path,
        metadata=metadata,
        dataframe=df,
    )


def fetch_public_odds(
    client: HttpClient,
    *,
    league: str,
    sport_key: str,
    source: str,
    teams_df: pd.DataFrame | None = None,
    upcoming_days: int | None = None,
) -> SourceFetchResult:
    lookahead_days = 14 if upcoming_days is None else max(0, int(upcoming_days))
    return _fetch_public_odds_for_date_keys(
        client,
        league=league,
        sport_key=sport_key,
        source=source,
        date_keys=_date_keys_for_window(lookahead_days),
        teams_df=teams_df,
    )


def fetch_public_odds_for_date_range(
    client: HttpClient,
    *,
    league: str,
    sport_key: str,
    source: str,
    start_date: date | datetime | str,
    end_date: date | datetime | str,
    teams_df: pd.DataFrame | None = None,
) -> SourceFetchResult:
    return _fetch_public_odds_for_date_keys(
        client,
        league=league,
        sport_key=sport_key,
        source=source,
        date_keys=_date_keys_for_range(start_date, end_date),
        teams_df=teams_df,
    )


def write_historical_odds_bundle(
    client: HttpClient,
    *,
    league: str,
    sport_key: str,
    source: str,
    output_dir: str | Path,
    start_date: date | datetime | str,
    end_date: date | datetime | str,
    games_path: str | Path,
    teams_df: pd.DataFrame | None = None,
    games_source: str | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    odds_result = fetch_public_odds_for_date_range(
        client,
        league=league,
        sport_key=sport_key,
        source=source,
        start_date=start_date,
        end_date=end_date,
        teams_df=teams_df,
    )

    requested_date_keys = odds_result.metadata.get("date_keys") or []
    first_date_key = requested_date_keys[0] if requested_date_keys else _coerce_date(start_date).strftime("%Y%m%d")
    last_date_key = requested_date_keys[-1] if requested_date_keys else _coerce_date(end_date).strftime("%Y%m%d")
    odds_filename = f"odds_{first_date_key}_{last_date_key}.csv"
    odds_path = output_root / odds_filename
    odds_result.dataframe.to_csv(odds_path, index=False)

    games_src_path = Path(games_path).resolve()
    copied_games_path = (output_root / games_src_path.name).resolve()
    if games_src_path != copied_games_path:
        shutil.copy2(games_src_path, copied_games_path)

    commence_values = odds_result.dataframe.get("commence_time_utc")
    if commence_values is not None and not odds_result.dataframe.empty:
        coverage_start = str(commence_values.dropna().astype(str).min()) if commence_values.notna().any() else None
        coverage_end = str(commence_values.dropna().astype(str).max()) if commence_values.notna().any() else None
    else:
        coverage_start = None
        coverage_end = None

    manifest_payload = {
        "league": str(league or "").upper(),
        "games": [
            {
                "path": copied_games_path.name,
                "source": games_source or f"{str(league or '').lower()}_historical_games",
                "extracted_at_utc": odds_result.extracted_at_utc,
                "metadata": {
                    "import_mode": "historical_bundle",
                    "copied_from": str(games_src_path),
                },
            }
        ],
        "odds_snapshots": [
            {
                "path": odds_path.name,
                "source": source,
                "snapshot_id": odds_result.snapshot_id,
                "as_of_utc": odds_result.extracted_at_utc,
                "metadata": {
                    "import_mode": "historical_bundle",
                    "coverage_start_utc": coverage_start,
                    "coverage_end_utc": coverage_end,
                }
                | dict(odds_result.metadata),
            }
        ],
        "bundle_metadata": {
            "requested_start_date": str(_coerce_date(start_date)),
            "requested_end_date": str(_coerce_date(end_date)),
            "requested_date_keys": requested_date_keys,
            "coverage_start_utc": coverage_start,
            "coverage_end_utc": coverage_end,
            "odds_rows": int(len(odds_result.dataframe)),
            "odds_events": int(odds_result.metadata.get("n_events") or 0),
            "games_path": copied_games_path.name,
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2))

    return {
        "manifest_path": str(manifest_path),
        "odds_path": str(odds_path),
        "games_path": str(copied_games_path),
        "odds_rows": int(len(odds_result.dataframe)),
        "odds_events": int(odds_result.metadata.get("n_events") or 0),
        "coverage_start_utc": coverage_start,
        "coverage_end_utc": coverage_end,
        "metadata": odds_result.metadata,
        "snapshot_id": odds_result.snapshot_id,
    }
