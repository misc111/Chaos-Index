from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.common.config import AppConfig
from src.common.logging import get_logger
from src.common.research import resolve_research_paths
from src.data_sources.base import HttpClient
from src.data_sources.odds_api import write_historical_odds_bundle
from src.league_registry import canonicalize_league, get_league_adapter
from src.services.ingest import client_from_config
from src.storage.db import Database

logger = get_logger(__name__)

CENTRAL_TZ = ZoneInfo("America/Chicago")
SPORT_KEY_BY_LEAGUE = {
    "NBA": "basketball_nba",
    "NCAAM": "basketball_ncaab",
    "NHL": "icehockey_nhl",
}


@dataclass(frozen=True)
class HistoricalOddsCacheBackfillResult:
    league: str
    manifest_path: Path
    chunk_count: int
    fetched_chunks: int
    skipped_chunks: int
    start_date: str
    end_date: str


def _coerce_date(value: date | datetime | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    raw = str(value or "").strip()
    if not raw:
        return None
    if len(raw) == 8 and raw.isdigit():
        return datetime.strptime(raw, "%Y%m%d").date()
    return date.fromisoformat(raw)


def _selected_seasons(db: Database, history_seasons: int) -> list[int]:
    rows = db.query("SELECT DISTINCT season FROM games WHERE season IS NOT NULL ORDER BY season DESC")
    seasons = [int(row["season"]) for row in rows if row.get("season") is not None]
    return seasons[: max(1, int(history_seasons))]


def _load_games_frame(db: Database, *, seasons: list[int]) -> pd.DataFrame:
    if not seasons:
        return pd.DataFrame()
    placeholders = ",".join("?" for _ in seasons)
    rows = db.query(
        f"""
        SELECT game_id, season, game_date_utc, start_time_utc, game_state,
               home_team, away_team, home_team_id, away_team_id, venue,
               is_neutral_site, home_score, away_score, went_ot, went_so,
               home_win, status_final, as_of_utc
        FROM games
        WHERE season IN ({placeholders})
        ORDER BY start_time_utc ASC, game_id ASC
        """,
        tuple(seasons),
    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["start_time_utc"] = pd.to_datetime(frame["start_time_utc"], errors="coerce", utc=True)
    frame = frame[frame["start_time_utc"].notna()].copy()
    frame["central_date"] = frame["start_time_utc"].dt.tz_convert(CENTRAL_TZ).dt.date
    return frame.reset_index(drop=True)


def _infer_date_bounds(
    games_df: pd.DataFrame,
    *,
    start_date: date | datetime | str | None,
    end_date: date | datetime | str | None,
) -> tuple[date, date]:
    if games_df.empty:
        raise RuntimeError("Historical odds backfill requires games in the local DB")
    inferred_start = min(games_df["central_date"].tolist())
    inferred_end = max(games_df["central_date"].tolist())
    resolved_start = _coerce_date(start_date) or inferred_start
    resolved_end = _coerce_date(end_date) or inferred_end
    if resolved_end < resolved_start:
        raise ValueError(f"Historical odds backfill date range must be ascending: start={resolved_start} end={resolved_end}")
    return resolved_start, resolved_end


def _chunk_ranges(start_date: date, end_date: date, *, chunk_days: int) -> list[tuple[date, date]]:
    width = max(1, int(chunk_days))
    ranges: list[tuple[date, date]] = []
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(cursor + timedelta(days=width - 1), end_date)
        ranges.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return ranges


def _chunk_dir_name(start_date: date, end_date: date) -> str:
    return f"backfill_{start_date.isoformat()}_{end_date.isoformat()}"


def _export_games_chunk(chunk_games: pd.DataFrame, *, chunk_dir: Path, start_date: date, end_date: date) -> Path:
    ordered = [
        "game_id",
        "season",
        "game_date_utc",
        "start_time_utc",
        "game_state",
        "home_team",
        "away_team",
        "home_team_id",
        "away_team_id",
        "venue",
        "is_neutral_site",
        "home_score",
        "away_score",
        "went_ot",
        "went_so",
        "home_win",
        "status_final",
        "as_of_utc",
    ]
    export = chunk_games[ordered].copy()
    export["start_time_utc"] = export["start_time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    path = chunk_dir / f"games_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    export.to_csv(path, index=False)
    return path


def _prefixed_entries(entries: list[dict], *, source_dir: Path, chunk_manifest: Path) -> list[dict]:
    prefix = chunk_manifest.parent.relative_to(source_dir)
    out: list[dict] = []
    for entry in entries:
        item = dict(entry)
        item["path"] = str(prefix / str(item["path"]))
        out.append(item)
    return out


def _rebuild_top_level_manifest(*, source_dir: Path, league: str, chunk_manifests: list[Path]) -> Path:
    payload: dict[str, object] = {
        "league": league,
        "games": [],
        "odds_snapshots": [],
        "bundle_manifests": [],
    }
    games_entries: list[dict] = []
    odds_entries: list[dict] = []
    manifest_entries: list[str] = []
    for chunk_manifest in sorted(chunk_manifests):
        raw = json.loads(chunk_manifest.read_text())
        games_entries.extend(_prefixed_entries(list(raw.get("games") or []), source_dir=source_dir, chunk_manifest=chunk_manifest))
        odds_entries.extend(
            _prefixed_entries(list(raw.get("odds_snapshots") or []), source_dir=source_dir, chunk_manifest=chunk_manifest)
        )
        manifest_entries.append(str(chunk_manifest.relative_to(source_dir)))

    payload["games"] = sorted(games_entries, key=lambda row: str(row.get("path") or ""))
    payload["odds_snapshots"] = sorted(odds_entries, key=lambda row: str(row.get("path") or ""))
    payload["bundle_manifests"] = manifest_entries

    manifest_path = source_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def backfill_historical_odds_cache(
    cfg: AppConfig,
    *,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    history_seasons: int | None = None,
    chunk_days: int = 30,
    client: HttpClient | None = None,
    teams_df: pd.DataFrame | None = None,
) -> HistoricalOddsCacheBackfillResult:
    league = canonicalize_league(cfg.data.league)
    sport_key = SPORT_KEY_BY_LEAGUE[league]

    db = Database(cfg.paths.db_path)
    db.init_schema()
    seasons = _selected_seasons(db, max(1, int(history_seasons or cfg.research.history_seasons)))
    games_df = _load_games_frame(db, seasons=seasons)
    resolved_start, resolved_end = _infer_date_bounds(games_df, start_date=start_date, end_date=end_date)
    selected_games = games_df[
        (games_df["central_date"] >= resolved_start) & (games_df["central_date"] <= resolved_end)
    ].copy()
    if selected_games.empty:
        raise RuntimeError(
            "Historical odds backfill did not find any games in the requested window. "
            f"start={resolved_start} end={resolved_end}"
        )

    paths = resolve_research_paths(cfg)
    paths.source_dir.mkdir(parents=True, exist_ok=True)

    resolved_client = client or client_from_config(cfg)
    resolved_teams = teams_df
    if resolved_teams is None:
        resolved_teams = get_league_adapter(league).fetch_teams(resolved_client).dataframe

    chunk_manifests: list[Path] = []
    fetched_chunks = 0
    skipped_chunks = 0

    for chunk_start, chunk_end in _chunk_ranges(resolved_start, resolved_end, chunk_days=chunk_days):
        chunk_games = selected_games[
            (selected_games["central_date"] >= chunk_start) & (selected_games["central_date"] <= chunk_end)
        ].copy()
        if chunk_games.empty:
            continue

        chunk_dir = paths.source_dir / _chunk_dir_name(chunk_start, chunk_end)
        chunk_manifest = chunk_dir / "manifest.json"
        if chunk_manifest.exists():
            chunk_manifests.append(chunk_manifest)
            skipped_chunks += 1
            continue

        chunk_dir.mkdir(parents=True, exist_ok=True)
        games_path = _export_games_chunk(chunk_games, chunk_dir=chunk_dir, start_date=chunk_start, end_date=chunk_end)
        write_historical_odds_bundle(
            resolved_client,
            league=league,
            sport_key=sport_key,
            source=f"{league.lower()}_historical_odds_backfill",
            output_dir=chunk_dir,
            start_date=chunk_start,
            end_date=chunk_end,
            games_path=games_path,
            teams_df=resolved_teams,
            games_source=f"{league.lower()}_historical_games_backfill",
        )
        chunk_manifests.append(chunk_manifest)
        fetched_chunks += 1

    if not chunk_manifests:
        raise RuntimeError(
            "Historical odds backfill did not produce any chunk manifests. "
            f"start={resolved_start} end={resolved_end}"
        )

    manifest_path = _rebuild_top_level_manifest(source_dir=paths.source_dir, league=league, chunk_manifests=chunk_manifests)
    logger.info(
        "Historical odds cache backfill complete | league=%s manifest=%s chunks=%d fetched=%d skipped=%d start=%s end=%s",
        league,
        manifest_path,
        len(chunk_manifests),
        fetched_chunks,
        skipped_chunks,
        resolved_start,
        resolved_end,
    )
    return HistoricalOddsCacheBackfillResult(
        league=league,
        manifest_path=manifest_path,
        chunk_count=len(chunk_manifests),
        fetched_chunks=fetched_chunks,
        skipped_chunks=skipped_chunks,
        start_date=resolved_start.isoformat(),
        end_date=resolved_end.isoformat(),
    )
