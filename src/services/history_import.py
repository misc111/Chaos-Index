from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.common.config import AppConfig
from src.common.logging import get_logger
from src.common.research import resolve_research_paths
from src.common.time import utc_now_iso
from src.common.utils import stable_hash
from src.data_sources.base import SourceFetchResult
from src.league_registry import canonicalize_league
from src.services.ingest import insert_odds_snapshot_and_lines, insert_snapshot, upsert_games, upsert_results
from src.storage.db import Database

logger = get_logger(__name__)

CENTRAL_TZ = ZoneInfo("America/Chicago")
REQUIRED_GAME_COLUMNS = {
    "game_id",
    "season",
    "game_date_utc",
    "start_time_utc",
    "home_team",
    "away_team",
}
REQUIRED_ODDS_COLUMNS = {
    "odds_event_id",
    "commence_time_utc",
    "home_team",
    "away_team",
    "bookmaker_key",
    "bookmaker_title",
    "market_key",
    "outcome_name",
    "outcome_side",
    "outcome_price",
}
GAME_COLUMN_ALIASES = {
    "date_utc": "game_date_utc",
    "start_time": "start_time_utc",
    "home_abbrev": "home_team",
    "away_abbrev": "away_team",
    "final_utc": "as_of_utc",
}
ODDS_COLUMN_ALIASES = {
    "event_id": "odds_event_id",
    "event_start_utc": "commence_time_utc",
    "api_home_abbrev": "api_home_team",
    "api_away_abbrev": "api_away_team",
    "home_abbrev": "home_team",
    "away_abbrev": "away_team",
    "book_key": "bookmaker_key",
    "book_title": "bookmaker_title",
    "book_last_update_utc": "bookmaker_last_update_utc",
    "price": "outcome_price",
    "american_odds": "outcome_price",
}


@dataclass(frozen=True)
class HistoricalEntry:
    path: str
    source: str
    snapshot_id: str | None
    extracted_at_utc: str | None
    season: int | None
    metadata: dict[str, Any]


def _manifest_entries(manifest: dict[str, Any], key: str, *, default_source: str) -> list[HistoricalEntry]:
    raw_entries = manifest.get(key, [])
    if not isinstance(raw_entries, list):
        return []
    out: list[HistoricalEntry] = []
    for payload in raw_entries:
        if not isinstance(payload, dict):
            continue
        path = str(payload.get("path") or "").strip()
        if not path:
            continue
        season_raw = payload.get("season")
        season = int(season_raw) if season_raw is not None and str(season_raw).strip() else None
        out.append(
            HistoricalEntry(
                path=path,
                source=str(payload.get("source") or default_source),
                snapshot_id=str(payload.get("snapshot_id") or "").strip() or None,
                extracted_at_utc=str(payload.get("extracted_at_utc") or payload.get("as_of_utc") or "").strip() or None,
                season=season,
                metadata=dict(payload.get("metadata") or {}),
            )
        )
    return out


def _load_tabular_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return pd.DataFrame(rows)
    if suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            if isinstance(payload.get("rows"), list):
                return pd.DataFrame(payload["rows"])
            if isinstance(payload.get("data"), list):
                return pd.DataFrame(payload["data"])
        raise ValueError(f"Unsupported JSON payload shape for historical import file: {path}")
    raise ValueError(f"Unsupported historical import file type: {path.suffix}")


def _normalize_game_frame(df: pd.DataFrame, *, extracted_at_utc: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=sorted(REQUIRED_GAME_COLUMNS | {"game_state", "status_final", "home_score", "away_score", "home_win", "as_of_utc"}))
    work = df.rename(columns={src: dest for src, dest in GAME_COLUMN_ALIASES.items() if src in df.columns}).copy()
    missing = REQUIRED_GAME_COLUMNS - set(work.columns)
    if missing:
        raise ValueError(f"Historical games payload missing required columns: {sorted(missing)}")

    if "as_of_utc" not in work.columns:
        work["as_of_utc"] = extracted_at_utc
    if "status_final" not in work.columns:
        work["status_final"] = pd.to_numeric(work.get("home_win"), errors="coerce").notna().astype(int)
    if "game_state" not in work.columns:
        work["game_state"] = work["status_final"].map(lambda value: "final" if int(value) == 1 else "scheduled")
    if "venue" not in work.columns:
        work["venue"] = None
    if "is_neutral_site" not in work.columns:
        work["is_neutral_site"] = 0
    if "went_ot" not in work.columns:
        work["went_ot"] = 0
    if "went_so" not in work.columns:
        work["went_so"] = 0
    if "home_team_id" not in work.columns:
        work["home_team_id"] = None
    if "away_team_id" not in work.columns:
        work["away_team_id"] = None

    work["game_id"] = pd.to_numeric(work["game_id"], errors="raise").astype(int)
    work["season"] = pd.to_numeric(work["season"], errors="raise").astype(int)
    work["status_final"] = pd.to_numeric(work["status_final"], errors="coerce").fillna(0).astype(int)
    for column in ["home_score", "away_score", "home_win", "home_team_id", "away_team_id"]:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
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
    return work[ordered].copy()


def _central_date_key(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = pd.Timestamp(raw, tz="UTC") if "T" not in raw else pd.Timestamp(raw)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed.tz_convert(CENTRAL_TZ).date().isoformat()


def _american_to_implied_probability(price: Any) -> float | None:
    try:
        value = float(price)
    except Exception:
        return None
    if value == 0:
        return None
    if value > 0:
        return 100.0 / (value + 100.0)
    return abs(value) / (abs(value) + 100.0)


def _normalize_odds_frame(df: pd.DataFrame, *, league: str, as_of_utc: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=sorted(REQUIRED_ODDS_COLUMNS | {"league", "sport_key", "commence_date_central", "api_home_team", "api_away_team", "bookmaker_last_update_utc", "outcome_team", "outcome_point", "implied_probability"}))
    work = df.rename(columns={src: dest for src, dest in ODDS_COLUMN_ALIASES.items() if src in df.columns}).copy()
    missing = REQUIRED_ODDS_COLUMNS - set(work.columns)
    if missing:
        raise ValueError(f"Historical odds payload missing required columns: {sorted(missing)}")

    work["league"] = league
    if "sport_key" not in work.columns:
        work["sport_key"] = league.lower()
    if "commence_date_central" not in work.columns:
        work["commence_date_central"] = work["commence_time_utc"].map(_central_date_key)
    if "api_home_team" not in work.columns:
        work["api_home_team"] = work["home_team"]
    if "api_away_team" not in work.columns:
        work["api_away_team"] = work["away_team"]
    if "bookmaker_last_update_utc" not in work.columns:
        work["bookmaker_last_update_utc"] = as_of_utc
    if "outcome_team" not in work.columns:
        work["outcome_team"] = work.apply(
            lambda row: row["home_team"] if str(row["outcome_side"]).strip().lower() == "home" else row["away_team"] if str(row["outcome_side"]).strip().lower() == "away" else None,
            axis=1,
        )
    if "outcome_point" not in work.columns:
        work["outcome_point"] = None
    if "implied_probability" not in work.columns:
        work["implied_probability"] = work["outcome_price"].map(_american_to_implied_probability)

    work["outcome_price"] = pd.to_numeric(work["outcome_price"], errors="raise")
    if "outcome_point" in work.columns:
        work["outcome_point"] = pd.to_numeric(work["outcome_point"], errors="coerce")
    if "implied_probability" in work.columns:
        work["implied_probability"] = pd.to_numeric(work["implied_probability"], errors="coerce")

    ordered = [
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
    return work[ordered].copy()


def _entry_path(base_dir: Path, manifest_path: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return candidate
    manifest_candidate = (manifest_path.parent / candidate).resolve()
    if manifest_candidate.exists():
        return manifest_candidate
    return (base_dir / candidate).resolve()


def _season_values_from_games(df: pd.DataFrame) -> list[int]:
    if df.empty or "season" not in df.columns:
        return []
    values = pd.to_numeric(df["season"], errors="coerce").dropna().astype(int).tolist()
    return sorted(set(values))


def _filter_games_to_recent_seasons(df: pd.DataFrame, selected_seasons: set[int]) -> pd.DataFrame:
    if df.empty or not selected_seasons:
        return df
    return df[pd.to_numeric(df["season"], errors="coerce").isin(selected_seasons)].copy()


def _filter_odds_to_recent_seasons(df: pd.DataFrame, selected_seasons: set[int]) -> pd.DataFrame:
    if df.empty or not selected_seasons:
        return df
    commence_ts = pd.to_datetime(df["commence_time_utc"], errors="coerce", utc=True)
    if commence_ts.isna().all():
        return df
    season_values = commence_ts.dt.year + (commence_ts.dt.month >= 7).astype(int)
    season_codes = (season_values - 1) * 10000 + season_values
    return df[season_codes.isin(selected_seasons)].copy()


def import_historical_data(
    cfg: AppConfig,
    *,
    history_seasons: int | None = None,
    source_manifest: str | None = None,
) -> None:
    league = canonicalize_league(cfg.data.league)
    if league != "NBA":
        logger.info("Historical import currently uses NBA as the deep-research pilot; initializing shared storage for %s only", league)

    paths = resolve_research_paths(cfg)
    manifest_path = Path(source_manifest).resolve() if source_manifest else paths.source_manifest.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Historical import manifest not found at {manifest_path}. "
            "Populate a manifest-backed historical source directory before running import-history."
        )

    raw_manifest = json.loads(manifest_path.read_text())
    manifest_league = canonicalize_league(raw_manifest.get("league", league))
    if manifest_league != league:
        raise ValueError(f"Historical import manifest league mismatch: manifest={manifest_league} config={league}")

    season_limit = max(1, int(history_seasons or cfg.research.history_seasons))
    games_entries = _manifest_entries(raw_manifest, "games", default_source=f"{league.lower()}_historical_games")
    odds_entries = _manifest_entries(raw_manifest, "odds_snapshots", default_source=f"{league.lower()}_historical_odds")
    if not games_entries:
        raise ValueError("Historical import manifest did not include any 'games' entries")
    if not odds_entries:
        raise ValueError("Historical import manifest did not include any 'odds_snapshots' entries")

    loaded_games: list[tuple[HistoricalEntry, pd.DataFrame]] = []
    season_values: set[int] = set()
    for entry in games_entries:
        path = _entry_path(paths.source_dir, manifest_path, entry.path)
        frame = _normalize_game_frame(_load_tabular_file(path), extracted_at_utc=entry.extracted_at_utc or utc_now_iso())
        loaded_games.append((entry, frame))
        season_values.update(_season_values_from_games(frame))
        if entry.season is not None:
            season_values.add(int(entry.season))

    selected_seasons = set(sorted(season_values, reverse=True)[:season_limit])
    if not selected_seasons:
        raise ValueError("Historical import could not infer any seasons from the supplied games files")

    db = Database(cfg.paths.db_path)
    db.init_schema()

    imported_game_snapshots = 0
    imported_games = 0
    for entry, frame in loaded_games:
        filtered = _filter_games_to_recent_seasons(frame, selected_seasons)
        if filtered.empty:
            continue
        snapshot_id = entry.snapshot_id or f"{entry.source}_{stable_hash({'path': str(entry.path), 'seasons': sorted(selected_seasons)})}"
        extracted_at_utc = entry.extracted_at_utc or utc_now_iso()
        fetch_result = SourceFetchResult(
            source=entry.source,
            snapshot_id=snapshot_id,
            extracted_at_utc=extracted_at_utc,
            raw_path=str(_entry_path(paths.source_dir, manifest_path, entry.path)),
            metadata={"import_mode": "historical_manifest", "selected_seasons": sorted(selected_seasons)} | dict(entry.metadata),
            dataframe=filtered,
        )
        insert_snapshot(db, fetch_result)
        upsert_games(db, filtered)
        upsert_results(
            db,
            filtered[
                [
                    "game_id",
                    "season",
                    "game_date_utc",
                    "start_time_utc",
                    "home_team",
                    "away_team",
                    "home_score",
                    "away_score",
                    "home_win",
                    "as_of_utc",
                ]
            ].rename(columns={"start_time_utc": "final_utc", "as_of_utc": "ingested_at_utc"}),
        )
        imported_game_snapshots += 1
        imported_games += len(filtered)

    imported_odds_snapshots = 0
    imported_odds_rows = 0
    for entry in odds_entries:
        path = _entry_path(paths.source_dir, manifest_path, entry.path)
        extracted_at_utc = entry.extracted_at_utc or utc_now_iso()
        frame = _normalize_odds_frame(_load_tabular_file(path), league=league, as_of_utc=extracted_at_utc)
        filtered = _filter_odds_to_recent_seasons(frame, selected_seasons)
        if filtered.empty:
            continue
        snapshot_id = entry.snapshot_id or f"{entry.source}_{stable_hash({'path': str(entry.path), 'as_of_utc': extracted_at_utc, 'rows': len(filtered)})}"
        fetch_result = SourceFetchResult(
            source=entry.source,
            snapshot_id=snapshot_id,
            extracted_at_utc=extracted_at_utc,
            raw_path=str(path),
            metadata={"import_mode": "historical_manifest", "selected_seasons": sorted(selected_seasons)} | dict(entry.metadata),
            dataframe=filtered,
        )
        insert_snapshot(db, fetch_result)
        insert_odds_snapshot_and_lines(db, league=league, odds_res=fetch_result)
        imported_odds_snapshots += 1
        imported_odds_rows += len(filtered)

    logger.info(
        "Historical import complete | league=%s seasons=%s game_snapshots=%d games=%d odds_snapshots=%d odds_rows=%d manifest=%s",
        league,
        sorted(selected_seasons),
        imported_game_snapshots,
        imported_games,
        imported_odds_snapshots,
        imported_odds_rows,
        manifest_path,
    )
