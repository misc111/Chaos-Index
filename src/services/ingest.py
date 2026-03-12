"""Shared ingest service for league-agnostic pipeline orchestration.

This service owns DB writes and interim artifact persistence. League-specific
behavior is injected through the typed adapter in :mod:`src.league_registry`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import AppConfig
from src.common.logging import get_logger
from src.common.time import utc_now_iso
from src.common.utils import ensure_dir, to_json
from src.data_sources.base import HttpClient, SourceFetchResult
from src.features.build_features import build_features_from_interim
from src.league_registry import LeagueAdapter, canonicalize_league, get_league_adapter
from src.storage.db import Database

logger = get_logger(__name__)


INTERIM_FILES = {
    "games": "games.parquet",
    "schedule": "schedule.parquet",
    "teams": "teams.parquet",
    "players": "players.parquet",
    "goalies": "goalies.parquet",
    "injuries": "injuries.parquet",
    "odds": "odds.parquet",
    "xg": "xg.parquet",
}


def client_from_config(cfg: AppConfig) -> HttpClient:
    return HttpClient(
        raw_dir=cfg.paths.raw_dir,
        timeout_seconds=cfg.data.timeout_seconds,
        max_retries=cfg.data.max_retries,
        backoff_seconds=cfg.data.backoff_seconds,
        offline_mode=cfg.data.offline_mode,
    )


def initialize_database(cfg: AppConfig) -> None:
    db = Database(cfg.paths.db_path)
    db.init_schema()
    logger.info("Initialized DB schema at %s", cfg.paths.db_path)


def save_interim(df: pd.DataFrame, interim_dir: str, name: str) -> str:
    path = Path(interim_dir) / INTERIM_FILES[name]
    ensure_dir(path.parent)
    try:
        df.to_parquet(path, index=False)
        return str(path)
    except Exception:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return str(csv_path)


def insert_snapshot(db: Database, res: SourceFetchResult) -> None:
    db.execute(
        """
        INSERT OR REPLACE INTO raw_snapshots(snapshot_id, source, extracted_at_utc, raw_path, metadata_json, freshness_utc, row_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            res.snapshot_id,
            res.source,
            res.extracted_at_utc,
            res.raw_path,
            to_json(res.metadata),
            res.extracted_at_utc,
            int(len(res.dataframe)),
        ),
    )


def upsert_games(db: Database, games_df: pd.DataFrame) -> None:
    if games_df.empty:
        return
    rows = [
        (
            int(r.game_id),
            int(r.season) if pd.notna(r.season) else None,
            r.game_date_utc,
            r.start_time_utc,
            r.game_state,
            r.home_team,
            r.away_team,
            int(r.home_team_id) if pd.notna(r.home_team_id) else None,
            int(r.away_team_id) if pd.notna(r.away_team_id) else None,
            r.venue,
            int(r.is_neutral_site) if pd.notna(r.is_neutral_site) else 0,
            int(r.home_score) if pd.notna(r.home_score) else None,
            int(r.away_score) if pd.notna(r.away_score) else None,
            int(r.went_ot) if pd.notna(r.went_ot) else 0,
            int(r.went_so) if pd.notna(r.went_so) else 0,
            int(r.home_win) if pd.notna(r.home_win) else None,
            int(r.status_final) if pd.notna(r.status_final) else 0,
            str(r.as_of_utc),
        )
        for r in games_df.itertuples(index=False)
    ]
    db.executemany(
        """
        INSERT OR REPLACE INTO games(
          game_id, season, game_date_utc, start_time_utc, game_state,
          home_team, away_team, home_team_id, away_team_id, venue,
          is_neutral_site, home_score, away_score, went_ot, went_so,
          home_win, status_final, as_of_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def upsert_results(db: Database, results_df: pd.DataFrame) -> None:
    if results_df.empty:
        return
    rows = [
        (
            int(r.game_id),
            int(r.season) if pd.notna(r.season) else None,
            r.game_date_utc,
            r.final_utc,
            r.home_team,
            r.away_team,
            int(r.home_score) if pd.notna(r.home_score) else None,
            int(r.away_score) if pd.notna(r.away_score) else None,
            int(r.home_win) if pd.notna(r.home_win) else None,
            r.ingested_at_utc,
        )
        for r in results_df.itertuples(index=False)
    ]
    db.executemany(
        """
        INSERT OR REPLACE INTO results(
          game_id, season, game_date_utc, final_utc, home_team, away_team,
          home_score, away_score, home_win, ingested_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def upsert_teams(
    db: Database,
    teams_df: pd.DataFrame,
    league: str,
    snapshot_id: str | None,
    as_of_utc: str,
) -> None:
    if teams_df.empty:
        return

    rows = []
    for r in teams_df.itertuples(index=False):
        team_abbrev = getattr(r, "team_abbrev", None)
        if team_abbrev is None or str(team_abbrev).strip() == "":
            continue
        rows.append(
            (
                league,
                str(team_abbrev),
                getattr(r, "team_name", None),
                getattr(r, "conference", None),
                getattr(r, "division", None),
                getattr(r, "as_of_date", None),
                as_of_utc,
                snapshot_id,
                to_json({}),
            )
        )

    db.executemany(
        """
        INSERT INTO teams(
          league, team_abbrev, team_name, conference, division,
          as_of_date, as_of_utc, snapshot_id, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def latest_teams_df(db: Database, league: str) -> pd.DataFrame:
    rows = db.query(
        """
        SELECT team_abbrev, team_name, conference, division, as_of_date
        FROM teams
        WHERE league = ?
          AND as_of_utc = (
            SELECT MAX(as_of_utc)
            FROM teams
            WHERE league = ?
          )
        ORDER BY team_abbrev ASC
        """,
        (league, league),
    )
    return pd.DataFrame(rows)


def parse_iso_or_none(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def map_odds_rows_to_games(db: Database, odds_df: pd.DataFrame) -> pd.DataFrame:
    if odds_df.empty:
        mapped = odds_df.copy()
        mapped["game_id"] = None
        return mapped

    required = {"odds_event_id", "home_team", "away_team"}
    if not required.issubset(set(odds_df.columns)):
        mapped = odds_df.copy()
        mapped["game_id"] = None
        return mapped

    games = pd.DataFrame(
        db.query(
            """
            SELECT game_id, game_date_utc, start_time_utc, home_team, away_team
            FROM games
            WHERE home_team IS NOT NULL
              AND away_team IS NOT NULL
            """
        )
    )

    if games.empty:
        mapped = odds_df.copy()
        mapped["game_id"] = None
        return mapped

    event_keys = (
        odds_df[["odds_event_id", "home_team", "away_team", "commence_date_central", "commence_time_utc"]]
        .drop_duplicates(subset=["odds_event_id"])
        .to_dict(orient="records")
    )

    event_to_game: dict[str, int | None] = {}
    for event in event_keys:
        event_id = str(event.get("odds_event_id") or "").strip()
        if not event_id:
            continue
        home_team = str(event.get("home_team") or "").strip()
        away_team = str(event.get("away_team") or "").strip()
        if not home_team or not away_team:
            event_to_game[event_id] = None
            continue

        candidates = games[(games["home_team"] == home_team) & (games["away_team"] == away_team)].copy()
        if candidates.empty:
            event_to_game[event_id] = None
            continue

        preferred_date = str(event.get("commence_date_central") or "").strip()
        if preferred_date:
            by_date = candidates[candidates["game_date_utc"].astype(str) == preferred_date]
            if len(by_date) == 1:
                event_to_game[event_id] = int(by_date.iloc[0]["game_id"])
                continue
            if len(by_date) > 1:
                candidates = by_date

        commence_dt = parse_iso_or_none(event.get("commence_time_utc"))
        if commence_dt is not None and not candidates.empty:
            candidates = candidates.copy()
            candidates["start_dt"] = candidates["start_time_utc"].apply(parse_iso_or_none)
            candidates = candidates[candidates["start_dt"].notna()]
            if not candidates.empty:
                candidates["abs_diff_seconds"] = candidates["start_dt"].apply(lambda dt: abs((dt - commence_dt).total_seconds()))
                winner = candidates.sort_values("abs_diff_seconds").iloc[0]
                if float(winner["abs_diff_seconds"]) <= 18 * 3600:
                    event_to_game[event_id] = int(winner["game_id"])
                    continue

        event_to_game[event_id] = int(candidates.iloc[0]["game_id"]) if len(candidates) == 1 else None

    mapped = odds_df.copy()
    mapped["game_id"] = mapped["odds_event_id"].map(event_to_game)
    return mapped


def insert_odds_snapshot_and_lines(
    db: Database,
    league: str,
    odds_res: SourceFetchResult,
) -> None:
    def nullable_text(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return text

    metadata = odds_res.metadata or {}
    rows_df = map_odds_rows_to_games(db, odds_res.dataframe)

    db.execute(
        """
        INSERT OR REPLACE INTO odds_snapshots(
          odds_snapshot_id, source, league, as_of_utc, raw_path,
          regions, markets, odds_format, date_format,
          event_count, row_count,
          requests_last, requests_used, requests_remaining,
          from_cache, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            odds_res.snapshot_id,
            odds_res.source,
            league,
            odds_res.extracted_at_utc,
            odds_res.raw_path,
            str(metadata.get("regions") or ""),
            str(metadata.get("markets") or ""),
            str(metadata.get("odds_format") or ""),
            str(metadata.get("date_format") or ""),
            int(metadata.get("n_events") or 0),
            int(len(rows_df)),
            int(metadata["requests_last"]) if metadata.get("requests_last") is not None else None,
            int(metadata["requests_used"]) if metadata.get("requests_used") is not None else None,
            int(metadata["requests_remaining"]) if metadata.get("requests_remaining") is not None else None,
            int(metadata.get("from_cache") or 0),
            to_json(metadata),
        ),
    )

    if rows_df.empty:
        return

    db.execute("DELETE FROM odds_market_lines WHERE odds_snapshot_id = ?", (odds_res.snapshot_id,))

    created_at_utc = utc_now_iso()
    insert_rows: list[tuple[Any, ...]] = []
    for row in rows_df.itertuples(index=False):
        game_id_raw = getattr(row, "game_id", None)
        game_id = int(game_id_raw) if pd.notna(game_id_raw) else None
        outcome_price = pd.to_numeric(getattr(row, "outcome_price", None), errors="coerce")
        outcome_point = pd.to_numeric(getattr(row, "outcome_point", None), errors="coerce")
        implied_probability = pd.to_numeric(getattr(row, "implied_probability", None), errors="coerce")

        insert_rows.append(
            (
                odds_res.snapshot_id,
                league,
                game_id,
                nullable_text(getattr(row, "sport_key", "")),
                nullable_text(getattr(row, "odds_event_id", "")),
                nullable_text(getattr(row, "commence_time_utc", "")),
                nullable_text(getattr(row, "commence_date_central", "")),
                nullable_text(getattr(row, "api_home_team", "")),
                nullable_text(getattr(row, "api_away_team", "")),
                nullable_text(getattr(row, "home_team", "")),
                nullable_text(getattr(row, "away_team", "")),
                nullable_text(getattr(row, "bookmaker_key", "")),
                nullable_text(getattr(row, "bookmaker_title", "")),
                nullable_text(getattr(row, "bookmaker_last_update_utc", "")),
                nullable_text(getattr(row, "market_key", "")),
                nullable_text(getattr(row, "outcome_name", "")),
                nullable_text(getattr(row, "outcome_side", "")),
                nullable_text(getattr(row, "outcome_team", "")),
                float(outcome_price) if pd.notna(outcome_price) else None,
                float(outcome_point) if pd.notna(outcome_point) else None,
                float(implied_probability) if pd.notna(implied_probability) else None,
                created_at_utc,
            )
        )

    db.executemany(
        """
        INSERT INTO odds_market_lines(
          odds_snapshot_id, league, game_id, sport_key, odds_event_id,
          commence_time_utc, commence_date_central, api_home_team, api_away_team,
          home_team, away_team, bookmaker_key, bookmaker_title, bookmaker_last_update_utc,
          market_key, outcome_name, outcome_side, outcome_team,
          outcome_price, outcome_point, implied_probability, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        insert_rows,
    )


def latest_snapshot_id(db: Database) -> str | None:
    rows = db.query("SELECT snapshot_id FROM raw_snapshots ORDER BY extracted_at_utc DESC LIMIT 1")
    return rows[0]["snapshot_id"] if rows else None


def _season_guess(games_df: pd.DataFrame) -> str:
    if not games_df.empty and "season" in games_df.columns and games_df["season"].notna().any():
        return str(int(games_df["season"].dropna().max()))
    now = datetime.now(timezone.utc)
    season_end_year = now.year + (1 if now.month >= 7 else 0)
    return f"{season_end_year - 1}{season_end_year}"


def _team_abbrevs(teams_df: pd.DataFrame) -> list[str]:
    return sorted(set(teams_df.get("team_abbrev", pd.Series(dtype=str)).dropna().astype(str).tolist()))


def fetch_data(cfg: AppConfig) -> None:
    db = Database(cfg.paths.db_path)
    db.init_schema()
    client = client_from_config(cfg)
    adapter = get_league_adapter(cfg.data.league)
    league = adapter.code

    end_date = datetime.now(timezone.utc) + timedelta(days=cfg.data.upcoming_days)
    start_date = datetime.now(timezone.utc) - timedelta(days=cfg.data.history_days)

    games_res = adapter.fetch_games(client, start_date=start_date, end_date=end_date)
    save_interim(games_res.dataframe, cfg.paths.interim_dir, "games")
    insert_snapshot(db, games_res)
    upsert_games(db, games_res.dataframe)

    schedule_res = adapter.fetch_upcoming_schedule(client, days_ahead=cfg.data.upcoming_days)
    save_interim(schedule_res.dataframe, cfg.paths.interim_dir, "schedule")
    insert_snapshot(db, schedule_res)

    teams_res = adapter.fetch_teams(client)
    save_interim(teams_res.dataframe, cfg.paths.interim_dir, "teams")
    insert_snapshot(db, teams_res)
    upsert_teams(
        db,
        teams_df=teams_res.dataframe,
        league=league,
        snapshot_id=teams_res.snapshot_id,
        as_of_utc=teams_res.extracted_at_utc,
    )

    team_abbrevs = _team_abbrevs(teams_res.dataframe)
    players_res = adapter.fetch_players(
        client,
        team_abbrevs=team_abbrevs,
        season=_season_guess(games_res.dataframe),
        games_df=games_res.dataframe,
    )
    save_interim(players_res.dataframe, cfg.paths.interim_dir, "players")
    insert_snapshot(db, players_res)

    final_ids = games_res.dataframe[games_res.dataframe["status_final"] == 1]["game_id"].astype(int).tolist()
    goalies_res = adapter.fetch_goalie_game_stats(client, game_ids=final_ids, max_games=350)
    save_interim(goalies_res.dataframe, cfg.paths.interim_dir, "goalies")
    insert_snapshot(db, goalies_res)

    injuries_res = adapter.fetch_injuries_proxy(client, teams=team_abbrevs)
    save_interim(injuries_res.dataframe, cfg.paths.interim_dir, "injuries")
    insert_snapshot(db, injuries_res)

    odds_res = adapter.fetch_public_odds_optional(client, teams_df=teams_res.dataframe, league=league)
    save_interim(odds_res.dataframe, cfg.paths.interim_dir, "odds")
    insert_snapshot(db, odds_res)
    insert_odds_snapshot_and_lines(db, league=league, odds_res=odds_res)

    xg_res = adapter.fetch_xg_optional(client)
    save_interim(xg_res.dataframe, cfg.paths.interim_dir, "xg")
    insert_snapshot(db, xg_res)

    results_df = adapter.build_results_from_games(games_res.dataframe)
    upsert_results(db, results_df)

    logger.info(
        "Fetch complete | league=%s games=%d final=%d upcoming=%d players=%d goalie_rows=%d",
        league,
        len(games_res.dataframe),
        int(games_res.dataframe["status_final"].sum()) if not games_res.dataframe.empty else 0,
        len(schedule_res.dataframe),
        len(players_res.dataframe),
        len(goalies_res.dataframe),
    )


def fetch_odds(cfg: AppConfig) -> None:
    db = Database(cfg.paths.db_path)
    db.init_schema()
    client = client_from_config(cfg)
    adapter = get_league_adapter(cfg.data.league)
    league = adapter.code

    teams_df = latest_teams_df(db, league)
    if teams_df.empty:
        teams_res = adapter.fetch_teams(client)
        save_interim(teams_res.dataframe, cfg.paths.interim_dir, "teams")
        insert_snapshot(db, teams_res)
        upsert_teams(
            db,
            teams_df=teams_res.dataframe,
            league=league,
            snapshot_id=teams_res.snapshot_id,
            as_of_utc=teams_res.extracted_at_utc,
        )
        teams_df = teams_res.dataframe

    odds_res = adapter.fetch_public_odds_optional(client, teams_df=teams_df, league=league)
    save_interim(odds_res.dataframe, cfg.paths.interim_dir, "odds")
    insert_snapshot(db, odds_res)
    insert_odds_snapshot_and_lines(db, league=league, odds_res=odds_res)

    logger.info(
        "Odds fetch complete | league=%s snapshot=%s events=%s rows=%d requests_remaining=%s",
        league,
        odds_res.snapshot_id,
        odds_res.metadata.get("n_events"),
        len(odds_res.dataframe),
        odds_res.metadata.get("requests_remaining"),
    )


def refresh_data(cfg: AppConfig) -> None:
    fetch_data(cfg)
    fetch_odds(cfg)
    logger.info("Data refresh complete | league=%s", canonicalize_league(cfg.data.league))


def build_features(cfg: AppConfig) -> None:
    db = Database(cfg.paths.db_path)
    db.init_schema()

    res = build_features_from_interim(cfg.paths.interim_dir, cfg.paths.processed_dir, league=cfg.data.league)
    db.execute(
        """
        INSERT OR REPLACE INTO feature_sets(feature_set_version, created_at_utc, snapshot_id, feature_columns_json, metadata_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            res.feature_set_version,
            utc_now_iso(),
            latest_snapshot_id(db),
            to_json(res.feature_columns),
            to_json(res.metadata),
        ),
    )
    logger.info("Features built | rows=%d features=%d version=%s", len(res.dataframe), len(res.feature_columns), res.feature_set_version)
