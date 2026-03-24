from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.common.utils import ensure_dir
from src.storage.prediction_history import DIAGNOSTIC_PREDICTION_SOURCES
from src.storage.schema import (
    EFFECTIVE_ODDS_MARKET_LINES_VIEW_NAME,
    EFFECTIVE_ODDS_MARKET_LINES_VIEW_REPLACE_SQL,
    SCHEMA_SQL,
)


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        ensure_dir(Path(db_path).parent)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
            self._apply_online_migrations(conn)

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self.connect() as conn:
            conn.execute(sql, params)

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        if not rows:
            return
        with self.connect() as conn:
            conn.executemany(sql, rows)

    def query(self, sql: str, params: tuple = ()):
        with self.connect() as conn:
            cur = conn.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def _apply_online_migrations(self, conn: sqlite3.Connection) -> None:
        diagnostic_sources = ", ".join(f"'{source}'" for source in DIAGNOSTIC_PREDICTION_SOURCES)
        conn.execute(
            f"""
            INSERT OR IGNORE INTO prediction_diagnostics(
              game_id, as_of_utc, model_name, model_run_id, feature_set_version, snapshot_id,
              game_date_utc, home_team, away_team, prob_home_win, pred_winner, prob_low, prob_high,
              uncertainty_flags_json, metadata_json
            )
            SELECT
              game_id, as_of_utc, model_name, model_run_id, feature_set_version, snapshot_id,
              game_date_utc, home_team, away_team, prob_home_win, pred_winner, prob_low, prob_high,
              uncertainty_flags_json, metadata_json
            FROM predictions
            WHERE COALESCE(json_extract(metadata_json, '$.source'), '') IN ({diagnostic_sources})
            """
        )
        conn.execute(
            f"""
            DELETE FROM predictions
            WHERE COALESCE(json_extract(metadata_json, '$.source'), '') IN ({diagnostic_sources})
            """
        )
        conn.execute(f"DROP VIEW IF EXISTS {EFFECTIVE_ODDS_MARKET_LINES_VIEW_NAME}")
        conn.execute(EFFECTIVE_ODDS_MARKET_LINES_VIEW_REPLACE_SQL)
