from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.common.utils import ensure_dir
from src.storage.schema import SCHEMA_SQL


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
