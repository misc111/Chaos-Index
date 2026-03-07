"""Shared query-layer contracts."""

from __future__ import annotations

from typing import Any, Protocol


class Queryable(Protocol):
    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        ...


class ConnectionQueryAdapter:
    def __init__(self, conn):
        self.conn = conn

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        cur = self.conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]
