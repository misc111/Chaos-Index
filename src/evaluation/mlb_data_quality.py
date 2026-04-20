from __future__ import annotations

import sqlite3
from typing import Any

import pandas as pd

from src.common.time import utc_now_iso


REQUIRED_TABLES = (
    "mlb_games",
    "mlb_odds_snapshots",
    "mlb_odds_market_lines",
)

ISSUE_COLUMNS = [
    "check",
    "issue_type",
    "table_name",
    "odds_snapshot_id",
    "game_id",
    "odds_event_id",
    "bookmaker_key",
    "market_key",
    "detail",
    "observed_value",
    "reference_value",
]


def _empty_issues() -> pd.DataFrame:
    return pd.DataFrame(columns=ISSUE_COLUMNS)


def _normalize_issue_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_issues()
    return frame.reindex(columns=ISSUE_COLUMNS, fill_value=None)


def _query_frame(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)


def _existing_tables(conn: sqlite3.Connection) -> set[str]:
    rows = _query_frame(
        conn,
        """
        SELECT name
        FROM sqlite_master
        WHERE type IN ('table', 'view')
        """,
    )
    return {str(value) for value in rows.get("name", pd.Series(dtype=str)).tolist()}


def _table_row_count(conn: sqlite3.Connection, table_name: str) -> int:
    return int(_query_frame(conn, f"SELECT COUNT(*) AS row_count FROM {table_name}")["row_count"].iloc[0])


def _duplicate_market_lines(conn: sqlite3.Connection) -> pd.DataFrame:
    return _query_frame(
        conn,
        """
        SELECT
          'no_duplicate_market_lines' AS "check",
          'duplicate_market_line' AS issue_type,
          NULL AS table_name,
          odds_snapshot_id,
          game_id,
          odds_event_id,
          bookmaker_key,
          market_key,
          'duplicate market line key' AS detail,
          row_count AS observed_value,
          1 AS reference_value
        FROM (
          SELECT
            odds_snapshot_id,
            game_id,
            odds_event_id,
            bookmaker_key,
            market_key,
            COALESCE(outcome_name, '') AS outcome_name_key,
            COALESCE(outcome_side, '') AS outcome_side_key,
            COALESCE(outcome_team, '') AS outcome_team_key,
            COALESCE(outcome_point, -999999.0) AS outcome_point_key,
            COUNT(*) AS row_count
          FROM mlb_odds_market_lines
          GROUP BY
            odds_snapshot_id,
            game_id,
            odds_event_id,
            bookmaker_key,
            market_key,
            outcome_name_key,
            outcome_side_key,
            outcome_team_key,
            outcome_point_key
          HAVING COUNT(*) > 1
        )
        """,
    )


def _orphan_snapshot_lines(conn: sqlite3.Connection) -> pd.DataFrame:
    return _query_frame(
        conn,
        """
        SELECT
          'no_orphan_snapshot_references' AS "check",
          'orphan_snapshot_reference' AS issue_type,
          NULL AS table_name,
          l.odds_snapshot_id,
          l.game_id,
          l.odds_event_id,
          l.bookmaker_key,
          l.market_key,
          'odds line references missing snapshot' AS detail,
          l.odds_snapshot_id AS observed_value,
          'mlb_odds_snapshots.odds_snapshot_id' AS reference_value
        FROM mlb_odds_market_lines l
        LEFT JOIN mlb_odds_snapshots s
          ON s.odds_snapshot_id = l.odds_snapshot_id
        WHERE s.odds_snapshot_id IS NULL
        """,
    )


def _orphan_game_lines(conn: sqlite3.Connection) -> pd.DataFrame:
    return _query_frame(
        conn,
        """
        SELECT
          'no_orphan_game_references' AS "check",
          'orphan_game_reference' AS issue_type,
          NULL AS table_name,
          l.odds_snapshot_id,
          l.game_id,
          l.odds_event_id,
          l.bookmaker_key,
          l.market_key,
          'odds line references missing game' AS detail,
          l.game_id AS observed_value,
          'mlb_games.game_id' AS reference_value
        FROM mlb_odds_market_lines l
        LEFT JOIN mlb_games g
          ON g.game_id = l.game_id
        WHERE l.game_id IS NOT NULL
          AND g.game_id IS NULL
        """,
    )


def _snapshot_timing_issues(conn: sqlite3.Connection) -> pd.DataFrame:
    frame = _query_frame(
        conn,
        """
        SELECT
          l.odds_snapshot_id,
          l.game_id,
          l.odds_event_id,
          l.bookmaker_key,
          l.market_key,
          s.as_of_utc AS snapshot_as_of_utc,
          COALESCE(g.start_time_utc, l.commence_time_utc) AS reference_start_time_utc
        FROM mlb_odds_market_lines l
        JOIN mlb_odds_snapshots s
          ON s.odds_snapshot_id = l.odds_snapshot_id
        LEFT JOIN mlb_games g
          ON g.game_id = l.game_id
        WHERE COALESCE(g.start_time_utc, l.commence_time_utc) IS NOT NULL
        """
    )
    if frame.empty:
        return _empty_issues()

    snapshot_ts = pd.to_datetime(frame["snapshot_as_of_utc"], errors="coerce", utc=True)
    start_ts = pd.to_datetime(frame["reference_start_time_utc"], errors="coerce", utc=True)
    bad = frame.loc[snapshot_ts.gt(start_ts)].copy()
    if bad.empty:
        return _empty_issues()

    bad["check"] = "pregame_snapshot_ordering"
    bad["issue_type"] = "snapshot_after_start_time"
    bad["table_name"] = None
    bad["detail"] = "snapshot captured after scheduled first pitch"
    bad["observed_value"] = bad["snapshot_as_of_utc"]
    bad["reference_value"] = bad["reference_start_time_utc"]
    return _normalize_issue_frame(bad)


def _probability_bounds_issues(conn: sqlite3.Connection) -> pd.DataFrame:
    return _query_frame(
        conn,
        """
        SELECT
          'implied_probability_bounds' AS "check",
          'implied_probability_out_of_bounds' AS issue_type,
          NULL AS table_name,
          odds_snapshot_id,
          game_id,
          odds_event_id,
          bookmaker_key,
          market_key,
          'implied probability must be between 0 and 1' AS detail,
          implied_probability AS observed_value,
          '[0, 1]' AS reference_value
        FROM mlb_odds_market_lines
        WHERE implied_probability IS NOT NULL
          AND (implied_probability < 0.0 OR implied_probability > 1.0)
        """,
    )


def _two_way_vig_issues(conn: sqlite3.Connection) -> pd.DataFrame:
    return _query_frame(
        conn,
        """
        SELECT
          'two_way_market_vig_bounds' AS "check",
          'two_way_market_vig_out_of_bounds' AS issue_type,
          NULL AS table_name,
          odds_snapshot_id,
          game_id,
          odds_event_id,
          bookmaker_key,
          market_key,
          'two-way market implied probability sum outside conservative bounds' AS detail,
          implied_probability_sum AS observed_value,
          '[0.95, 1.20]' AS reference_value
        FROM (
          SELECT
            odds_snapshot_id,
            game_id,
            odds_event_id,
            bookmaker_key,
            market_key,
            COALESCE(ABS(outcome_point), -999999.0) AS point_key,
            COUNT(*) AS outcome_count,
            SUM(implied_probability) AS implied_probability_sum
          FROM mlb_odds_market_lines
          WHERE implied_probability IS NOT NULL
          GROUP BY
            odds_snapshot_id,
            game_id,
            odds_event_id,
            bookmaker_key,
            market_key,
            point_key
          HAVING COUNT(*) = 2
             AND (SUM(implied_probability) < 0.95 OR SUM(implied_probability) > 1.20)
        )
        """,
    )


def assess_mlb_data_quality(db_path: str) -> tuple[dict[str, Any], pd.DataFrame]:
    checks: dict[str, dict[str, Any]] = {}

    with sqlite3.connect(db_path) as conn:
        existing_tables = _existing_tables(conn)
        missing_tables = [table for table in REQUIRED_TABLES if table not in existing_tables]
        row_counts = {
            table: (_table_row_count(conn, table) if table in existing_tables else 0)
            for table in REQUIRED_TABLES
        }

        if missing_tables:
            issues = _normalize_issue_frame(
                pd.DataFrame(
                    [
                        {
                            "check": "required_tables_present",
                            "issue_type": "missing_required_table",
                            "table_name": table_name,
                            "detail": "required MLB data-quality table is missing",
                            "observed_value": "missing",
                            "reference_value": "present",
                        }
                        for table_name in missing_tables
                    ]
                )
            )
            checks["required_tables_present"] = {
                "passed": False,
                "issue_count": len(missing_tables),
            }
        else:
            issues_by_check = {
                "no_duplicate_market_lines": _normalize_issue_frame(_duplicate_market_lines(conn)),
                "no_orphan_snapshot_references": _normalize_issue_frame(_orphan_snapshot_lines(conn)),
                "no_orphan_game_references": _normalize_issue_frame(_orphan_game_lines(conn)),
                "pregame_snapshot_ordering": _normalize_issue_frame(_snapshot_timing_issues(conn)),
                "implied_probability_bounds": _normalize_issue_frame(_probability_bounds_issues(conn)),
                "two_way_market_vig_bounds": _normalize_issue_frame(_two_way_vig_issues(conn)),
            }

            checks["required_tables_present"] = {"passed": True, "issue_count": 0}
            for check_name, frame in issues_by_check.items():
                checks[check_name] = {
                    "passed": frame.empty,
                    "issue_count": int(len(frame)),
                }
            issues = _normalize_issue_frame(pd.concat(list(issues_by_check.values()), ignore_index=True))

    failed_checks = [name for name, payload in checks.items() if not payload["passed"]]
    summary = {
        "league": "MLB",
        "generated_at_utc": utc_now_iso(),
        "db_path": db_path,
        "required_tables": list(REQUIRED_TABLES),
        "missing_tables": missing_tables,
        "row_counts": row_counts,
        "checks": checks,
        "failed_checks": failed_checks,
        "n_failed_checks": len(failed_checks),
        "total_issues": int(len(issues)),
        "passed": len(failed_checks) == 0,
    }
    return summary, issues
