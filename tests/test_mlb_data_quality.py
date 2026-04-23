import json
import sqlite3

import pandas as pd

from src.common.config import load_config
from src.evaluation.mlb_data_quality import assess_mlb_data_quality
from src.evaluation.validation_pipeline import build_validation_tasks, run_validation_pipeline
from src.storage.db import Database


def test_assess_mlb_data_quality_flags_schema_and_integrity_failures(tmp_path):
    db_path = tmp_path / "mlb_quality.db"
    db = Database(str(db_path))
    db.init_schema()

    db.execute(
        """
        INSERT INTO mlb_games(
          game_id, season, game_date_utc, start_time_utc, game_state,
          home_team, away_team, home_team_id, away_team_id, venue, as_of_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1001,
            2026,
            "2026-04-15T00:00:00+00:00",
            "2026-04-15T18:10:00+00:00",
            "scheduled",
            "CHC",
            "STL",
            1,
            2,
            "Wrigley Field",
            "2026-04-15T10:00:00+00:00",
        ),
    )
    db.execute(
        """
        INSERT INTO mlb_odds_snapshots(
          odds_snapshot_id, source, league, as_of_utc, event_count, row_count
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("snap-good", "unit", "MLB", "2026-04-15T17:00:00+00:00", 1, 2),
    )
    db.execute(
        """
        INSERT INTO mlb_odds_snapshots(
          odds_snapshot_id, source, league, as_of_utc, event_count, row_count
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("snap-late", "unit", "MLB", "2026-04-15T19:00:00+00:00", 1, 2),
    )

    base_line = (
        "snap-good",
        "MLB",
        1001,
        "baseball_mlb",
        "event-1001",
        "2026-04-15T18:10:00+00:00",
        "2026-04-15",
        "Cubs",
        "Cardinals",
        "CHC",
        "STL",
        "book-a",
        "Book A",
        "2026-04-15T17:00:00+00:00",
        "h2h",
        "CHC",
        "home",
        "CHC",
        -120.0,
        None,
        0.5454545455,
        "2026-04-15T17:00:01+00:00",
    )
    insert_sql = """
        INSERT INTO mlb_odds_market_lines(
          odds_snapshot_id, league, game_id, sport_key, odds_event_id, commence_time_utc,
          commence_date_central, api_home_team, api_away_team, home_team, away_team,
          bookmaker_key, bookmaker_title, bookmaker_last_update_utc, market_key,
          outcome_name, outcome_side, outcome_team, outcome_price, outcome_point,
          implied_probability, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    db.execute(insert_sql, base_line)
    db.execute(insert_sql, base_line)
    db.execute(
        insert_sql,
        (
            "snap-good",
            "MLB",
            1001,
            "baseball_mlb",
            "event-1001",
            "2026-04-15T18:10:00+00:00",
            "2026-04-15",
            "Cubs",
            "Cardinals",
            "CHC",
            "STL",
            "book-a",
            "Book A",
            "2026-04-15T17:00:00+00:00",
            "h2h",
            "STL",
            "away",
            "STL",
            110.0,
            None,
            0.5238095238,
            "2026-04-15T17:00:01+00:00",
        ),
    )
    db.execute(
        insert_sql,
        (
            "snap-missing",
            "MLB",
            1001,
            "baseball_mlb",
            "event-missing-snapshot",
            "2026-04-15T18:10:00+00:00",
            "2026-04-15",
            "Cubs",
            "Cardinals",
            "CHC",
            "STL",
            "book-b",
            "Book B",
            "2026-04-15T17:05:00+00:00",
            "spreads",
            "CHC",
            "home",
            "CHC",
            -110.0,
            -1.5,
            0.5238095238,
            "2026-04-15T17:05:01+00:00",
        ),
    )
    db.execute(
        insert_sql,
        (
            "snap-good",
            "MLB",
            9999,
            "baseball_mlb",
            "event-missing-game",
            "2026-04-15T18:10:00+00:00",
            "2026-04-15",
            "Ghosts",
            "Phantoms",
            "GST",
            "PHN",
            "book-c",
            "Book C",
            "2026-04-15T17:06:00+00:00",
            "h2h",
            "GST",
            "home",
            "GST",
            -130.0,
            None,
            0.5652173913,
            "2026-04-15T17:06:01+00:00",
        ),
    )
    db.execute(
        insert_sql,
        (
            "snap-late",
            "MLB",
            1001,
            "baseball_mlb",
            "event-late",
            "2026-04-15T18:10:00+00:00",
            "2026-04-15",
            "Cubs",
            "Cardinals",
            "CHC",
            "STL",
            "book-d",
            "Book D",
            "2026-04-15T19:00:00+00:00",
            "h2h",
            "CHC",
            "home",
            "CHC",
            -105.0,
            None,
            1.2,
            "2026-04-15T19:00:01+00:00",
        ),
    )
    db.execute(
        insert_sql,
        (
            "snap-late",
            "MLB",
            1001,
            "baseball_mlb",
            "event-late",
            "2026-04-15T18:10:00+00:00",
            "2026-04-15",
            "Cubs",
            "Cardinals",
            "CHC",
            "STL",
            "book-d",
            "Book D",
            "2026-04-15T19:00:00+00:00",
            "h2h",
            "STL",
            "away",
            "STL",
            -105.0,
            None,
            0.52,
            "2026-04-15T19:00:01+00:00",
        ),
    )

    summary, issues = assess_mlb_data_quality(str(db_path))

    assert summary["passed"] is False
    assert summary["missing_tables"] == []
    assert summary["row_counts"]["mlb_odds_market_lines"] == 7
    assert summary["checks"]["required_tables_present"]["passed"] is True
    assert summary["checks"]["no_duplicate_market_lines"]["issue_count"] == 1
    assert summary["checks"]["no_orphan_snapshot_references"]["issue_count"] == 1
    assert summary["checks"]["no_orphan_game_references"]["issue_count"] == 1
    assert summary["checks"]["pregame_snapshot_ordering"]["issue_count"] == 2
    assert summary["checks"]["implied_probability_bounds"]["issue_count"] == 1
    assert summary["checks"]["two_way_market_vig_bounds"]["issue_count"] == 1
    assert summary["coverage_status"] == "partial"
    assert summary["n_pending_checks"] >= 1
    assert "market_pair_completeness" in summary["pending_checks"]
    assert summary["checks"]["market_pair_completeness"]["status"] == "pending"
    assert summary["checks"]["market_pair_completeness"]["implementation_status"] == "pending_not_implemented"

    assert set(issues["issue_type"]) == {
        "duplicate_market_line",
        "orphan_snapshot_reference",
        "orphan_game_reference",
        "snapshot_after_start_time",
        "implied_probability_out_of_bounds",
        "two_way_market_vig_out_of_bounds",
    }


def test_validation_pipeline_writes_mlb_data_quality_artifacts(tmp_path):
    db_path = tmp_path / "processed" / "mlb_forecast.db"
    db = Database(str(db_path))
    db.init_schema()
    db.execute(
        """
        INSERT INTO mlb_games(
          game_id, season, game_date_utc, start_time_utc, game_state,
          home_team, away_team, home_team_id, away_team_id, venue, as_of_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            2001,
            2026,
            "2026-04-20T00:00:00+00:00",
            "2026-04-20T19:10:00+00:00",
            "scheduled",
            "LAD",
            "SDP",
            3,
            4,
            "Dodger Stadium",
            "2026-04-20T11:00:00+00:00",
        ),
    )
    db.execute(
        """
        INSERT INTO mlb_odds_snapshots(
          odds_snapshot_id, source, league, as_of_utc, event_count, row_count
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("snap-1", "unit", "MLB", "2026-04-20T20:00:00+00:00", 1, 1),
    )
    db.execute(
        """
        INSERT INTO mlb_odds_market_lines(
          odds_snapshot_id, league, game_id, sport_key, odds_event_id, commence_time_utc,
          commence_date_central, api_home_team, api_away_team, home_team, away_team,
          bookmaker_key, bookmaker_title, bookmaker_last_update_utc, market_key,
          outcome_name, outcome_side, outcome_team, outcome_price, outcome_point,
          implied_probability, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "snap-1",
            "MLB",
            2001,
            "baseball_mlb",
            "event-2001",
            "2026-04-20T19:10:00+00:00",
            "2026-04-20",
            "Dodgers",
            "Padres",
            "LAD",
            "SDP",
            "book-a",
            "Book A",
            "2026-04-20T20:00:00+00:00",
            "h2h",
            "LAD",
            "home",
            "LAD",
            -110.0,
            None,
            0.5238095238,
            "2026-04-20T20:00:01+00:00",
        ),
    )

    cfg = load_config("configs/default.yaml")
    cfg.paths.db_path = str(db_path)
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.data.league = "MLB"

    tasks = [task for task in build_validation_tasks() if task.name == "mlb_data_quality"]
    outputs = run_validation_pipeline(
        {
            "models": {},
            "train_df": pd.DataFrame(
                {
                    "game_date_utc": ["2026-04-20T00:00:00+00:00"],
                    "home_win": [1],
                }
            ),
            "feature_columns": [],
        },
        cfg,
        tasks=tasks,
    )

    assert [spec.section for spec in outputs.sections] == [
        "mlb_data_quality_summary",
        "mlb_data_quality_issues",
    ]

    root = tmp_path / "artifacts" / "validation" / "mlb" / "diagnostics" / "data_quality"
    summary = json.loads((root / "validation_mlb_data_quality_summary.json").read_text())
    issues = pd.read_csv(root / "validation_mlb_data_quality_issues.csv")
    manifest = json.loads((tmp_path / "artifacts" / "validation" / "mlb" / "validation_manifest.json").read_text())

    assert summary["passed"] is False
    assert summary["checks"]["pregame_snapshot_ordering"]["passed"] is False
    assert summary["coverage_status"] == "partial"
    assert summary["n_pending_checks"] >= 1
    assert summary["checks"]["snapshot_staleness_window"]["status"] == "pending"
    assert "snapshot_after_start_time" in set(issues["issue_type"])
    assert [section["section"] for section in manifest["sections"]] == [
        "mlb_data_quality_summary",
        "mlb_data_quality_issues",
    ]


def test_assess_mlb_data_quality_reports_missing_required_tables(tmp_path):
    db_path = tmp_path / "bare.db"
    with sqlite3.connect(db_path):
        pass

    summary, issues = assess_mlb_data_quality(str(db_path))

    assert summary["passed"] is False
    assert summary["checks"]["required_tables_present"]["passed"] is False
    assert set(summary["missing_tables"]) == {"mlb_games", "mlb_odds_market_lines", "mlb_odds_snapshots"}
    assert summary["coverage_status"] == "partial"
    assert summary["n_pending_checks"] >= 1
    assert set(issues["issue_type"]) == {"missing_required_table"}
