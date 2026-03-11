from pathlib import Path

from src.storage.db import Database


def test_init_schema_moves_legacy_diagnostics_out_of_predictions(tmp_path: Path):
    db = Database(str(tmp_path / "x.db"))
    db.init_schema()

    db.execute(
        """INSERT INTO predictions(
            game_id, as_of_utc, model_name, model_run_id, feature_set_version, snapshot_id,
            game_date_utc, home_team, away_team, prob_home_win, pred_winner, metadata_json
        ) VALUES (
            1, '2025-12-31T00:00:00Z', 'ensemble', 'live_run', 'f1', 's1',
            '2026-01-01', 'TOR', 'MTL', 0.6, 'TOR', '{"source":"train_upcoming"}'
        )"""
    )
    db.execute(
        """INSERT INTO predictions(
            game_id, as_of_utc, model_name, model_run_id, feature_set_version, snapshot_id,
            game_date_utc, home_team, away_team, prob_home_win, pred_winner, metadata_json
        ) VALUES (
            2, '2025-12-31T00:00:00Z', 'ensemble', 'diag_run', 'f1', 's1',
            '2026-01-02', 'BOS', 'NYK', 0.4, 'NYK', '{"source":"walk_forward_backtest"}'
        )"""
    )

    db.init_schema()

    live_rows = db.query("SELECT game_id, model_run_id FROM predictions ORDER BY game_id")
    diagnostic_rows = db.query("SELECT game_id, model_run_id FROM prediction_diagnostics ORDER BY game_id")

    assert live_rows == [{"game_id": 1, "model_run_id": "live_run"}]
    assert diagnostic_rows == [{"game_id": 2, "model_run_id": "diag_run"}]
