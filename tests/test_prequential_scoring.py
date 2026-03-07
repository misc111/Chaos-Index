from pathlib import Path

from src.storage.db import Database
from src.training.prequential import score_predictions



def test_prequential_scoring(tmp_path: Path):
    db = Database(str(tmp_path / "x.db"))
    db.init_schema()

    db.execute(
        "INSERT INTO results(game_id, season, game_date_utc, final_utc, home_team, away_team, home_score, away_score, home_win, ingested_at_utc) VALUES (1, 20252026, '2026-01-01', '2026-01-01T00:00:00Z', 'TOR', 'MTL', 3, 2, 1, '2026-01-01T01:00:00Z')"
    )
    db.execute(
        "INSERT INTO predictions(game_id, as_of_utc, model_name, model_run_id, feature_set_version, snapshot_id, game_date_utc, home_team, away_team, prob_home_win, pred_winner) VALUES (1, '2025-12-31T00:00:00Z', 'glm_ridge', 'run1', 'f1', 's1', '2026-01-01', 'TOR', 'MTL', 0.6, 'TOR')"
    )

    out = score_predictions(db, windows_days=[7, 30])
    assert out["n_scored"] >= 1
    agg = db.query("SELECT COUNT(*) as n FROM performance_aggregates")
    assert agg[0]["n"] >= 1
