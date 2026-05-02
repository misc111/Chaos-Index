from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.registry.commands import command_names
from src.services.current_season_predictiveness import run_current_season_predictiveness
from src.storage.db import Database


def _cfg(tmp_path: Path, db_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(
            db_path=str(db_path),
            artifacts_dir=str(tmp_path / "artifacts"),
        ),
        data=SimpleNamespace(
            league="MLB",
            season_start="2026-03-26",
            season_end="2026-11-01",
        ),
        modeling=SimpleNamespace(rolling_windows_days=[7, 30]),
    )


def _insert_game_and_result(
    db: Database,
    *,
    game_id: int,
    game_date_utc: str,
    start_time_utc: str,
    home_win: int,
) -> None:
    db.execute(
        """
        INSERT INTO games(
            game_id, season, game_date_utc, start_time_utc, game_state,
            home_team, away_team, home_score, away_score, home_win, status_final, as_of_utc
        )
        VALUES (?, 2026, ?, ?, 'Final', 'HOM', 'AWY', ?, ?, ?, 1, '2026-04-01T00:00:00Z')
        """,
        (
            game_id,
            game_date_utc,
            start_time_utc,
            5 if home_win else 2,
            2 if home_win else 5,
            home_win,
        ),
    )
    db.execute(
        """
        INSERT INTO results(
            game_id, season, game_date_utc, final_utc,
            home_team, away_team, home_score, away_score, home_win, ingested_at_utc
        )
        VALUES (?, 2026, ?, ?, 'HOM', 'AWY', ?, ?, ?, '2026-04-02T06:00:00Z')
        """,
        (
            game_id,
            game_date_utc,
            f"{game_date_utc}T23:59:59Z",
            5 if home_win else 2,
            2 if home_win else 5,
            home_win,
        ),
    )


def _insert_prediction(
    db: Database,
    *,
    game_id: int,
    as_of_utc: str,
    game_date_utc: str,
    prob_home_win: float,
    model_name: str = "glm_ridge",
    source: str = "train_upcoming",
) -> None:
    db.execute(
        """
        INSERT INTO predictions(
            game_id, as_of_utc, model_name, model_run_id, feature_set_version, snapshot_id,
            game_date_utc, home_team, away_team, prob_home_win, pred_winner, metadata_json
        )
        VALUES (?, ?, ?, ?, 'features_v1', 'snapshot_v1', ?, 'HOM', 'AWY', ?, 'HOM', ?)
        """,
        (
            game_id,
            as_of_utc,
            model_name,
            f"run_1__{model_name}",
            game_date_utc,
            prob_home_win,
            json.dumps({"source": source}, sort_keys=True),
        ),
    )


def test_current_season_predictiveness_writes_explicit_not_measurable_artifacts(tmp_path: Path) -> None:
    db_path = tmp_path / "mlb.db"
    db = Database(str(db_path))
    db.init_schema()
    _insert_game_and_result(
        db,
        game_id=1,
        game_date_utc="2026-03-27",
        start_time_utc="2026-03-27T18:00:00Z",
        home_win=1,
    )

    result = run_current_season_predictiveness(
        _cfg(tmp_path, db_path),
        as_of_utc="2026-05-02T12:00:00Z",
    )

    assert result["status"] == "not_measurable"
    assert result["league"] == "MLB"
    assert result["scoreboard"]["regular_season_results"] == 1
    assert result["scoreboard"]["scored_live_predictions"] == 0
    assert result["targets"]["moneyline_home_win"]["status"] == "not_measurable"
    assert result["targets"]["runline_home_cover"]["status"] == "ledger_not_available"
    assert result["targets"]["totals_over"]["status"] == "ledger_not_available"
    assert result["betting_overlay"]["status"] == "not_measured"

    json_path = Path(result["artifact_paths"]["summary_json"])
    csv_path = Path(result["artifact_paths"]["metrics_csv"])
    assert json_path.exists()
    assert csv_path.exists()
    payload = json.loads(json_path.read_text())
    rows = pd.read_csv(csv_path)
    assert payload == result
    assert rows.empty


def test_current_season_predictiveness_scores_frozen_pregame_predictions_and_quarantines_leakage(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "mlb.db"
    db = Database(str(db_path))
    db.init_schema()
    outcomes = [1, 0, 1, 0]
    probabilities = [0.82, 0.22, 0.76, 0.31]
    for offset, (outcome, probability) in enumerate(zip(outcomes, probabilities, strict=True), start=1):
        game_date = f"2026-04-0{offset}"
        _insert_game_and_result(
            db,
            game_id=offset,
            game_date_utc=game_date,
            start_time_utc=f"{game_date}T18:00:00Z",
            home_win=outcome,
        )
        _insert_prediction(
            db,
            game_id=offset,
            as_of_utc=f"{game_date}T12:00:00Z",
            game_date_utc=game_date,
            prob_home_win=probability,
        )

    _insert_game_and_result(
        db,
        game_id=99,
        game_date_utc="2026-04-05",
        start_time_utc="2026-04-05T18:00:00Z",
        home_win=1,
    )
    _insert_prediction(
        db,
        game_id=99,
        as_of_utc="2026-04-05T19:00:00Z",
        game_date_utc="2026-04-05",
        prob_home_win=0.99,
    )
    _insert_game_and_result(
        db,
        game_id=100,
        game_date_utc="2026-04-05",
        start_time_utc="2026-04-05T18:00:00Z",
        home_win=0,
    )
    _insert_prediction(
        db,
        game_id=100,
        as_of_utc="2026-04-05T12:00:00Z",
        game_date_utc="2026-04-05",
        prob_home_win=0.01,
        source="train_oof_history",
    )

    result = run_current_season_predictiveness(
        _cfg(tmp_path, db_path),
        as_of_utc="2026-05-02T12:00:00Z",
    )

    assert result["status"] == "measurable"
    assert result["scoreboard"]["scored_live_predictions"] == 4
    assert result["scoreboard"]["quarantined_post_start_predictions"] == 1
    assert result["scoreboard"]["ignored_non_live_predictions"] == 1

    moneyline = result["targets"]["moneyline_home_win"]
    assert moneyline["status"] == "measurable"
    assert moneyline["models"]["glm_ridge"]["cumulative"]["n_games"] == 4
    assert moneyline["models"]["glm_ridge"]["cumulative"]["accuracy"] == 1.0
    assert moneyline["models"]["glm_ridge"]["cumulative"]["auc"] == 1.0
    assert moneyline["models"]["glm_ridge"]["cumulative"]["brier"] < 0.08
    assert moneyline["baselines"]["intercept_only"]["cumulative"]["n_games"] == 4
    assert moneyline["model_vs_baseline"]["glm_ridge"]["intercept_only"]["log_loss_delta"] < 0

    csv_path = Path(result["artifact_paths"]["metrics_csv"])
    rows = pd.read_csv(csv_path)
    assert rows["row_type"].tolist() == ["baseline", "model"]
    assert rows["model_name"].tolist() == ["intercept_only", "glm_ridge"]


def test_current_season_predictiveness_command_is_registered() -> None:
    assert "current-season-predictiveness" in command_names()
