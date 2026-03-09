import json

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.evaluation.validation_pipeline import build_validation_tasks
from src.services.validate import run_saved_validation
from src.storage.db import Database


def test_run_saved_validation_regenerates_glm_artifacts_without_training(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.processed_dir = str(tmp_path / "processed")
    cfg.paths.db_path = str(tmp_path / "processed" / "nba_forecast.db")
    cfg.data.league = "NBA"

    rng = np.random.default_rng(41)
    n = 240
    signal = rng.normal(0.0, 1.0, n)
    counter = rng.normal(0.0, 1.0, n)
    logits = 1.0 * signal - 0.8 * counter
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob)

    features_df = pd.DataFrame(
        {
            "start_time_utc": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "game_date_utc": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "home_win": y,
            "signal": signal,
            "counter": counter,
        }
    )
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(processed_dir / "features.csv", index=False)

    artifact_dir = tmp_path / "artifacts" / "models" / "saved_validation_run"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_payload = {
        "model_run_id": "run_saved_validation",
        "feature_set_version": "fset_saved_validation",
        "selected_models": ["glm_ridge"],
        "feature_columns": ["signal", "counter"],
        "glm_feature_columns": ["signal", "counter"],
        "model_feature_columns": {"glm_ridge": ["signal", "counter"]},
        "model_dir": str(artifact_dir),
    }
    (artifact_dir / "run_payload.json").write_text(json.dumps(run_payload, indent=2, sort_keys=True))

    db = Database(cfg.paths.db_path)
    db.init_schema()
    db.execute(
        """
        INSERT INTO model_runs(
          model_run_id, model_name, run_type, created_at_utc, snapshot_id,
          feature_set_version, params_json, metrics_json, artifact_path, model_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "run_saved_validation__glm_ridge",
            "glm_ridge",
            "daily_train",
            "2026-03-08T03:10:00+00:00",
            None,
            "fset_saved_validation",
            "{}",
            "{}",
            str(artifact_dir),
            "run_saved_validation",
        ),
    )

    tasks = [task for task in build_validation_tasks() if task.name == "glm_diagnostics"]
    outputs = run_saved_validation(cfg, tasks=tasks)

    assert [spec.section for spec in outputs.sections] == [
        "glm_residual_summary",
        "glm_residual_feature_summary",
        "glm_working_residual_bins_linear_predictor",
        "glm_working_residual_bins_features",
        "glm_working_residual_bins_weight",
        "glm_partial_residual_bins",
    ]

    validation_root = tmp_path / "artifacts" / "validation" / "nba"
    assert (validation_root / "glm" / "residuals" / "validation_glm_residual_summary.json").exists()
    assert (validation_root / "glm" / "residuals" / "plots" / "glm_validation_partial_residuals_signal.png").exists()

    metadata = json.loads((validation_root / "validation_run_metadata.json").read_text())
    assert metadata["model_run_id"] == "run_saved_validation"


def test_run_saved_validation_supports_elastic_net_saved_runs(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.processed_dir = str(tmp_path / "processed")
    cfg.paths.db_path = str(tmp_path / "processed" / "nhl_forecast.db")
    cfg.data.league = "NHL"

    rng = np.random.default_rng(13)
    n = 220
    signal = rng.normal(0.0, 1.0, n)
    counter = rng.normal(0.0, 1.0, n)
    logits = 0.9 * signal - 0.6 * counter
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob)

    features_df = pd.DataFrame(
        {
            "start_time_utc": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "game_date_utc": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "home_win": y,
            "signal": signal,
            "counter": counter,
        }
    )
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(processed_dir / "features.csv", index=False)

    artifact_dir = tmp_path / "artifacts" / "models" / "saved_elastic_net_run"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_payload = {
        "model_run_id": "run_saved_elastic_net",
        "feature_set_version": "fset_saved_elastic_net",
        "selected_models": ["glm_elastic_net"],
        "feature_columns": ["signal", "counter"],
        "model_feature_columns": {"glm_ridge": ["signal", "counter"]},
        "glm_tuning_by_model": {"glm_elastic_net": {"best_c": 0.5, "best_l1_ratio": 0.5}},
        "model_dir": str(artifact_dir),
    }
    (artifact_dir / "run_payload.json").write_text(json.dumps(run_payload, indent=2, sort_keys=True))

    db = Database(cfg.paths.db_path)
    db.init_schema()
    db.execute(
        """
        INSERT INTO model_runs(
          model_run_id, model_name, run_type, created_at_utc, snapshot_id,
          feature_set_version, params_json, metrics_json, artifact_path, model_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "run_saved_elastic_net__glm_elastic_net",
            "glm_elastic_net",
            "daily_train",
            "2026-03-09T04:00:00+00:00",
            None,
            "fset_saved_elastic_net",
            "{}",
            "{}",
            str(artifact_dir),
            "run_saved_elastic_net",
        ),
    )

    tasks = [task for task in build_validation_tasks() if task.name == "glm_diagnostics"]
    outputs = run_saved_validation(cfg, tasks=tasks)

    assert [spec.section for spec in outputs.sections] == [
        "glm_residual_summary",
        "glm_residual_feature_summary",
        "glm_working_residual_bins_linear_predictor",
        "glm_working_residual_bins_features",
        "glm_working_residual_bins_weight",
        "glm_partial_residual_bins",
    ]

    summary = json.loads(
        (tmp_path / "artifacts" / "validation" / "nhl" / "glm" / "residuals" / "validation_glm_residual_summary.json").read_text()
    )
    assert "glm_elastic_net" in summary["headline"]
