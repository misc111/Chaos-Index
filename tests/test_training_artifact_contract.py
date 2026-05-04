from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.services.train import persist_predictions, train_models
from src.storage.db import Database
from src.training.train import train_and_predict


def _synthetic_features(n_train: int = 220, n_upcoming: int = 16) -> pd.DataFrame:
    n = n_train + n_upcoming
    rng = np.random.default_rng(77)
    start = pd.date_range("2026-01-01", periods=n, freq="D")
    signal = rng.normal(0.0, 1.0, n)
    counter = rng.normal(0.0, 1.0, n)
    travel_diff = rng.normal(0.0, 1.0, n)
    rest_diff = rng.normal(0.0, 1.0, n)
    diff_bullpen_fatigue = rng.normal(0.0, 1.0, n)
    diff_lineup_talent = rng.normal(0.0, 1.0, n)
    lineup_stability_diff = rng.normal(0.0, 1.0, n)
    park_run_effect = rng.normal(0.0, 1.0, n)
    park_weather_run_effect = rng.normal(0.0, 1.0, n)
    starting_pitcher_hand_matchup = rng.normal(0.0, 1.0, n)
    logits = 1.1 * signal - 0.8 * counter + 0.25 * rest_diff - 0.2 * travel_diff
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob[:n_train]).astype(float)
    prior_model_offset_logit = 0.85 * signal - 0.35 * counter + 0.15 * rest_diff
    market_offset_logit = prior_model_offset_logit + 0.20 * diff_lineup_talent - 0.10 * diff_bullpen_fatigue

    frame = pd.DataFrame(
        {
            "game_id": np.arange(1, n + 1),
            "season": [2026] * n,
            "game_date_utc": start.date.astype(str),
            "start_time_utc": start.astype(str),
            "home_team": ["CHC"] * n,
            "away_team": ["LAD"] * n,
            "venue": ["Wrigley Field"] * n,
            "as_of_utc": ["2026-01-01T00:00:00+00:00"] * n,
            "status_final": [1] * n_train + [0] * n_upcoming,
            "diff_starting_pitcher_quality": signal,
            "diff_bullpen_quality": counter,
            "diff_bullpen_availability": -diff_bullpen_fatigue,
            "diff_bullpen_fatigue": diff_bullpen_fatigue,
            "diff_lineup_talent": diff_lineup_talent,
            "lineup_stability_diff": lineup_stability_diff,
            "starter_quality_edge": signal,
            "bullpen_quality_edge": counter,
            "lineup_confirmation_share": np.ones(n, dtype=float),
            "park_run_effect": park_run_effect,
            "park_weather_run_effect": park_weather_run_effect,
            "starting_pitcher_hand_matchup": starting_pitcher_hand_matchup,
            "travel_diff": travel_diff,
            "rest_diff": rest_diff,
            "prior_model_offset_logit": prior_model_offset_logit,
            "market_offset_logit": market_offset_logit,
            "home_field_advantage": np.ones(n, dtype=float),
            "home_win": list(y) + [np.nan] * n_upcoming,
            "home_score": [5 if value else 3 for value in y] + [np.nan] * n_upcoming,
            "away_score": [3 if value else 5 for value in y] + [np.nan] * n_upcoming,
        }
    )
    return frame


def test_train_and_predict_emits_model_run_contract(tmp_path: Path) -> None:
    out = train_and_predict(
        features_df=_synthetic_features(),
        feature_set_version="artifact_contract_test",
        artifacts_dir=str(tmp_path / "artifacts"),
        bayes_cfg={},
        selected_models=["glm_lasso"],
        league="MLB",
    )

    run_payload = out["run_payload"]
    assert run_payload["contract_version"] == 1
    assert run_payload["training_output_files"]["run_payload"] == "run_payload.json"
    assert "run_contract" in run_payload

    artifact_rows = {row["model_name"]: row for row in run_payload["model_artifacts"]}
    assert "glm_lasso" in artifact_rows
    assert artifact_rows["glm_lasso"]["lane"] == "core"
    assert artifact_rows["glm_lasso"]["penalty"]["penalty_family"] == "lasso"
    assert artifact_rows["glm_lasso"]["artifact_files"]["binary"].endswith(".joblib")
    assert artifact_rows["glm_lasso"]["artifact_files"]["coefficients"].endswith("_coefficients.csv")
    assert artifact_rows["glm_lasso"]["artifact_files"]["fit_metadata"].endswith("_fit_metadata.json")
    assert artifact_rows["glm_lasso"]["metrics_summary"]["train"]["log_loss"] >= 0.0
    assert artifact_rows["ensemble"]["model_family"] == "ensemble"
    assert artifact_rows["ensemble"]["artifact_files"]["upcoming_forecasts"] in {"upcoming_forecasts.parquet", "upcoming_forecasts.csv"}

    contract = run_payload["run_contract"]
    assert contract["league"] == "MLB"
    assert contract["model_run_id"] == run_payload["model_run_id"]
    assert {row["model_name"] for row in contract["model_artifacts"]} >= {"glm_lasso", "ensemble"}


def test_train_models_persists_artifact_contract_and_validation_rows(tmp_path: Path) -> None:
    cfg = load_config("configs/mlb.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.processed_dir = str(tmp_path / "processed")
    cfg.paths.db_path = str(tmp_path / "processed" / "mlb_forecast.db")
    cfg.feature_policy.mode = "research"
    cfg.feature_policy.registry_path = str(tmp_path / "feature_registry_{league}.yaml")

    processed_dir = Path(cfg.paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    _synthetic_features().to_csv(processed_dir / "features.csv", index=False)
    db = Database(cfg.paths.db_path)
    db.init_schema()
    db.execute(
        """
        INSERT INTO feature_sets(
          feature_set_version, created_at_utc, snapshot_id, feature_columns_json, metadata_json
        ) VALUES (
          'features_v1', '2026-04-01T12:00:00Z', 'snapshot_v1', '[]', '{}'
        )
        """
    )

    train_models(cfg, models_arg="glm_ridge")

    db = Database(cfg.paths.db_path)
    model_runs = db.query("SELECT model_name, params_json, artifact_path FROM model_runs ORDER BY model_name")
    assert model_runs
    ridge_row = next(row for row in model_runs if row["model_name"] == "glm_ridge")
    params_payload = json.loads(ridge_row["params_json"])
    assert params_payload["contract_version"] == 1
    assert params_payload["artifact_contract"]["model_name"] == "glm_ridge"
    assert Path(params_payload["run_contract_path"]).name == "run_payload.json"
    assert Path(str(ridge_row["artifact_path"])).name == "glm_ridge.joblib"
    persisted_run_payload = json.loads(Path(params_payload["run_contract_path"]).read_text())
    assert persisted_run_payload["validation_outputs"]
    assert persisted_run_payload["run_contract"]["validation_outputs"]

    validation_rows = db.query("SELECT validation_name, split_label, artifact_path FROM validation_results")
    assert validation_rows
    assert {row["split_label"] for row in validation_rows} == {"train_test:time"}
    assert any(str(row["artifact_path"]).endswith(".json") or str(row["artifact_path"]).endswith(".csv") for row in validation_rows)


def test_train_models_fails_fast_without_feature_set_metadata(tmp_path: Path) -> None:
    cfg = load_config("configs/mlb.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.processed_dir = str(tmp_path / "processed")
    cfg.paths.db_path = str(tmp_path / "processed" / "mlb_forecast.db")
    cfg.feature_policy.mode = "research"
    cfg.feature_policy.registry_path = str(tmp_path / "feature_registry_{league}.yaml")

    processed_dir = Path(cfg.paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    _synthetic_features().to_csv(processed_dir / "features.csv", index=False)

    try:
        train_models(cfg, models_arg="glm_ridge")
    except RuntimeError as exc:
        assert "feature_sets" in str(exc)
    else:
        raise AssertionError("train_models must fail when feature-build metadata is missing")


def test_persist_predictions_requires_pregame_start_time_and_preserves_it(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "mlb.db"))
    db.init_schema()
    forecasts = pd.DataFrame(
        [
            {
                "game_id": 1,
                "game_date_utc": "2026-04-01",
                "start_time_utc": "2026-04-01T18:00:00Z",
                "home_team": "CHC",
                "away_team": "LAD",
                "as_of_utc": "2026-04-01T12:00:00Z",
                "ensemble_prob_home_win": 0.56,
                "per_model_probs_json": "{}",
                "spread_min": 0.0,
                "spread_median": 0.0,
                "spread_max": 0.0,
                "spread_mean": 0.0,
                "spread_sd": 0.0,
                "spread_iqr": 0.0,
                "bayes_ci_low": None,
                "bayes_ci_high": None,
                "uncertainty_flags_json": "{}",
            }
        ]
    )
    per_model_probs = pd.DataFrame([{"game_id": 1, "glm_ridge": 0.55}])

    persist_predictions(db, forecasts, per_model_probs, "run_1", "features_v1")

    prediction_rows = db.query("SELECT game_id, model_name, start_time_utc FROM predictions ORDER BY model_name")
    forecast_rows = db.query("SELECT game_id, start_time_utc FROM upcoming_game_forecasts")
    assert prediction_rows == [
        {"game_id": 1, "model_name": "ensemble", "start_time_utc": "2026-04-01T18:00:00Z"},
        {"game_id": 1, "model_name": "glm_ridge", "start_time_utc": "2026-04-01T18:00:00Z"},
    ]
    assert forecast_rows == [{"game_id": 1, "start_time_utc": "2026-04-01T18:00:00Z"}]

    post_start = forecasts.copy()
    post_start.loc[0, "as_of_utc"] = "2026-04-01T18:00:00Z"
    try:
        persist_predictions(db, post_start, per_model_probs, "run_2", "features_v1")
    except RuntimeError as exc:
        assert "strictly pregame" in str(exc)
    else:
        raise AssertionError("post-start forecasts must not enter the frozen prediction ledger")

    missing_start = forecasts.drop(columns=["start_time_utc"])
    try:
        persist_predictions(db, missing_start, per_model_probs, "run_3", "features_v1")
    except RuntimeError as exc:
        assert "start_time_utc" in str(exc)
    else:
        raise AssertionError("forecasts without scheduled first pitch must not enter the frozen prediction ledger")


def test_train_and_predict_emits_lasso_credibility_contracts_end_to_end(tmp_path: Path) -> None:
    out = train_and_predict(
        features_df=_synthetic_features(),
        feature_set_version="lasso_credibility_contract_test",
        artifacts_dir=str(tmp_path / "artifacts"),
        bayes_cfg={},
        selected_models=["glm_lasso_prior_credibility"],
        selected_model_feature_columns={
            "glm_lasso_prior_credibility": [
                "diff_starting_pitcher_quality",
                "diff_bullpen_quality",
                "rest_diff",
            ]
        },
        league="MLB",
    )

    run_payload = out["run_payload"]
    assert "glm_lasso_prior_credibility" in run_payload["credibility_tuning_by_model"]
    artifact_rows = {row["model_name"]: row for row in run_payload["model_artifacts"]}
    credibility_row = artifact_rows["glm_lasso_prior_credibility"]

    assert credibility_row["lane"] == "core"
    assert credibility_row["model_family"] == "credibility"
    assert credibility_row["target_name"] == "home_win"
    assert credibility_row["distribution"] == "binomial"
    assert credibility_row["link_function"] == "logit"
    assert credibility_row["offset_columns"] == ["prior_model_offset_logit"]
    assert credibility_row["credibility"]["complement_kind"] == "prior"
    assert credibility_row["credibility"]["complement_column"] == "prior_model_offset_logit"
    assert credibility_row["credibility"]["complement_label"] == "Prior-model offset logit"
    assert credibility_row["credibility"]["offset_scale"] == "logit"
    assert credibility_row["credibility"]["p_values_reported"] is False
    assert credibility_row["penalty"]["penalty_family"] == "lasso"
    assert credibility_row["artifact_files"]["binary"].endswith(".joblib")
    assert credibility_row["artifact_files"]["coefficients"].endswith("_coefficients.csv")
    assert credibility_row["artifact_files"]["credibility_metadata"].endswith("_credibility_metadata.json")
    assert credibility_row["artifact_files"]["relativity"].endswith("_relativity.csv")
    assert credibility_row["fit_summary"]["fit_variant"] == "prior_offset_lasso_credibility"

    contract = run_payload["run_contract"]
    contract_rows = {row["model_name"]: row for row in contract["model_artifacts"]}
    assert contract_rows["glm_lasso_prior_credibility"]["credibility"]["complement_label"] == "Prior-model offset logit"
    assert contract_rows["glm_lasso_prior_credibility"]["penalty"]["lambda_value"] > 0.0
