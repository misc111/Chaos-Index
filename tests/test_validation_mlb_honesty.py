import json

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.evaluation.validation_pipeline import build_validation_tasks, run_validation_pipeline
from src.evaluation.validation_stability import break_test_trade_deadline


def test_validation_pipeline_reports_honest_mlb_penalized_significance_and_contract(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.data.league = "MLB"

    rng = np.random.default_rng(2026)
    n = 260
    start = pd.date_range("2025-03-25", periods=n, freq="D")
    market_offset_logit = rng.normal(0.0, 0.45, n)
    diff_starter_era = rng.normal(0.0, 1.0, n)
    starting_pitcher_hand_matchup = rng.integers(0, 2, n)
    diff_bullpen_quality = rng.normal(0.0, 1.0, n)
    diff_lineup_talent = rng.normal(0.0, 1.0, n)
    park_weather_run_effect = rng.normal(0.0, 1.0, n)
    travel_diff = rng.normal(0.0, 1.0, n)
    rest_diff = rng.normal(0.0, 1.0, n)
    elo_home_prob = np.clip(0.5 + rng.normal(0.0, 0.08, n), 0.2, 0.8)
    logits = (
        0.85 * market_offset_logit
        - 0.55 * diff_starter_era
        + 0.45 * diff_bullpen_quality
        + 0.35 * diff_lineup_talent
        + 0.20 * park_weather_run_effect
        - 0.18 * travel_diff
        + 1.10 * (elo_home_prob - 0.5)
    )
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob)

    feature_cols = [
        "market_offset_logit",
        "diff_starter_era",
        "starting_pitcher_hand_matchup",
        "diff_bullpen_quality",
        "diff_lineup_talent",
        "park_weather_run_effect",
        "travel_diff",
        "rest_diff",
        "elo_home_prob",
    ]
    train_df = pd.DataFrame(
        {
            "start_time_utc": start.astype(str),
            "game_date_utc": start.date.astype(str),
            "home_win": y,
            "market_offset_logit": market_offset_logit,
            "diff_starter_era": diff_starter_era,
            "starting_pitcher_hand_matchup": starting_pitcher_hand_matchup,
            "diff_bullpen_quality": diff_bullpen_quality,
            "diff_lineup_talent": diff_lineup_talent,
            "park_weather_run_effect": park_weather_run_effect,
            "travel_diff": travel_diff,
            "rest_diff": rest_diff,
            "elo_home_prob": elo_home_prob,
        }
    )

    result = {
        "models": {},
        "train_df": train_df,
        "feature_columns": feature_cols,
        "run_payload": {
            "model_run_id": "run_mlb_validation_honesty",
            "feature_set_version": "fset_mlb_validation_honesty",
            "selected_models": ["glm_lasso"],
            "glm_primary_model": "glm_lasso",
            "model_feature_columns": {"glm_lasso": feature_cols},
            "glm_tuning_by_model": {"glm_lasso": {"best_c": 0.7}},
            "credibility_metadata_by_model": {
                "glm_lasso": {
                    "complement_kind": "market",
                    "complement_label": "Vig-free market logit",
                    "complement_column": "market_offset_logit",
                    "offset_scale": "logit",
                    "driver_features": ["diff_starter_era", "diff_bullpen_quality"],
                    "complement_summary": {"source": "vig_free_market"},
                }
            },
        },
    }

    tasks = [task for task in build_validation_tasks() if task.name in {"significance", "fragility"}]
    run_validation_pipeline(result, cfg, tasks=tasks)

    validation_root = tmp_path / "artifacts" / "validation" / "mlb"
    significance = pd.read_csv(validation_root / "diagnostics" / "significance" / "validation_significance.csv")
    assert {
        "market_credibility_block",
        "starting_pitcher_block",
        "bullpen_block",
        "lineup_block",
        "park_weather_block",
        "schedule_travel_block",
        "rating_block",
    }.issubset(set(significance["block"]))
    assert significance["p_value"].isna().all()
    assert significance["inference_applicability"].eq("not_applicable").all()
    assert significance.loc[significance["block"] == "market_credibility_block", "block_role"].eq("complement").all()

    fragility = pd.read_csv(validation_root / "diagnostics" / "fragility" / "validation_fragility_missingness.csv")
    assert fragility["scenario"].tolist() == [
        "market_complement_removed",
        "unknown_starter_context",
        "bullpen_context_removed",
        "lineup_context_removed",
        "park_weather_removed",
    ]
    market_row = fragility.set_index("scenario").loc["market_complement_removed"]
    assert market_row["scenario_status"] == "ok"
    assert market_row["fill_strategy"] == "zero"

    contract_payload = json.loads((validation_root / "validation_outputs_contract.json").read_text())
    records = {record["task_name"]: record for record in contract_payload["validation_outputs"]}
    assert records["significance"]["applicability"] == "partial"
    assert records["significance"]["summary"]["model_family"] == "linear"
    assert records["significance"]["summary"]["model_lane"] == "core"
    assert records["significance"]["summary"]["penalty"]["penalty_family"] == "lasso"
    assert records["significance"]["summary"]["credibility"]["complement_column"] == "market_offset_logit"
    assert records["fragility"]["summary"]["scenario_count"] == 5


def test_validation_pipeline_reports_honest_mlb_calibration_and_classification_applicability(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.data.league = "MLB"

    rng = np.random.default_rng(77)
    n = 180
    start = pd.date_range("2025-03-25", periods=n, freq="D")
    signal = rng.normal(0.0, 1.0, n)
    counter = rng.normal(0.0, 1.0, n)
    logits = 0.9 * signal - 0.4 * counter
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob)
    y[-54:] = 1

    feature_cols = ["signal", "counter"]
    train_df = pd.DataFrame(
        {
            "start_time_utc": start.astype(str),
            "game_date_utc": start.date.astype(str),
            "home_win": y,
            "signal": signal,
            "counter": counter,
        }
    )
    result = {
        "models": {},
        "train_df": train_df,
        "feature_columns": feature_cols,
        "run_payload": {
            "model_run_id": "run_mlb_validation_classification_honesty",
            "feature_set_version": "fset_mlb_validation_classification_honesty",
            "selected_models": ["glm_ridge"],
            "glm_primary_model": "glm_ridge",
            "model_feature_columns": {"glm_ridge": feature_cols},
            "glm_tuning_by_model": {"glm_ridge": {"best_c": 1.0}},
        },
    }

    tasks = [task for task in build_validation_tasks() if task.name in {"calibration", "classification_curves"}]
    run_validation_pipeline(result, cfg, tasks=tasks)

    validation_root = tmp_path / "artifacts" / "validation" / "mlb"
    contract_payload = json.loads((validation_root / "validation_outputs_contract.json").read_text())
    records = {record["task_name"]: record for record in contract_payload["validation_outputs"]}

    calibration_summary = records["calibration"]["summary"]
    classification_summary = records["classification_curves"]["summary"]
    assert records["calibration"]["applicability"] == "partial"
    assert records["classification_curves"]["applicability"] == "partial"
    assert calibration_summary["metric_applicability"]["alpha_beta"] == "not_applicable"
    assert classification_summary["metric_applicability"]["lift_curve"] == "not_applicable"
    assert classification_summary["model_family_applicability"]["status"] == "partial"
    assert classification_summary["model_family_applicability"]["model_family"] == "linear"
    assert classification_summary["model_family_applicability"]["model_lane"] == "core"
    assert classification_summary["holdout_profile"]["has_class_contrast"] is False
    assert np.isnan(calibration_summary["calibration_alpha"])
    assert np.isnan(calibration_summary["calibration_beta"])

    classification_summary_file = json.loads(
        (
            validation_root / "diagnostics" / "classification" / "validation_logit_classification_summary.json"
        ).read_text()
    )
    assert classification_summary_file["metric_applicability"]["lift_curve"] == "not_applicable"
    assert classification_summary_file["model_family_applicability"]["status"] == "partial"


def test_validation_pipeline_writes_mlb_run_metadata_with_scope_and_source_labels(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.data.league = "MLB"

    rng = np.random.default_rng(901)
    n = 120
    start = pd.date_range("2025-03-25", periods=n, freq="D")
    signal = rng.normal(0.0, 1.0, n)
    logits = 0.8 * signal
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob)
    train_df = pd.DataFrame(
        {
            "start_time_utc": start.astype(str),
            "game_date_utc": start.date.astype(str),
            "home_win": y,
            "signal": signal,
        }
    )

    result = {
        "models": {},
        "train_df": train_df,
        "feature_columns": ["signal"],
        "run_payload": {
            "model_run_id": "run_mlb_fixture_demo_metadata",
            "feature_set_version": "fset_mlb_fixture_demo_metadata",
            "selected_models": ["glm_ridge"],
            "metadata": {"execution_metadata": {}},
            "run_contract": {
                "metadata": {
                    "execution_data_scope": "fixture_demo_window",
                    "source_label": "demo_fixture_payload",
                }
            },
        },
    }

    tasks = [task for task in build_validation_tasks() if task.name == "split_summary"]
    run_validation_pipeline(result, cfg, tasks=tasks)

    validation_root = tmp_path / "artifacts" / "validation" / "mlb"
    metadata = json.loads((validation_root / "validation_run_metadata.json").read_text())
    manifest = json.loads((validation_root / "validation_manifest.json").read_text())

    assert validation_root.exists()
    assert not (tmp_path / "artifacts" / "validation" / "validation_manifest.json").exists()
    assert (validation_root / "split" / "validation_split_summary.json").exists()
    assert metadata["league"] == "MLB"
    assert metadata["model_run_id"] == "run_mlb_fixture_demo_metadata"
    assert metadata["selected_models"] == ["glm_ridge"]
    assert metadata["latest_validation_dir"] == "validation/mlb"
    assert metadata["latest_performance_dir"] == "plots/mlb/glm/performance"
    assert metadata["archive_dir"].startswith("validation-runs/mlb/")
    assert (tmp_path / "artifacts" / metadata["archive_dir"]).exists()
    assert metadata["execution_data_scope"] == "fixture_demo_window"
    assert metadata["execution_source_label"] == "demo_fixture_payload"
    assert metadata["execution_label_class"] == "fixture_or_demo"
    assert metadata["execution_metadata"]["execution_data_scope"] == "fixture_demo_window"
    assert metadata["execution_metadata"]["source_label"] == "demo_fixture_payload"
    assert metadata["artifact_counts"]["validation_subdir_files"] >= 1
    assert metadata["artifact_counts"]["total_files"] >= metadata["artifact_counts"]["validation_root_files"]
    assert manifest["league"] == "MLB"
    assert [section["file_name"] for section in manifest["sections"]] == ["split/validation_split_summary.json"]


def test_break_test_trade_deadline_uses_mlb_calendar_cut():
    rng = np.random.default_rng(11)
    n = 180
    dates = pd.date_range("2025-05-01", periods=n, freq="D")
    signal = rng.normal(0.0, 1.0, n)
    logits = 0.9 * signal
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob)
    df = pd.DataFrame(
        {
            "game_date_utc": dates.astype(str),
            "home_win": y,
            "signal": signal,
        }
    )

    report = break_test_trade_deadline(df, features=["signal"], league="MLB", model_name="glm_ridge", c=1.0)

    assert report["deadline"] == "2025-07-31"
    assert report["n_pre"] > 0
    assert report["n_post"] > 0
