from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.evaluation.validation_pipeline import build_validation_tasks, run_validation_pipeline


def test_validation_outputs_contract_carries_governance_scope_and_split_labels(tmp_path):
    """Every validation task record should say what kind of evidence it is.

    The validation output contract is read by training persistence, staging, and
    dashboard surfaces.  A task-level record that only says "calibration" or
    "market_truth" leaves readers to infer whether the evidence is CAS-theory
    diagnostics, engineering support, or a betting overlay.  This regression
    test pins those labels at the central pipeline boundary.
    """

    cfg = load_config("configs/default.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.data.league = "MLB"

    rng = np.random.default_rng(9017)
    dates = pd.date_range("2025-03-25", periods=96, freq="D")
    signal = rng.normal(0.0, 1.0, len(dates))
    home_win = rng.binomial(1, 1.0 / (1.0 + np.exp(-signal)))
    train_df = pd.DataFrame(
        {
            "game_id": np.arange(1, len(dates) + 1),
            "start_time_utc": dates.astype(str),
            "game_date_utc": dates.date.astype(str),
            "home_win": home_win,
            "signal": signal,
        }
    )
    result = {
        "models": {},
        "train_df": train_df,
        "feature_columns": ["signal"],
        "run_payload": {
            "model_run_id": "run_governance_labels",
            "feature_set_version": "fset_governance_labels",
            "selected_models": ["glm_ridge"],
            "glm_primary_model": "glm_ridge",
            "model_feature_columns": {"glm_ridge": ["signal"]},
            "target_name": "moneyline_home_win",
            "target_col": "home_win",
        },
    }

    tasks = [task for task in build_validation_tasks() if task.name in {"split_summary", "calibration"}]
    outputs = run_validation_pipeline(result, cfg, tasks=tasks)

    records = {record["task_name"]: record for record in outputs.task_records}
    split_record = records["split_summary"]
    calibration_record = records["calibration"]

    assert split_record["governance_contract_version"] == 1
    assert split_record["validation_lane"] == "theory_compatible_engineering"
    assert split_record["theory_governance"] == "theory-compatible engineering support"
    assert split_record["split_label"] == "train_test:time"
    assert split_record["target_scope"] == {
        "league": "MLB",
        "target_name": "moneyline_home_win",
        "target_column": "home_win",
        "market": "moneyline",
        "product_scope": ["moneyline", "runline", "totals"],
    }

    assert calibration_record["validation_lane"] == "theory_core_diagnostics"
    assert calibration_record["theory_governance"] == "direct theory implementation"
    assert calibration_record["model_lane"] == "core"
    assert calibration_record["theory_classification"] == "core-supported"
    assert calibration_record["summary"]["validation_lane"] == "theory_core_diagnostics"
    assert calibration_record["summary"]["theory_classification"] == "core-supported"

    validation_root = tmp_path / "artifacts" / "validation" / "mlb"
    persisted = json.loads((validation_root / "validation_outputs_contract.json").read_text())
    persisted_records = {record["task_name"]: record for record in persisted["validation_outputs"]}
    assert persisted_records["calibration"]["target_scope"] == calibration_record["target_scope"]


def test_validation_task_registry_declares_explicit_governance_lanes():
    """The task registry should be the single declaration point for evidence lanes."""

    tasks = {task.name: task for task in build_validation_tasks()}

    assert tasks["split_summary"].validation_lane == "theory_compatible_engineering"
    assert tasks["split_summary"].theory_governance == "theory-compatible engineering support"
    assert tasks["glm_diagnostics"].validation_lane == "theory_core_diagnostics"
    assert tasks["glm_diagnostics"].theory_governance == "direct theory implementation"
    assert tasks["market_truth"].validation_lane == "betting_overlay"
    assert tasks["market_truth"].theory_governance == "betting overlay"
