from __future__ import annotations

import pandas as pd

from src.evaluation.validation_metadata import (
    fixture_demo_evidence_labels,
    resolve_execution_metadata,
    resolve_execution_scope_labels,
)
from src.evaluation.validation_models import selected_validation_models, validation_feature_columns
from src.evaluation.validation_tasks import build_validation_tasks


def test_validation_metadata_helpers_preserve_fixture_demo_evidence_labels():
    """Metadata extraction remains independent from validation task execution."""

    run_payload = {
        "metadata": {"execution_metadata": {}},
        "run_contract": {
            "metadata": {
                "execution_data_scope": "fixture_demo_window",
                "source_label": "demo_fixture_payload",
                "production_grade": False,
            }
        },
    }

    execution_metadata = resolve_execution_metadata(run_payload)
    scope_labels = resolve_execution_scope_labels(run_payload)
    evidence_labels = fixture_demo_evidence_labels(scope_labels)

    assert execution_metadata["execution_data_scope"] == "fixture_demo_window"
    assert execution_metadata["source_label"] == "demo_fixture_payload"
    assert scope_labels == {
        "execution_data_scope": "fixture_demo_window",
        "execution_source_label": "demo_fixture_payload",
        "execution_label_class": "fixture_or_demo",
    }
    assert evidence_labels["evidence_grade"] == "non_production_fixture_or_demo"


def test_validation_model_helpers_preserve_alias_and_feature_resolution():
    """Model fitting helpers expose pure selection behavior for focused tests."""

    result = {"models": {}, "train_df": pd.DataFrame(), "feature_columns": ["signal", "other"]}
    run_payload = {
        "selected_models": ["glm_ridge", "glm_lasso"],
        "model_feature_columns": {"glm_ridge": ["signal", "missing"], "glm_lasso": ["other"]},
    }

    selected = selected_validation_models(result, run_payload)

    assert selected == ["glm_ridge", "glm_lasso"]
    assert validation_feature_columns(["signal", "other"], run_payload, "glm_ridge") == ["signal"]


def test_validation_tasks_registry_is_available_without_pipeline_importing_tasks():
    """The task registry can be imported as its own orchestration surface."""

    task_names = [task.name for task in build_validation_tasks()]

    assert task_names[:3] == ["split_summary", "mlb_data_quality", "glm_diagnostics"]
    assert "market_truth" in task_names
