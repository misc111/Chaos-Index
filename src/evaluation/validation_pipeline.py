"""Compatibility facade for validation pipeline orchestration.

Implementation lives in focused sibling modules.  This facade preserves the
historical import surface used by services, scripts, and tests.
"""

from __future__ import annotations

from src.evaluation.validation_artifacts import ValidationOutputs
from src.evaluation.validation_context import ValidationContext
from src.evaluation.validation_execution import run_validation_pipeline
from src.evaluation.validation_metadata import (
    fixture_demo_evidence_labels,
    resolve_execution_metadata,
    resolve_execution_scope_labels,
)
from src.evaluation.validation_models import (
    fit_validation_models,
    resolve_primary_validation_model,
    selected_validation_models,
    validation_feature_columns,
)
from src.evaluation.validation_task_support import ValidationTask, ValidationTaskResult
from src.evaluation.validation_tasks import build_validation_tasks

__all__ = [
    "ValidationContext",
    "ValidationOutputs",
    "ValidationTask",
    "ValidationTaskResult",
    "build_validation_tasks",
    "fixture_demo_evidence_labels",
    "fit_validation_models",
    "resolve_execution_metadata",
    "resolve_execution_scope_labels",
    "resolve_primary_validation_model",
    "run_validation_pipeline",
    "selected_validation_models",
    "validation_feature_columns",
]
