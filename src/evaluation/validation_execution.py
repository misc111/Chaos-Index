"""Top-level execution loop for validation tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.common.config import AppConfig
from src.evaluation.validation_artifacts import ValidationOutputs
from src.evaluation.validation_context import ValidationContext
from src.evaluation.validation_contract import build_validation_artifact_record
from src.evaluation.validation_metadata import (
    _archive_validation_outputs,
    _reset_validation_output_dirs,
    _validation_manifest_metadata,
)
from src.evaluation.validation_task_support import (
    ValidationTask,
    ValidationTaskResult,
    _record_task_summary,
    _skipped_task_result,
)
from src.evaluation.validation_tasks import build_validation_tasks

def run_validation_pipeline(
    result: dict[str, Any],
    cfg: AppConfig,
    *,
    tasks: Sequence[ValidationTask] | None = None,
    extra_tasks: Sequence[ValidationTask] | None = None,
) -> ValidationOutputs:
    ctx = ValidationContext.from_result(result, cfg)
    _reset_validation_output_dirs(ctx)
    outputs = ValidationOutputs()
    selected_tasks = list(tasks) if tasks is not None else build_validation_tasks(extra_tasks=extra_tasks)

    for task in selected_tasks:
        if not task.should_run(ctx):
            normalized = _skipped_task_result(ctx, task)
        else:
            task_result = task.runner(ctx)
            if isinstance(task_result, ValidationOutputs):
                if task_result.sections:
                    normalized = ValidationTaskResult(outputs=task_result)
                else:
                    normalized = ValidationTaskResult(
                        outputs=task_result,
                        applicability="not_applicable",
                        summary={"status": "not_applicable", "note": "task_returned_no_artifacts"},
                    )
            else:
                normalized = task_result
        summary = _record_task_summary(
            ctx,
            task_name=task.name,
            task_result=normalized,
        )
        outputs.task_records.append(
            build_validation_artifact_record(
                task_name=task.name,
                family=task.family,
                applicability=normalized.applicability,
                artifacts=[spec.file_name for spec in normalized.outputs.sections],
                summary=summary,
            ).to_dict()
        )
        if normalized.outputs.sections:
            outputs.merge(normalized.outputs)

    outputs.write(
        ctx.out_dir,
        league=ctx.league,
        manifest_metadata=_validation_manifest_metadata(ctx),
    )
    _archive_validation_outputs(ctx, outputs)
    return outputs
