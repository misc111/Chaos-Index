"""Validation task registry facade.

The individual task implementations live in smaller modules by diagnostic family;
this module composes them into the canonical execution order.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.evaluation.validation_task_support import ValidationTask, _has_glm, _has_holdout
from src.evaluation.validation_governance import (
    BETTING_OVERLAY_TASK_GOVERNANCE,
    ENGINEERING_TASK_GOVERNANCE,
    THEORY_CORE_TASK_GOVERNANCE,
    ValidationTaskGovernance,
)
from src.evaluation.validation_tasks_core import (
    _task_collinearity,
    _task_glm_diagnostics,
    _task_mlb_data_quality,
    _task_nonlinearity,
    _task_permutation_importance,
    _task_split_summary,
)
from src.evaluation.validation_tasks_model import _task_fragility, _task_influence, _task_significance, _task_stability
from src.evaluation.validation_tasks_probability import _task_calibration, _task_classification_curves, _task_market_truth

def _governance_kwargs(governance: ValidationTaskGovernance) -> dict[str, str]:
    return {
        "validation_lane": governance.validation_lane,
        "theory_governance": governance.theory_governance,
    }


def build_validation_tasks(
    *,
    extra_tasks: Sequence[ValidationTask] | None = None,
) -> list[ValidationTask]:
    tasks = [
        ValidationTask(name="split_summary", runner=_task_split_summary, family="split", **_governance_kwargs(ENGINEERING_TASK_GOVERNANCE)),
        ValidationTask(
            name="mlb_data_quality",
            runner=_task_mlb_data_quality,
            enabled=lambda ctx: ctx.league == "MLB",
            family="data_quality",
            **_governance_kwargs(ENGINEERING_TASK_GOVERNANCE),
        ),
        ValidationTask(
            name="glm_diagnostics",
            runner=_task_glm_diagnostics,
            enabled=_has_glm,
            family="glm_diagnostics",
            **_governance_kwargs(THEORY_CORE_TASK_GOVERNANCE),
        ),
        ValidationTask(
            name="permutation_importance",
            runner=_task_permutation_importance,
            enabled=_has_holdout,
            family="permutation_importance",
            **_governance_kwargs(ENGINEERING_TASK_GOVERNANCE),
        ),
        ValidationTask(
            name="collinearity",
            runner=_task_collinearity,
            family="collinearity",
            **_governance_kwargs(THEORY_CORE_TASK_GOVERNANCE),
        ),
        ValidationTask(
            name="nonlinearity",
            runner=_task_nonlinearity,
            enabled=_has_holdout,
            family="nonlinearity",
            **_governance_kwargs(THEORY_CORE_TASK_GOVERNANCE),
        ),
        ValidationTask(
            name="significance",
            runner=_task_significance,
            enabled=lambda ctx: _has_holdout(ctx) and not (ctx.fit_df if not ctx.fit_df.empty else ctx.tr).empty,
            family="significance",
            **_governance_kwargs(THEORY_CORE_TASK_GOVERNANCE),
        ),
        ValidationTask(
            name="stability",
            runner=_task_stability,
            enabled=_has_glm,
            family="stability",
            **_governance_kwargs(THEORY_CORE_TASK_GOVERNANCE),
        ),
        ValidationTask(
            name="influence",
            runner=_task_influence,
            enabled=_has_glm,
            family="influence",
            **_governance_kwargs(THEORY_CORE_TASK_GOVERNANCE),
        ),
        ValidationTask(
            name="fragility",
            runner=_task_fragility,
            enabled=_has_glm,
            family="fragility",
            **_governance_kwargs(THEORY_CORE_TASK_GOVERNANCE),
        ),
        ValidationTask(
            name="calibration",
            runner=_task_calibration,
            enabled=lambda ctx: _has_glm(ctx) and _has_holdout(ctx),
            family="calibration",
            **_governance_kwargs(THEORY_CORE_TASK_GOVERNANCE),
        ),
        ValidationTask(
            name="market_truth",
            runner=_task_market_truth,
            enabled=lambda ctx: ctx.league == "MLB" and _has_glm(ctx) and _has_holdout(ctx),
            family="market_truth",
            **_governance_kwargs(BETTING_OVERLAY_TASK_GOVERNANCE),
        ),
        ValidationTask(
            name="classification_curves",
            runner=_task_classification_curves,
            enabled=lambda ctx: _has_glm(ctx) and _has_holdout(ctx),
            family="classification",
            **_governance_kwargs(THEORY_CORE_TASK_GOVERNANCE),
        ),
    ]
    if extra_tasks:
        tasks.extend(extra_tasks)
    return tasks
