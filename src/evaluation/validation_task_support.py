"""Shared task contracts and applicability helpers for validation diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.evaluation.validation_artifacts import ValidationOutputs
from src.evaluation.validation_context import ValidationContext, holdout_df as _holdout_df
from src.evaluation.validation_groups import build_feature_blocks
from src.registry.models import planned_credibility_model_catalog

ValidationTaskRunner = Callable[["ValidationContext"], "ValidationOutputs | ValidationTaskResult"]
ValidationTaskPredicate = Callable[["ValidationContext"], bool]


@dataclass(slots=True)
class ValidationTaskResult:
    outputs: ValidationOutputs = field(default_factory=ValidationOutputs)
    applicability: str = "applicable"
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ValidationTask:
    name: str
    runner: ValidationTaskRunner
    enabled: ValidationTaskPredicate | None = None
    family: str = "diagnostics"

    def should_run(self, ctx: ValidationContext) -> bool:
        return True if self.enabled is None else bool(self.enabled(ctx))


def _has_glm(ctx: ValidationContext) -> bool:
    return ctx.glm is not None


def _has_holdout(ctx: ValidationContext) -> bool:
    return not _holdout_df(ctx).empty


def _feature_blocks_for(ctx: ValidationContext) -> dict[str, list[str]]:
    return build_feature_blocks(
        ctx.league,
        ctx.diagnostic_feature_cols,
        credibility=ctx.glm_validation_metadata.credibility,
    )


_PENALIZED_VALIDATION_MODEL_KEYS = {"glm_ridge", "glm_lasso", "glm_elastic_net"}
_CREDIBILITY_VALIDATION_MODEL_KEYS = {"glm_lasso_market_credibility", "glm_lasso_prior_credibility"}
_EXTENSION_VALIDATION_MODEL_KEYS = {"gam_spline", "glmm_logit", "dglm_margin", "goals_poisson"}


def _primary_model_key(ctx: ValidationContext) -> str:
    return str(ctx.glm_validation_metadata.model_key or ctx.glm_validation_metadata.model_name or "").strip()


def _is_penalized_primary_model(ctx: ValidationContext) -> bool:
    return _primary_model_key(ctx) in _PENALIZED_VALIDATION_MODEL_KEYS


def _is_credibility_primary_model(ctx: ValidationContext) -> bool:
    return _primary_model_key(ctx) in _CREDIBILITY_VALIDATION_MODEL_KEYS or ctx.glm_validation_metadata.credibility is not None


def _is_vanilla_primary_model(ctx: ValidationContext) -> bool:
    return _primary_model_key(ctx) == "glm_vanilla"


def _is_extension_primary_model(ctx: ValidationContext) -> bool:
    return _primary_model_key(ctx) in _EXTENSION_VALIDATION_MODEL_KEYS or str(ctx.glm_validation_metadata.model_lane or "") == "extension"


def _task_support_payload(
    ctx: ValidationContext,
    *,
    task_name: str,
    support_status: str,
    route: str,
    note: str,
) -> dict[str, Any]:
    return {
        "task_name": task_name,
        "task_support_status": support_status,
        "task_support_route": route,
        "task_support_note": note,
        "primary_model_resolution": dict(ctx.primary_model_resolution),
    }


def _unsupported_task_result(
    ctx: ValidationContext,
    *,
    task_name: str,
    note: str,
    applicability: str = "not_applicable",
    route: str = "unsupported_for_selected_model_family",
) -> ValidationTaskResult:
    return _task_result(
        ctx,
        ValidationOutputs(),
        summary=_task_support_payload(
            ctx,
            task_name=task_name,
            support_status=applicability,
            route=route,
            note=note,
        ),
        applicability=applicability,
    )


def _validation_surrogate_penalized_spec(
    ctx: ValidationContext,
) -> tuple[str | None, float | None, float | None, dict[str, Any]]:
    model_key = _primary_model_key(ctx)
    if _is_penalized_primary_model(ctx):
        return (
            model_key,
            float(getattr(ctx.glm, "c", 1.0)) if ctx.glm is not None else 1.0,
            getattr(ctx.glm, "l1_ratio", None) if ctx.glm is not None else None,
            _task_support_payload(
                ctx,
                task_name="penalized_surrogate",
                support_status="applicable",
                route="selected_model_direct",
                note="Using the selected penalized GLM directly for penalized-family diagnostics.",
            ),
        )

    if _is_credibility_primary_model(ctx):
        driver_model = "glm_lasso"
        catalog_payload = planned_credibility_model_catalog().get(model_key, {})
        planned_driver = str(catalog_payload.get("driver_source_model") or "").strip()
        if planned_driver in _PENALIZED_VALIDATION_MODEL_KEYS:
            driver_model = planned_driver
        credibility_tuning: Mapping[str, Any] | None = None
        for key in ("credibility_tuning_by_model", "lasso_credibility_tuning_by_model"):
            payload = ctx.run_payload.get(key)
            if isinstance(payload, Mapping):
                candidate = payload.get(model_key)
                if isinstance(candidate, Mapping):
                    credibility_tuning = candidate
                    break
        surrogate_c = 1.0
        if credibility_tuning is not None:
            surrogate_c = float(credibility_tuning.get("best_c", credibility_tuning.get("c", 1.0)) or 1.0)
        if not np.isfinite(surrogate_c) or surrogate_c <= 0.0:
            surrogate_c = 1.0
        return (
            driver_model,
            surrogate_c,
            None,
            _task_support_payload(
                ctx,
                task_name="penalized_surrogate",
                support_status="partial",
                route="credibility_driver_surrogate",
                note=(
                    "Using the credibility lane's penalized driver model for diagnostics that do not "
                    "support offsets or credibility shrinkage directly."
                ),
            ),
        )

    return None, None, None, _task_support_payload(
        ctx,
        task_name="penalized_surrogate",
        support_status="not_applicable",
        route="no_penalized_surrogate_available",
        note="No valid penalized-GLM surrogate is available for the selected validation primary model.",
    )


def _binary_holdout_profile(y_true: np.ndarray | pd.Series) -> dict[str, Any]:
    y_raw = np.asarray(y_true, dtype=float)
    y = y_raw[np.isfinite(y_raw)].astype(int)
    n_obs = int(len(y))
    if n_obs <= 0:
        return {
            "n_obs": 0,
            "positive_count": 0,
            "negative_count": 0,
            "positive_rate": float("nan"),
            "has_class_contrast": False,
            "overall_applicability": "not_applicable",
        }

    positive_count = int(np.sum(y))
    negative_count = max(0, n_obs - positive_count)
    has_class_contrast = positive_count > 0 and negative_count > 0
    return {
        "n_obs": n_obs,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": float(positive_count / n_obs),
        "has_class_contrast": bool(has_class_contrast),
        "overall_applicability": "applicable" if has_class_contrast else "partial",
    }


def _probability_model_family_applicability(
    ctx: ValidationContext,
    *,
    task_name: str,
    requires_class_contrast: bool,
    class_contrast_available: bool,
    n_obs: int,
) -> dict[str, Any]:
    if ctx.glm is None:
        resolution_status = str(ctx.primary_model_resolution.get("status") or "")
        if resolution_status in {"partial", "not_supported"}:
            status = "pending"
        else:
            status = "not_applicable"
    elif n_obs <= 0:
        status = "not_applicable"
    elif requires_class_contrast and not class_contrast_available:
        status = "partial"
    else:
        status = "applicable"
    return {
        "task": task_name,
        "status": status,
        "model_family": str(ctx.glm_validation_metadata.model_family or "unknown"),
        "model_lane": str(ctx.glm_validation_metadata.model_lane or "unknown"),
        "model_key": str(ctx.glm_validation_metadata.model_key or ""),
        "requires_class_contrast": bool(requires_class_contrast),
        "class_contrast_available": bool(class_contrast_available),
        "probability_output_method": "predict_proba" if ctx.glm is not None and hasattr(ctx.glm, "predict_proba") else "missing",
        "primary_model_resolution": dict(ctx.primary_model_resolution),
    }


def _task_summary_with_model_metadata(ctx: ValidationContext, payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(payload)
    summary |= ctx.glm_validation_metadata.to_summary()
    return summary


def _task_result(
    ctx: ValidationContext,
    outputs: ValidationOutputs,
    *,
    summary: dict[str, Any] | None = None,
    applicability: str = "applicable",
) -> ValidationTaskResult:
    resolved_summary = _task_summary_with_model_metadata(ctx, summary or {}) if summary else {}
    return ValidationTaskResult(outputs=outputs, applicability=applicability, summary=resolved_summary)


_MODEL_DEPENDENT_TASKS = {
    "glm_diagnostics",
    "stability",
    "influence",
    "fragility",
    "calibration",
    "market_truth",
    "classification_curves",
}
_HOLDOUT_DEPENDENT_TASKS = {
    "nonlinearity",
    "significance",
    "permutation_importance",
    "calibration",
    "market_truth",
    "classification_curves",
}


def _default_model_family_applicability(
    ctx: ValidationContext,
    *,
    task_name: str,
    contract_applicability: str,
) -> dict[str, Any]:
    if task_name in _MODEL_DEPENDENT_TASKS:
        resolution_status = str(ctx.primary_model_resolution.get("status") or "")
        if ctx.glm is not None:
            status = "applicable" if contract_applicability == "applicable" else contract_applicability
        elif resolution_status in {"partial", "not_supported"}:
            status = "pending"
        else:
            status = "not_applicable"
        reason = (
            str(ctx.primary_model_resolution.get("note") or "")
            if status in {"pending", "partial"}
            else "primary_model_available"
        )
    else:
        status = "not_applicable"
        reason = "task_not_model_family_specific"

    return {
        "task": task_name,
        "status": status,
        "reason": reason,
        "model_family": str(ctx.glm_validation_metadata.model_family or "unknown"),
        "model_lane": str(ctx.glm_validation_metadata.model_lane or "unknown"),
        "model_key": str(ctx.glm_validation_metadata.model_key or ""),
        "primary_model_resolution": dict(ctx.primary_model_resolution),
    }


def _record_task_summary(
    ctx: ValidationContext,
    *,
    task_name: str,
    task_result: ValidationTaskResult,
) -> dict[str, Any]:
    summary = dict(task_result.summary or {})
    model_family_applicability = summary.get("model_family_applicability")
    if not isinstance(model_family_applicability, Mapping):
        model_family_applicability = _default_model_family_applicability(
            ctx,
            task_name=task_name,
            contract_applicability=task_result.applicability,
        )
        summary["model_family_applicability"] = model_family_applicability
    summary["contract_applicability"] = str(task_result.applicability)
    summary["applicability_split"] = {
        "contract_level": str(task_result.applicability),
        "model_family_level": str(model_family_applicability.get("status") or "not_applicable"),
    }
    return _task_summary_with_model_metadata(ctx, summary)


def _skipped_task_result(ctx: ValidationContext, task: "ValidationTask") -> ValidationTaskResult:
    if task.name in _MODEL_DEPENDENT_TASKS and ctx.glm is None:
        resolution_status = str(ctx.primary_model_resolution.get("status") or "")
        applicability = "pending" if resolution_status in {"partial", "not_supported"} else "not_applicable"
        reason = "primary_validation_model_unavailable"
    elif task.name in _HOLDOUT_DEPENDENT_TASKS and _holdout_df(ctx).empty:
        applicability = "not_applicable"
        reason = "holdout_split_unavailable"
    else:
        applicability = "pending"
        reason = "task_predicate_not_met"
    summary = {
        "status": "skipped",
        "skip_reason": reason,
        "primary_model_resolution": dict(ctx.primary_model_resolution),
    }
    return ValidationTaskResult(outputs=ValidationOutputs(), applicability=applicability, summary=_task_summary_with_model_metadata(ctx, summary))
