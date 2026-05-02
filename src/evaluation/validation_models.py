"""Validation model selection, fitting, and primary-model resolution.

This module owns the model-family plumbing used by validation reports.  It keeps
fitting decisions separate from artifact writing and task execution so each lane
can be tested and reasoned about independently.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.models.experimental.gbdt import GBDTModel
from src.models.experimental.rf import RFModel
from src.models.core.glm_penalized import build_penalized_glm
from src.models.core.glm_vanilla import VanillaGLMModel
from src.models.core.lasso_credibility import LassoCredibilityModel
from src.models.extensions.glm_goals import GoalsPoissonModel
from src.models.extensions.glm_variants import DGLMMarginModel, GAMSplineModel, GLMMLogitModel
from src.registry.models import planned_credibility_model_catalog
from src.training.lasso_credibility import selected_lasso_credibility_models
from src.training.model_catalog import LEGACY_MODEL_KEYS, MODEL_ALIASES
from src.training.penalized_glm import selected_penalized_glm_models
from src.training.tune import quick_tune_penalized_glm

def _selected_validation_models(result: dict[str, Any], run_payload: dict[str, Any]) -> list[str]:
    def _canonical_model_name(model_name: Any) -> str:
        token = str(model_name).strip().lower()
        if not token:
            return ""
        canonical = MODEL_ALIASES.get(token, token)
        if not canonical:
            return ""
        token = str(canonical).strip()
        return token

    payload_models = run_payload.get("selected_models", [])
    if isinstance(payload_models, list):
        selected = [_canonical_model_name(model) for model in payload_models if str(model).strip()]
        if selected:
            return selected

    raw_models = result.get("models", {})
    if isinstance(raw_models, dict):
        return [_canonical_model_name(model) for model in raw_models.keys() if str(model).strip()]

    return []

def _validation_feature_columns(
    feature_cols: list[str],
    run_payload: dict[str, Any],
    model_name: str,
) -> list[str]:
    model_feature_map = run_payload.get("model_feature_columns", {})
    if isinstance(model_feature_map, dict):
        candidate_keys = [model_name]
        candidate_keys.extend(LEGACY_MODEL_KEYS.get(model_name, ()))
        for key in candidate_keys:
            requested = model_feature_map.get(key, [])
            if isinstance(requested, list):
                resolved = [str(col) for col in requested if str(col) in feature_cols]
                if resolved:
                    return resolved

    if model_name in {"glm_ridge", "glm_lasso", "glm_elastic_net", "glm_vanilla"}:
        glm_feature_columns = run_payload.get("glm_feature_columns", [])
        if isinstance(glm_feature_columns, list):
            resolved = [str(col) for col in glm_feature_columns if str(col) in feature_cols]
            if resolved:
                return resolved

    return list(feature_cols)


def _fit_validation_models(
    tr: pd.DataFrame,
    *,
    feature_cols: list[str],
    selected_models: list[str],
    run_payload: dict[str, Any],
) -> tuple[dict[str, object], dict[str, dict[str, Any]]]:
    fit_status: dict[str, dict[str, Any]] = {}
    selected = [str(model) for model in selected_models if str(model).strip()]
    if tr.empty:
        for model_name in selected:
            fit_status[model_name] = {"status": "not_applicable", "reason": "empty_training_frame"}
        return {}, fit_status

    selected_set = set(selected)
    models: dict[str, object] = {}
    credibility_catalog = planned_credibility_model_catalog()

    def _mark_status(model_name: str, *, status: str, **payload: Any) -> None:
        fit_status[str(model_name)] = {"status": str(status), **payload}

    for model_name in selected_penalized_glm_models(selected):
        if model_name not in selected_set:
            continue
        glm_cols = _validation_feature_columns(feature_cols, run_payload, model_name)
        try:
            glm_tuning_by_model = run_payload.get("glm_tuning_by_model")
            if isinstance(glm_tuning_by_model, Mapping):
                glm_tune = dict(glm_tuning_by_model.get(model_name, {}))
            else:
                glm_tune = {}
            if "start_time_utc" in tr.columns:
                glm_tune = quick_tune_penalized_glm(
                    tr,
                    glm_cols,
                    model_name=model_name,
                    n_splits=3,
                    min_train_size=min(140, max(70, len(tr) // 2)),
                )
            glm = build_penalized_glm(
                model_name,
                c=float(glm_tune.get("best_c", 1.0)),
                l1_ratio=glm_tune.get("best_l1_ratio"),
            )
            glm.fit(tr, glm_cols)
            models[glm.model_name] = glm
            _mark_status(model_name, status="fit_ok", model_kind="penalized_glm", feature_count=len(glm_cols))
        except Exception as exc:
            _mark_status(
                model_name,
                status="fit_failed",
                model_kind="penalized_glm",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    credibility_tuning_by_model: dict[str, Any] = {}
    for key in ("credibility_tuning_by_model", "lasso_credibility_tuning_by_model"):
        payload = run_payload.get(key)
        if isinstance(payload, Mapping):
            credibility_tuning_by_model |= dict(payload)
    for model_name in selected_lasso_credibility_models(selected):
        if model_name not in selected_set:
            continue
        payload = dict(credibility_catalog.get(model_name, {}))
        if not payload:
            _mark_status(
                model_name,
                status="not_supported",
                model_kind="lasso_credibility",
                reason="missing_credibility_catalog_payload",
            )
            continue
        params = dict(credibility_tuning_by_model.get(model_name, {}))
        lambda_value = float(params.get("best_lambda", params.get("lambda_value", 1.0)) or 1.0)
        if not np.isfinite(lambda_value) or lambda_value <= 0:
            lambda_value = 1.0
        credibility_cols = _validation_feature_columns(feature_cols, run_payload, model_name)
        try:
            model = LassoCredibilityModel(
                model_name=model_name,
                lambda_value=lambda_value,
                complement_kind=str(payload.get("complement_kind") or ""),
                complement_column=str(payload.get("complement_column") or ""),
                complement_label=str(payload.get("complement_label") or ""),
                complement_input_scale=str(payload.get("offset_scale") or "logit"),
            )
            model.fit(tr, credibility_cols)
            models[model_name] = model
            _mark_status(model_name, status="fit_ok", model_kind="lasso_credibility", feature_count=len(credibility_cols))
        except Exception as exc:
            _mark_status(
                model_name,
                status="fit_failed",
                model_kind="lasso_credibility",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    extension_builders: dict[str, type] = {
        "glm_vanilla": VanillaGLMModel,
        "gam_spline": GAMSplineModel,
        "glmm_logit": GLMMLogitModel,
        "dglm_margin": DGLMMarginModel,
    }
    for model_name, model_cls in extension_builders.items():
        if model_name not in selected_set:
            continue
        model_cols = _validation_feature_columns(feature_cols, run_payload, model_name)
        try:
            model = model_cls()
            model.fit(tr, model_cols)
            models[model_name] = model
            _mark_status(model_name, status="fit_ok", model_kind="extension_or_vanilla", feature_count=len(model_cols))
        except Exception as exc:
            _mark_status(
                model_name,
                status="fit_failed",
                model_kind="extension_or_vanilla",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    if "goals_poisson" in selected_set:
        try:
            model = GoalsPoissonModel()
            model.fit(tr)
            models[model.model_name] = model
            _mark_status("goals_poisson", status="fit_ok", model_kind="extension_or_vanilla", feature_count=0)
        except Exception as exc:
            _mark_status(
                "goals_poisson",
                status="fit_failed",
                model_kind="extension_or_vanilla",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    for model_name, model_cls in (("gbdt", GBDTModel), ("rf", RFModel)):
        if model_name not in selected_set:
            continue
        model_cols = _validation_feature_columns(feature_cols, run_payload, model_name)
        try:
            model = model_cls()
            model.fit(tr, model_cols)
            models[model_name] = model
            _mark_status(model_name, status="fit_ok", model_kind="ml_baseline", feature_count=len(model_cols))
        except Exception as exc:
            _mark_status(
                model_name,
                status="fit_failed",
                model_kind="ml_baseline",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    for model_name in selected:
        fit_status.setdefault(model_name, {"status": "not_supported", "reason": "validation_wrapper_unavailable"})

    return models, fit_status


def _resolve_primary_validation_model(
    *,
    selected_models: list[str],
    models: Mapping[str, object],
    model_fit_status: Mapping[str, Mapping[str, Any]],
    run_payload: Mapping[str, Any],
) -> tuple[str | None, object | None, dict[str, Any]]:
    preferred_default_order = [
        "glm_ridge",
        "glm_elastic_net",
        "glm_lasso",
        "glm_lasso_market_credibility",
        "glm_lasso_prior_credibility",
        "glm_vanilla",
        "gam_spline",
        "glmm_logit",
        "dglm_margin",
        "goals_poisson",
        "gbdt",
        "rf",
    ]
    requested_primary_raw = str(run_payload.get("glm_primary_model") or "").strip().lower()
    requested_primary = MODEL_ALIASES.get(requested_primary_raw, requested_primary_raw) if requested_primary_raw else ""
    candidate_order: list[str] = []
    if requested_primary:
        candidate_order.append(str(requested_primary))
    candidate_order.extend([str(name) for name in selected_models if str(name).strip()])
    candidate_order.extend(preferred_default_order)

    resolved_model_name: str | None = None
    resolved_model: object | None = None
    for model_name in list(dict.fromkeys(candidate_order)):
        model = models.get(model_name)
        if model is None or not callable(getattr(model, "predict_proba", None)):
            continue
        resolved_model_name = model_name
        resolved_model = model
        break

    if resolved_model_name is None:
        resolution_status = "not_supported" if selected_models else "not_applicable"
        resolution_note = "No selected validation model could be fit with a probability interface."
    elif requested_primary and resolved_model_name != requested_primary:
        resolution_status = "partial"
        resolution_note = (
            f"Requested primary model '{requested_primary}' was unavailable for validation; "
            f"fell back to '{resolved_model_name}'."
        )
    else:
        resolution_status = "supported"
        resolution_note = f"Resolved primary validation model '{resolved_model_name}'."

    resolution = {
        "status": resolution_status,
        "note": resolution_note,
        "requested_primary_model": requested_primary or None,
        "resolved_primary_model": resolved_model_name,
        "selected_models": [str(name) for name in selected_models if str(name).strip()],
        "model_fit_status": {str(key): dict(value) for key, value in model_fit_status.items()},
    }
    return resolved_model_name, resolved_model, resolution


selected_validation_models = _selected_validation_models
validation_feature_columns = _validation_feature_columns
fit_validation_models = _fit_validation_models
resolve_primary_validation_model = _resolve_primary_validation_model
