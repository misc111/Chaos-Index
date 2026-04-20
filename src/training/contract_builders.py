from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.registry.models import get_model_registry_entry
from src.training.contracts import (
    CandidateScorecardRecord,
    LassoCredibilityMetadata,
    ModelArtifactRecord,
    ModelRunContract,
    PenaltySelection,
    PromotionDecisionRecord,
    ValidationArtifactRecord,
)


_ACTIVE_COEF_TOLERANCE = 1e-10
_DEFAULT_STATISTICAL_IDENTITY: dict[str, tuple[str, str, str]] = {
    "glm_ridge": ("home_win", "binomial", "logit"),
    "glm_lasso": ("home_win", "binomial", "logit"),
    "glm_elastic_net": ("home_win", "binomial", "logit"),
    "glm_vanilla": ("home_win", "binomial", "logit"),
    "glm_lasso_market_credibility": ("home_win", "binomial", "logit"),
    "glm_lasso_prior_credibility": ("home_win", "binomial", "logit"),
    "gam_spline": ("home_win", "binomial", "logit"),
    "mars_hinge": ("home_win", "binomial", "logit"),
    "glmm_logit": ("home_win", "binomial", "logit"),
    "dglm_margin": ("home_win", "binomial", "logit"),
    "two_stage": ("home_win", "frequency_severity", "mixed"),
    "goals_poisson": ("home_runs", "poisson", "log"),
    "ensemble": ("home_win", "ensemble", "blend"),
}


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if np.isfinite(numeric) else None


def _active_features_from_model(model: object) -> list[str]:
    coef_frame_fn = getattr(model, "coef_frame", None)
    if not callable(coef_frame_fn):
        return []
    try:
        frame = coef_frame_fn()
    except Exception:
        return []
    if frame is None or getattr(frame, "empty", True):
        return []
    coef_column = "coef"
    if coef_column not in frame.columns:
        for candidate in ("coef_original", "coef_scaled"):
            if candidate in frame.columns:
                coef_column = candidate
                break
    if "feature" not in frame.columns or coef_column not in frame.columns:
        return []
    active = frame[np.abs(frame[coef_column].astype(float)) > _ACTIVE_COEF_TOLERANCE]
    return [str(feature) for feature in active["feature"].tolist() if str(feature).strip()]


def build_penalty_selection(
    model_name: str,
    *,
    tuning: Mapping[str, Any] | None = None,
    model: object | None = None,
) -> PenaltySelection | None:
    token = str(model_name or "").strip()
    if token not in {"glm_ridge", "glm_lasso", "glm_elastic_net"}:
        return None
    tuning_dict = dict(tuning or {})
    family = token.removeprefix("glm_")
    active_features = _active_features_from_model(model) if model is not None else []
    return PenaltySelection(
        model_name=token,
        penalty_family=family,
        lambda_value=_safe_float(tuning_dict.get("best_lambda")),
        c_value=_safe_float(tuning_dict.get("best_c")),
        l1_ratio=_safe_float(tuning_dict.get("best_l1_ratio")),
        grid_results=[dict(row) for row in tuning_dict.get("results", []) if isinstance(row, Mapping)],
        fold_metrics=[dict(row) for row in tuning_dict.get("fold_metrics", []) if isinstance(row, Mapping)],
        active_parameter_count=len(active_features) if active_features else None,
        active_features=active_features,
    )


def build_lasso_credibility_metadata(
    model_name: str,
    *,
    model: object | None = None,
    credibility_payload: Mapping[str, Any] | None = None,
) -> LassoCredibilityMetadata | None:
    payload = dict(credibility_payload or {})
    if model is not None:
        model_payload = getattr(model, "credibility_metadata", None)
        if isinstance(model_payload, Mapping):
            payload = {**payload, **dict(model_payload)}
    if not payload:
        return None
    complement_kind = str(payload.get("complement_kind") or "").strip()
    complement_column = str(payload.get("complement_column") or "").strip()
    if not complement_kind or not complement_column:
        return None
    return LassoCredibilityMetadata(
        model_name=model_name,
        complement_kind=complement_kind,
        complement_label=str(payload.get("complement_label") or complement_column),
        complement_column=complement_column,
        offset_scale=str(payload.get("offset_scale") or "logit"),
        driver_features=[str(value) for value in payload.get("driver_features", []) if str(value).strip()],
        complement_summary=dict(payload.get("complement_summary") or {}),
        relativity_summary=dict(payload.get("relativity_summary") or {}),
        exposure_summary=dict(payload.get("exposure_summary") or {}),
        lambda_choice_note=str(payload.get("lambda_choice_note") or ""),
        complement_choice_note=str(payload.get("complement_choice_note") or ""),
        p_values_reported=bool(payload.get("p_values_reported", False)),
    )


def build_model_artifact_record(
    model_name: str,
    *,
    feature_columns: list[str],
    artifact_files: Mapping[str, str] | None = None,
    metrics_summary: Mapping[str, Any] | None = None,
    fit_summary: Mapping[str, Any] | None = None,
    penalty_tuning: Mapping[str, Any] | None = None,
    penalty_selection: PenaltySelection | None = None,
    credibility_payload: Mapping[str, Any] | None = None,
    model: object | None = None,
) -> ModelArtifactRecord:
    if model_name == "ensemble":
        model_family = "ensemble"
        lane = "derived"
    else:
        entry = get_model_registry_entry(model_name)
        model_family = entry.family
        lane = entry.lane
    target_name, distribution, link_function = _DEFAULT_STATISTICAL_IDENTITY.get(model_name, ("", "", ""))
    if model is not None:
        target_name = str(getattr(model, "target_name", target_name) or target_name)
        distribution = str(getattr(model, "distribution", distribution) or distribution)
        link_function = str(getattr(model, "link_function", link_function) or link_function)
    credibility = build_lasso_credibility_metadata(
        model_name,
        model=model,
        credibility_payload=credibility_payload,
    )
    offset_columns = [credibility.complement_column] if credibility is not None and credibility.complement_column else []
    return ModelArtifactRecord(
        model_name=model_name,
        model_family=model_family,
        lane=lane,
        target_name=target_name,
        distribution=distribution,
        link_function=link_function,
        offset_columns=offset_columns,
        artifact_files=dict(artifact_files or {}),
        feature_columns=list(feature_columns),
        metrics_summary=dict(metrics_summary or {}),
        fit_summary=dict(fit_summary or {}),
        penalty=penalty_selection or build_penalty_selection(model_name, tuning=penalty_tuning, model=model),
        credibility=credibility,
    )


def build_model_run_contract(
    *,
    model_run_id: str,
    league: str,
    feature_set_version: str,
    selected_models: list[str],
    feature_columns: list[str],
    model_feature_columns: dict[str, list[str]],
    model_artifacts: list[ModelArtifactRecord] | None = None,
    validation_outputs: list[ValidationArtifactRecord] | None = None,
    candidate_scorecards: list[CandidateScorecardRecord] | None = None,
    promotion_decision: PromotionDecisionRecord | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ModelRunContract:
    return ModelRunContract(
        model_run_id=model_run_id,
        league=league,
        feature_set_version=feature_set_version,
        selected_models=list(selected_models),
        feature_columns=list(feature_columns),
        model_feature_columns={str(key): list(value) for key, value in (model_feature_columns or {}).items()},
        model_artifacts=list(model_artifacts or []),
        validation_outputs=list(validation_outputs or []),
        candidate_scorecards=list(candidate_scorecards or []),
        promotion_decision=promotion_decision,
        metadata=dict(metadata or {}),
    )
