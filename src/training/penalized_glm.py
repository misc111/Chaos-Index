from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.models.glm_penalized import PENALIZED_GLM_MODEL_NAMES, penalized_glm_config
from src.training.contracts import PenaltySelection
from src.training.feature_selection import glm_feature_subset, resolve_model_feature_columns
from src.training.lambda_search import penalty_choice
from src.training.tune import quick_tune_penalized_glm

PREFERRED_VALIDATION_PENALIZED_GLM_MODELS = ("glm_elastic_net", "glm_lasso", "glm_ridge")


def selected_penalized_glm_models(selected_models: list[str] | None) -> list[str]:
    if not selected_models:
        return []
    return [model_name for model_name in selected_models if model_name in PENALIZED_GLM_MODEL_NAMES]


def resolve_penalized_glm_feature_columns(
    feature_cols: list[str],
    *,
    selected_models: list[str],
    model_feature_columns: dict[str, list[str]] | None,
    fallback_columns: list[str] | None = None,
) -> dict[str, list[str]]:
    fallback = list(fallback_columns) if fallback_columns is not None else glm_feature_subset(feature_cols)
    resolved: dict[str, list[str]] = {}
    for model_name in selected_penalized_glm_models(selected_models):
        resolved[model_name] = resolve_model_feature_columns(
            feature_cols,
            model_name=model_name,
            model_feature_columns=model_feature_columns,
            fallback_columns=fallback,
        )
    return resolved


def tune_penalized_glm_models(
    train_df,
    *,
    selected_models: list[str],
    feature_columns_by_model: dict[str, list[str]],
    n_splits: int,
    min_train_size: int,
) -> dict[str, dict]:
    tuned: dict[str, dict] = {}
    for model_name in selected_penalized_glm_models(selected_models):
        tuned[model_name] = quick_tune_penalized_glm(
            train_df,
            feature_cols=feature_columns_by_model.get(model_name, []),
            model_name=model_name,
            n_splits=n_splits,
            min_train_size=min_train_size,
        )
    return tuned


def penalized_glm_fit_summary(model: object | None, *, top_n: int = 8) -> dict[str, Any]:
    fit_summary_fn = getattr(model, "fit_metadata_dict", None)
    if not callable(fit_summary_fn):
        return {}
    try:
        summary = fit_summary_fn(top_n=top_n)
    except TypeError:
        summary = fit_summary_fn()
    return dict(summary) if isinstance(summary, dict) else {}


def build_penalized_glm_penalty_selection(
    model_name: str,
    *,
    tuning: Mapping[str, Any] | None = None,
    model: object | None = None,
) -> PenaltySelection | None:
    token = str(model_name or "").strip()
    if token not in PENALIZED_GLM_MODEL_NAMES:
        return None

    tuning_dict = dict(tuning or {})
    fit_summary = penalized_glm_fit_summary(model)
    config = penalized_glm_config(token)
    best_lambda = tuning_dict.get("best_lambda")
    best_c = tuning_dict.get("best_c", getattr(model, "c", config.default_c))
    best_l1_ratio = tuning_dict.get("best_l1_ratio", getattr(model, "l1_ratio", config.default_l1_ratio))
    choice = penalty_choice(token, lambda_value=best_lambda, c_value=best_c, l1_ratio=best_l1_ratio)

    active_features = fit_summary.get("active_features")
    if not active_features:
        active_features = tuning_dict.get("best_active_features", [])

    active_parameter_count = fit_summary.get("active_parameter_count")
    if active_parameter_count is None:
        active_parameter_count = tuning_dict.get("best_active_parameter_count")

    return PenaltySelection(
        model_name=token,
        penalty_family=str(choice["penalty_family"]),
        lambda_value=float(choice["lambda"]),
        c_value=float(choice["c"]),
        l1_ratio=None if choice["l1_ratio"] is None else float(choice["l1_ratio"]),
        grid_results=[dict(row) for row in tuning_dict.get("results", []) if isinstance(row, Mapping)],
        fold_metrics=[dict(row) for row in tuning_dict.get("fold_metrics", []) if isinstance(row, Mapping)],
        active_parameter_count=None if active_parameter_count is None else int(active_parameter_count),
        active_features=[str(value) for value in active_features if str(value).strip()],
    )


def collect_penalized_glm_artifact_payloads(
    *,
    selected_models: list[str],
    models: Mapping[str, object] | None = None,
    tuning_by_model: Mapping[str, Mapping[str, Any]] | None = None,
    top_n: int = 8,
) -> dict[str, dict[str, Any]]:
    resolved_models = dict(models or {})
    resolved_tuning = dict(tuning_by_model or {})
    payloads: dict[str, dict[str, Any]] = {}
    for model_name in selected_penalized_glm_models(selected_models):
        model = resolved_models.get(model_name)
        tuning = resolved_tuning.get(model_name)
        fit_summary = penalized_glm_fit_summary(model, top_n=top_n)
        if not fit_summary and isinstance(tuning, Mapping):
            fit_summary = {
                "model_name": model_name,
                "penalty_family": str(tuning.get("penalty_family") or model_name.removeprefix("glm_")),
                "active_parameter_count": tuning.get("best_active_parameter_count"),
                "active_features": list(tuning.get("best_active_features", [])),
                "active_coefficient_summary": list(tuning.get("best_active_coefficient_summary", [])),
            }
        payloads[model_name] = {
            "fit_summary": fit_summary,
            "penalty": build_penalized_glm_penalty_selection(
                model_name,
                tuning=tuning,
                model=model,
            ),
            "coefficient_path_metadata": dict(fit_summary.get("coefficient_path_metadata", {})),
        }
    return payloads


def primary_penalized_glm_name(selected_models: list[str], models: dict[str, object]) -> str | None:
    available = {str(name) for name in models.keys()}
    selected = set(selected_penalized_glm_models(selected_models))
    for model_name in PREFERRED_VALIDATION_PENALIZED_GLM_MODELS:
        if model_name in selected and model_name in available:
            return model_name
    for model_name in selected_penalized_glm_models(selected_models):
        if model_name in available:
            return model_name
    for model_name in PREFERRED_VALIDATION_PENALIZED_GLM_MODELS:
        if model_name in available:
            return model_name
    return None
