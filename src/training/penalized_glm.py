from __future__ import annotations

from src.models.glm_penalized import PENALIZED_GLM_MODEL_NAMES
from src.training.feature_selection import glm_feature_subset, resolve_model_feature_columns
from src.training.tune import quick_tune_penalized_glm

PREFERRED_VALIDATION_PENALIZED_GLM_MODELS = ("glm_elastic_net", "glm_ridge")


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
