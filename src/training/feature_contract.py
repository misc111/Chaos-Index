"""Deterministic feature-column contract for MLB training entry points.

This module is intentionally narrow: it resolves the feature columns that the
training and backtest orchestration layers pass into model fitting. Keeping that
resolution in one place prevents train/backtest drift while preserving the
existing model math and feature ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.training.feature_selection import glm_feature_subset, select_feature_columns
from src.training.lasso_credibility import resolve_lasso_credibility_feature_columns
from src.training.penalized_glm import resolve_penalized_glm_feature_columns


@dataclass(frozen=True)
class TrainingFeatureContract:
    """Resolved feature selections used by one MLB training or backtest run."""

    feature_columns: list[str]
    glm_feature_columns: list[str]
    penalized_glm_feature_columns: dict[str, list[str]]
    lasso_credibility_feature_columns: dict[str, list[str]]


def validate_selected_feature_columns(df: pd.DataFrame, selected_feature_columns: list[str]) -> list[str]:
    """Validate an explicit feature list and return it in caller-provided order."""

    requested = [str(column) for column in selected_feature_columns]
    missing_cols = [column for column in requested if column not in df.columns]
    if missing_cols:
        raise ValueError(f"selected_feature_columns includes missing columns: {missing_cols}")

    non_numeric = [column for column in requested if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric:
        raise ValueError(f"selected_feature_columns includes non-numeric columns: {non_numeric}")
    return requested


def resolve_base_feature_columns(
    df: pd.DataFrame,
    *,
    selected_feature_columns: list[str] | None = None,
) -> list[str]:
    """Resolve the top-level feature list used by all selected models."""

    if selected_feature_columns is None:
        return select_feature_columns(df)
    return validate_selected_feature_columns(df, selected_feature_columns)


def _primary_glm_feature_columns(
    *,
    feature_columns: list[str],
    penalized_glm_feature_columns: dict[str, list[str]],
) -> list[str]:
    """Return the deterministic GLM fallback used by legacy training call sites."""

    return (
        penalized_glm_feature_columns.get("glm_ridge")
        or penalized_glm_feature_columns.get("glm_elastic_net")
        or penalized_glm_feature_columns.get("glm_lasso")
        or glm_feature_subset(feature_columns)
    )

def resolve_training_feature_contract(
    df: pd.DataFrame,
    *,
    selected_models: list[str],
    selected_feature_columns: list[str] | None = None,
    selected_model_feature_columns: dict[str, list[str]] | None = None,
) -> TrainingFeatureContract:
    """Resolve deterministic run-wide and model-specific training features."""

    feature_columns = resolve_base_feature_columns(df, selected_feature_columns=selected_feature_columns)
    default_glm_columns = glm_feature_subset(feature_columns)
    penalized_glm_feature_columns = resolve_penalized_glm_feature_columns(
        feature_columns,
        selected_models=selected_models,
        model_feature_columns=selected_model_feature_columns,
        fallback_columns=default_glm_columns,
    )
    lasso_credibility_feature_columns = resolve_lasso_credibility_feature_columns(
        feature_columns,
        selected_models=selected_models,
        model_feature_columns=selected_model_feature_columns,
        fallback_columns=default_glm_columns,
    )
    return TrainingFeatureContract(
        feature_columns=feature_columns,
        glm_feature_columns=_primary_glm_feature_columns(
            feature_columns=feature_columns,
            penalized_glm_feature_columns=penalized_glm_feature_columns,
        ),
        penalized_glm_feature_columns=penalized_glm_feature_columns,
        lasso_credibility_feature_columns=lasso_credibility_feature_columns,
    )
