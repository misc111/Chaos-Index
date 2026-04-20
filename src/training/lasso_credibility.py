from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import metric_bundle
from src.models.lasso_credibility import (
    ACTIVE_COEF_TOLERANCE,
    LassoCredibilityModel,
    default_lasso_credibility_label,
    normalize_lasso_credibility_kind,
)
from src.registry.models import planned_credibility_model_catalog
from src.training.contracts import (
    CandidateScorecardRecord,
    LassoCredibilityMetadata,
    ModelArtifactRecord,
    PenaltySelection,
)
from src.training.cv import time_series_splits
from src.training.lambda_search import default_lambda_grid, lambda_to_c


_PLANNED_CREDIBILITY_CATALOG = planned_credibility_model_catalog()
_MODEL_BY_KIND = {
    "market": "glm_lasso_market_credibility",
    "prior": "glm_lasso_prior_credibility",
}
LASSO_CREDIBILITY_MODEL_NAMES = tuple(_MODEL_BY_KIND.values())


def resolve_lasso_credibility_model_name(complement_kind: str) -> str:
    return _MODEL_BY_KIND[normalize_lasso_credibility_kind(complement_kind)]


def selected_lasso_credibility_models(selected_models: list[str] | None) -> list[str]:
    if not selected_models:
        return []
    return [model_name for model_name in selected_models if model_name in LASSO_CREDIBILITY_MODEL_NAMES]


def _resolve_model_payload(complement_kind: str, model_name: str | None = None) -> dict[str, object]:
    token = str(model_name or resolve_lasso_credibility_model_name(complement_kind)).strip()
    if token not in _PLANNED_CREDIBILITY_CATALOG:
        raise ValueError(
            f"Unsupported lasso-credibility model '{token}'. "
            f"Valid={sorted(_PLANNED_CREDIBILITY_CATALOG)}"
        )
    return dict(_PLANNED_CREDIBILITY_CATALOG[token])


def _complement_kind_for_model(model_name: str) -> str:
    payload = _resolve_model_payload("market" if "market" in str(model_name) else "prior", model_name=model_name)
    raw_kind = str(payload.get("complement_kind") or "")
    return normalize_lasso_credibility_kind(raw_kind)


def _default_input_scale(complement_column: str | None, model_payload: Mapping[str, object]) -> str:
    planned_column = str(model_payload.get("complement_column") or "").strip()
    if complement_column and complement_column == planned_column:
        return str(model_payload.get("offset_scale") or "logit").strip().lower()
    return "probability"


def _resolve_complement_label(
    complement_kind: str,
    complement_column: str,
    *,
    complement_label: str | None,
    model_payload: Mapping[str, object],
) -> str:
    if complement_label:
        return str(complement_label)
    planned_column = str(model_payload.get("complement_column") or "").strip()
    if complement_column == planned_column:
        planned_label = str(model_payload.get("complement_label") or "").strip()
        if planned_label:
            return planned_label
    return default_lasso_credibility_label(complement_kind, complement_column)


def _default_tuning_result(complement_kind: str, model_name: str | None = None) -> dict[str, Any]:
    resolved_model_name = str(model_name or resolve_lasso_credibility_model_name(complement_kind))
    best_lambda = 1.0
    return {
        "model_name": resolved_model_name,
        "best_params": {
            "lambda": best_lambda,
            "c": lambda_to_c(best_lambda),
        },
        "best_lambda": best_lambda,
        "best_c": lambda_to_c(best_lambda),
        "best_l1_ratio": None,
        "results": [],
        "fold_metrics": [],
        "lambda_choice_note": (
            "Insufficient rows for time-series lambda tuning; defaulted to lambda=1.0 and kept the fit fully regularized."
        ),
    }


def quick_tune_lasso_credibility(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    complement_kind: str,
    complement_column: str | None = None,
    complement_label: str | None = None,
    complement_input_scale: str | None = None,
    exposure_column: str | None = None,
    model_name: str | None = None,
    target_col: str = "home_win",
    lambda_grid: list[float] | None = None,
    n_splits: int = 4,
    min_train_size: int = 220,
) -> dict[str, Any]:
    normalized_kind = normalize_lasso_credibility_kind(complement_kind)
    model_payload = _resolve_model_payload(normalized_kind, model_name=model_name)
    resolved_model_name = str(model_name or resolve_lasso_credibility_model_name(normalized_kind))
    resolved_complement_column = str(complement_column or model_payload.get("complement_column") or "").strip()
    if not resolved_complement_column:
        raise ValueError("Lasso credibility tuning requires a complement_column.")
    resolved_input_scale = str(
        complement_input_scale or _default_input_scale(resolved_complement_column, model_payload)
    ).strip().lower()
    lambda_values = list(lambda_grid) if lambda_grid is not None else default_lambda_grid("glm_lasso")
    default_result = _default_tuning_result(normalized_kind, model_name=resolved_model_name)

    if target_col not in df.columns:
        raise ValueError(f"Lasso credibility tuning requires target column '{target_col}'.")
    train = df[df[target_col].notna()].copy().sort_values("start_time_utc")
    if len(train) < max(80, min_train_size + 20):
        return default_result

    splits = time_series_splits(train, n_splits=n_splits, min_train_size=min_train_size)
    if not splits:
        return default_result

    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for lambda_value in lambda_values:
        fold_scores: list[dict[str, float]] = []
        for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
            tr = train.loc[tr_idx].copy().sort_values("start_time_utc")
            va = train.loc[va_idx].copy().sort_values("start_time_utc")
            if tr.empty or va.empty:
                continue
            model = LassoCredibilityModel(
                lambda_value=float(lambda_value),
                complement_kind=normalized_kind,
                complement_column=resolved_complement_column,
                complement_label=_resolve_complement_label(
                    normalized_kind,
                    resolved_complement_column,
                    complement_label=complement_label,
                    model_payload=model_payload,
                ),
                complement_input_scale=resolved_input_scale,
                exposure_column=exposure_column,
            )
            model.fit(tr, feature_columns=feature_cols, target_col=target_col)
            pred = model.predict_proba(va)
            metrics = metric_bundle(va[target_col].astype(int).to_numpy(), pred)
            fold_scores.append(metrics)
            fold_rows.append(
                {
                    "model_name": resolved_model_name,
                    "complement_kind": normalized_kind,
                    "complement_column": resolved_complement_column,
                    "lambda": float(lambda_value),
                    "c": float(lambda_to_c(float(lambda_value))),
                    "fold": int(fold),
                    "n_train": int(len(tr)),
                    "n_valid": int(len(va)),
                    "log_loss": float(metrics["log_loss"]),
                    "brier": float(metrics["brier"]),
                    "accuracy": float(metrics["accuracy"]),
                    "auc": float(metrics["auc"]),
                }
            )
        if not fold_scores:
            continue
        rows.append(
            {
                "model_name": resolved_model_name,
                "complement_kind": normalized_kind,
                "complement_column": resolved_complement_column,
                "lambda": float(lambda_value),
                "c": float(lambda_to_c(float(lambda_value))),
                "folds_used": int(len(fold_scores)),
                "log_loss": float(pd.DataFrame(fold_scores)["log_loss"].mean()),
                "brier": float(pd.DataFrame(fold_scores)["brier"].mean()),
                "accuracy": float(pd.DataFrame(fold_scores)["accuracy"].mean()),
                "auc": float(pd.DataFrame(fold_scores)["auc"].mean()),
            }
        )

    if not rows:
        out = dict(default_result)
        out["fold_metrics"] = fold_rows
        return out

    best = sorted(rows, key=lambda row: (row["log_loss"], row["brier"], -row["accuracy"]))[0]
    best_lambda = float(best["lambda"])
    return {
        "model_name": resolved_model_name,
        "best_params": {
            "lambda": best_lambda,
            "c": float(best["c"]),
        },
        "best_lambda": best_lambda,
        "best_c": float(best["c"]),
        "best_l1_ratio": None,
        "results": rows,
        "fold_metrics": fold_rows,
        "lambda_choice_note": (
            "Selected lambda with rolling time-series splits using mean log loss, "
            "then Brier score and accuracy as tiebreakers."
        ),
    }


def resolve_lasso_credibility_feature_columns(
    feature_cols: list[str],
    *,
    selected_models: list[str],
    model_feature_columns: dict[str, list[str]] | None,
    fallback_columns: list[str] | None = None,
) -> dict[str, list[str]]:
    fallback = list(fallback_columns) if fallback_columns is not None else list(feature_cols)
    resolved: dict[str, list[str]] = {}
    for model_name in selected_lasso_credibility_models(selected_models):
        requested = list((model_feature_columns or {}).get(model_name, []))
        if requested:
            missing = [column for column in requested if column not in feature_cols]
            if missing:
                raise ValueError(f"model_feature_columns[{model_name}] includes missing columns: {missing}")
            resolved[model_name] = requested
        else:
            resolved[model_name] = fallback
    return resolved


def tune_lasso_credibility_models(
    train_df: pd.DataFrame,
    *,
    selected_models: list[str],
    feature_columns_by_model: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    tuned: dict[str, dict[str, Any]] = {}
    for model_name in selected_lasso_credibility_models(selected_models):
        complement_kind = _complement_kind_for_model(model_name)
        payload = _resolve_model_payload(complement_kind, model_name=model_name)
        tuned[model_name] = quick_tune_lasso_credibility(
            train_df,
            feature_cols=feature_columns_by_model.get(model_name, []),
            complement_kind=complement_kind,
            complement_column=str(payload.get("complement_column") or ""),
            complement_label=str(payload.get("complement_label") or ""),
            complement_input_scale="logit",
            model_name=model_name,
        )
    return tuned


def _active_feature_list(model: LassoCredibilityModel) -> list[str]:
    frame = model.coef_frame()
    if frame.empty:
        return []
    active = frame[np.abs(frame["coef_scaled"].astype(float)) > ACTIVE_COEF_TOLERANCE]
    return [str(feature) for feature in active["feature"].tolist() if str(feature).strip()]


def build_lasso_credibility_penalty_selection(
    model_name: str,
    *,
    model: LassoCredibilityModel,
    tuning: Mapping[str, Any] | None = None,
) -> PenaltySelection:
    tuning_payload = dict(tuning or {})
    active_features = _active_feature_list(model)
    active_parameter_count = int(len(active_features) + (abs(model.intercept_scaled) > ACTIVE_COEF_TOLERANCE))
    return PenaltySelection(
        model_name=model_name,
        penalty_family="lasso",
        lambda_value=float(tuning_payload.get("best_lambda", model.lambda_value)),
        c_value=float(tuning_payload.get("best_c", lambda_to_c(model.lambda_value))),
        l1_ratio=None,
        grid_results=[dict(row) for row in tuning_payload.get("results", []) if isinstance(row, Mapping)],
        fold_metrics=[dict(row) for row in tuning_payload.get("fold_metrics", []) if isinstance(row, Mapping)],
        active_parameter_count=active_parameter_count,
        active_features=active_features,
    )


def collect_lasso_credibility_artifact_payloads(
    *,
    selected_models: list[str],
    models: Mapping[str, object] | None = None,
    tuning_by_model: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    resolved_models = dict(models or {})
    resolved_tuning = dict(tuning_by_model or {})
    payloads: dict[str, dict[str, Any]] = {}
    for model_name in selected_lasso_credibility_models(selected_models):
        model = resolved_models.get(model_name)
        if not isinstance(model, LassoCredibilityModel):
            continue
        payloads[model_name] = {
            "fit_summary": model.fit_summary(),
            "penalty": build_lasso_credibility_penalty_selection(
                model_name,
                model=model,
                tuning=resolved_tuning.get(model_name),
            ),
            "credibility_metadata": dict(model.credibility_metadata),
        }
    return payloads


def build_lasso_credibility_contracts(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    complement_kind: str,
    complement_column: str | None = None,
    complement_label: str | None = None,
    complement_input_scale: str | None = None,
    exposure_column: str | None = None,
    target_col: str = "home_win",
    model_name: str | None = None,
    lambda_value: float | None = None,
    tuning: Mapping[str, Any] | None = None,
    artifact_files: Mapping[str, str] | None = None,
    metrics_summary: Mapping[str, Any] | None = None,
    stability_metrics: Mapping[str, Any] | None = None,
    calibration_summary: Mapping[str, Any] | None = None,
    rejection_reasons: list[str] | None = None,
) -> tuple[LassoCredibilityModel, ModelArtifactRecord, CandidateScorecardRecord]:
    normalized_kind = normalize_lasso_credibility_kind(complement_kind)
    resolved_model_name = str(model_name or resolve_lasso_credibility_model_name(normalized_kind))
    model_payload = _resolve_model_payload(normalized_kind, model_name=resolved_model_name)
    resolved_complement_column = str(complement_column or model_payload.get("complement_column") or "").strip()
    if not resolved_complement_column:
        raise ValueError("Lasso credibility contract building requires a complement_column.")
    resolved_input_scale = str(
        complement_input_scale or _default_input_scale(resolved_complement_column, model_payload)
    ).strip().lower()

    tuning_payload = dict(tuning or {})
    resolved_lambda = float(lambda_value if lambda_value is not None else tuning_payload.get("best_lambda", 1.0))
    model = LassoCredibilityModel(
        lambda_value=resolved_lambda,
        complement_kind=normalized_kind,
        complement_column=resolved_complement_column,
        complement_label=_resolve_complement_label(
            normalized_kind,
            resolved_complement_column,
            complement_label=complement_label,
            model_payload=model_payload,
        ),
        complement_input_scale=resolved_input_scale,
        exposure_column=exposure_column,
    )
    model.fit(df, feature_columns=feature_cols, target_col=target_col)

    if metrics_summary is None:
        train_metrics = metric_bundle(model.train_y, model.train_prob)  # type: ignore[arg-type]
    else:
        train_metrics = dict(metrics_summary)
    fit_summary = model.fit_summary()
    fit_summary["driver_source_model"] = str(model_payload.get("driver_source_model") or "glm_lasso")
    fit_summary["planned_model_key"] = resolved_model_name

    active_features = _active_feature_list(model)
    penalty = PenaltySelection(
        model_name=resolved_model_name,
        penalty_family="lasso",
        lambda_value=resolved_lambda,
        c_value=float(lambda_to_c(resolved_lambda)),
        l1_ratio=None,
        grid_results=[dict(row) for row in tuning_payload.get("results", []) if isinstance(row, Mapping)],
        fold_metrics=[dict(row) for row in tuning_payload.get("fold_metrics", []) if isinstance(row, Mapping)],
        active_parameter_count=int(len(active_features) + (abs(model.intercept_scaled) > ACTIVE_COEF_TOLERANCE)),
        active_features=active_features,
    )
    credibility = LassoCredibilityMetadata(
        model_name=resolved_model_name,
        complement_kind=str(model.credibility_metadata["complement_kind"]),
        complement_label=str(model.credibility_metadata["complement_label"]),
        complement_column=str(model.credibility_metadata["complement_column"]),
        offset_scale=str(model.credibility_metadata["offset_scale"]),
        driver_features=list(model.credibility_metadata.get("driver_features", [])),
        complement_summary=dict(model.credibility_metadata.get("complement_summary", {})),
        relativity_summary=dict(model.credibility_metadata.get("relativity_summary", {})),
        exposure_summary=dict(model.credibility_metadata.get("exposure_summary", {})),
        lambda_choice_note=str(
            tuning_payload.get("lambda_choice_note") or model.credibility_metadata.get("lambda_choice_note") or ""
        ),
        complement_choice_note=str(model.credibility_metadata.get("complement_choice_note") or ""),
        p_values_reported=False,
    )

    artifact = ModelArtifactRecord(
        model_name=resolved_model_name,
        model_family=str(model_payload.get("family") or "credibility"),
        lane=str(model_payload.get("lane") or "core"),
        artifact_files=dict(artifact_files or {}),
        feature_columns=list(model.feature_columns),
        metrics_summary=dict(train_metrics),
        fit_summary=dict(fit_summary),
        penalty=penalty,
        credibility=credibility,
    )
    scorecard = CandidateScorecardRecord(
        model_name=resolved_model_name,
        lane=str(model_payload.get("lane") or "core"),
        target_name=target_col,
        distribution="binomial",
        link_function="logit",
        feature_count=int(len(model.feature_columns)),
        active_parameter_count=int(len(active_features) + (abs(model.intercept_scaled) > ACTIVE_COEF_TOLERANCE)),
        validation_metrics=dict(train_metrics),
        stability_metrics=dict(stability_metrics or {}),
        calibration_summary=dict(calibration_summary or {}),
        complement_summary=dict(credibility.complement_summary),
        rejection_reasons=[str(reason) for reason in (rejection_reasons or []) if str(reason).strip()],
    )
    return model, artifact, scorecard


__all__ = [
    "build_lasso_credibility_contracts",
    "build_lasso_credibility_penalty_selection",
    "collect_lasso_credibility_artifact_payloads",
    "LASSO_CREDIBILITY_MODEL_NAMES",
    "quick_tune_lasso_credibility",
    "resolve_lasso_credibility_model_name",
    "resolve_lasso_credibility_feature_columns",
    "selected_lasso_credibility_models",
    "tune_lasso_credibility_models",
]
