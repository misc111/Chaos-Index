from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

from src.models.core.glm_penalized import (
    ACTIVE_COEF_TOLERANCE,
    PENALIZED_GLM_MODEL_NAMES,
    penalized_glm_config,
)
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


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if math.isfinite(numeric) else None


def _safe_int(value: Any) -> int | None:
    numeric = _safe_float(value)
    return int(numeric) if numeric is not None else None


def _mapping_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _clean_feature_names(values: Any, *, top_n: int | None = None) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    names: list[str] = []
    for value in values:
        token = str(value).strip()
        if token:
            names.append(token)
    if top_n is not None:
        return names[: max(int(top_n), 0)]
    return names


def _penalty_key(*, lambda_value: Any = None, c_value: Any = None, l1_ratio: Any = None) -> tuple[float | None, float | None, float | None]:
    lam = _safe_float(lambda_value)
    c = _safe_float(c_value)
    l1 = _safe_float(l1_ratio)
    return (
        None if lam is None else round(lam, 12),
        None if c is None else round(c, 12),
        None if l1 is None else round(l1, 12),
    )


def _row_penalty_key(row: Mapping[str, Any]) -> tuple[float | None, float | None, float | None]:
    return _penalty_key(
        lambda_value=row.get("lambda"),
        c_value=row.get("c"),
        l1_ratio=row.get("l1_ratio"),
    )


def _metric_sort_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, float, float]:
    log_loss = _safe_float(row.get("log_loss"))
    brier = _safe_float(row.get("brier"))
    accuracy = _safe_float(row.get("accuracy"))
    lam = _safe_float(row.get("lambda"))
    c = _safe_float(row.get("c"))
    l1 = _safe_float(row.get("l1_ratio"))
    return (
        float("inf") if log_loss is None else log_loss,
        float("inf") if brier is None else brier,
        float("inf") if accuracy is None else -accuracy,
        float("inf") if lam is None else lam,
        float("inf") if c is None else c,
        float("inf") if l1 is None else l1,
    )


def _rank_grid_results(
    rows: list[dict[str, Any]],
    *,
    selected_penalty: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    selected_key = _penalty_key(
        lambda_value=selected_penalty.get("lambda"),
        c_value=selected_penalty.get("c"),
        l1_ratio=selected_penalty.get("l1_ratio"),
    )
    sorted_rows = sorted((dict(row) for row in rows), key=_metric_sort_key)
    best_row = sorted_rows[0]
    best_log_loss = _safe_float(best_row.get("log_loss"))
    best_brier = _safe_float(best_row.get("brier"))
    best_accuracy = _safe_float(best_row.get("accuracy"))

    ranked_rows: list[dict[str, Any]] = []
    for index, row in enumerate(sorted_rows, start=1):
        enriched = dict(row)
        log_loss = _safe_float(enriched.get("log_loss"))
        brier = _safe_float(enriched.get("brier"))
        accuracy = _safe_float(enriched.get("accuracy"))
        enriched["tuning_rank"] = int(index)
        enriched["is_best_candidate"] = bool(index == 1)
        enriched["matches_selected_penalty"] = bool(_row_penalty_key(enriched) == selected_key)
        enriched["log_loss_delta_to_best"] = (
            None if log_loss is None or best_log_loss is None else float(log_loss - best_log_loss)
        )
        enriched["brier_delta_to_best"] = (
            None if brier is None or best_brier is None else float(brier - best_brier)
        )
        enriched["accuracy_delta_to_best"] = (
            None if accuracy is None or best_accuracy is None else float(accuracy - best_accuracy)
        )
        ranked_rows.append(enriched)
    return ranked_rows


def _annotate_fold_metrics(
    rows: list[dict[str, Any]],
    *,
    selected_penalty: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    selected_key = _penalty_key(
        lambda_value=selected_penalty.get("lambda"),
        c_value=selected_penalty.get("c"),
        l1_ratio=selected_penalty.get("l1_ratio"),
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    fold_order: list[str] = []
    for row in rows:
        fold_key = str(row.get("fold"))
        if fold_key not in grouped:
            grouped[fold_key] = []
            fold_order.append(fold_key)
        grouped[fold_key].append(dict(row))

    annotated: list[dict[str, Any]] = []
    for fold_key in fold_order:
        fold_rows = sorted(grouped.get(fold_key, []), key=_metric_sort_key)
        if not fold_rows:
            continue
        best_log_loss = _safe_float(fold_rows[0].get("log_loss"))
        for index, row in enumerate(fold_rows, start=1):
            enriched = dict(row)
            log_loss = _safe_float(enriched.get("log_loss"))
            enriched["fold_rank"] = int(index)
            enriched["is_fold_best"] = bool(index == 1)
            enriched["matches_selected_penalty"] = bool(_row_penalty_key(enriched) == selected_key)
            enriched["log_loss_delta_to_fold_best"] = (
                None if log_loss is None or best_log_loss is None else float(log_loss - best_log_loss)
            )
            annotated.append(enriched)
    return annotated


def _active_coefficient_totals_from_rows(rows: list[dict[str, Any]], *, top_n: int) -> dict[str, Any]:
    top_n_requested = max(int(top_n), 0)
    abs_scaled_values: list[float] = []
    positive_count = 0
    negative_count = 0
    zero_count = 0
    for row in rows:
        abs_scaled = _safe_float(row.get("abs_coef_scaled"))
        if abs_scaled is not None:
            abs_scaled_values.append(abs_scaled)
        sign = _safe_int(row.get("sign"))
        if sign is None:
            continue
        if sign > 0:
            positive_count += 1
        elif sign < 0:
            negative_count += 1
        else:
            zero_count += 1
    abs_scaled_total = float(sum(abs_scaled_values))
    abs_scaled_top_n = float(sum(abs_scaled_values[:top_n_requested]))
    strongest = rows[0] if rows else {}
    return {
        "top_n_requested": int(top_n_requested),
        "top_n_effective": int(min(len(rows), top_n_requested)),
        "active_parameter_count": int(len(rows)),
        "abs_coef_scaled_total": float(abs_scaled_total),
        "abs_coef_scaled_top_n": float(abs_scaled_top_n),
        "top_n_abs_coef_scaled_share": (
            float(abs_scaled_top_n / abs_scaled_total) if abs_scaled_total > 0 else None
        ),
        "positive_count": int(positive_count),
        "negative_count": int(negative_count),
        "zero_count": int(zero_count),
        "strongest_feature": str(strongest.get("feature")).strip() if strongest else None,
        "strongest_abs_coef_scaled": _safe_float(strongest.get("abs_coef_scaled")),
    }


def _coefficient_path_metadata_from_tuning(
    model_name: str,
    *,
    tuning: Mapping[str, Any],
    top_n: int,
) -> dict[str, Any]:
    tuning_dict = dict(tuning)
    best_fit_summary = dict(tuning_dict.get("best_fit_summary") or {})
    if isinstance(best_fit_summary.get("coefficient_path_metadata"), Mapping):
        path_metadata = dict(best_fit_summary.get("coefficient_path_metadata") or {})
        path_metadata.setdefault("source", "best_fit_summary")
        return path_metadata

    active_rows = _mapping_rows(tuning_dict.get("best_active_coefficient_summary"))
    active_features = _clean_feature_names(tuning_dict.get("best_active_features"))
    top_n_requested = max(int(top_n), 0)
    top_rows: list[dict[str, Any]] = []
    for row in active_rows[:top_n_requested]:
        feature = str(row.get("feature") or "").strip()
        if not feature:
            continue
        top_rows.append(
            {
                "feature": feature,
                "coef_scaled": _safe_float(row.get("coef_scaled")),
                "coef_original": _safe_float(row.get("coef_original")),
                "abs_coef_scaled": _safe_float(row.get("abs_coef_scaled")),
                "abs_coef_original": _safe_float(row.get("abs_coef_original")),
                "active": True,
                "active_rank": _safe_int(row.get("active_rank")),
                "sign": _safe_int(row.get("sign")),
            }
        )
        if feature not in active_features:
            active_features.append(feature)

    config = penalized_glm_config(model_name)
    best_penalty = dict(
        tuning_dict.get("best_penalty")
        or penalty_choice(
            model_name,
            lambda_value=tuning_dict.get("best_lambda"),
            c_value=tuning_dict.get("best_c", config.default_c),
            l1_ratio=tuning_dict.get("best_l1_ratio", config.default_l1_ratio),
        )
    )
    totals = _active_coefficient_totals_from_rows(active_rows, top_n=top_n)
    feature_count = _safe_int(best_fit_summary.get("feature_count"))
    if feature_count is None:
        feature_count = int(len(active_features))

    return {
        "source": "tuning_best_active_coefficient_summary",
        "path_columns": _clean_feature_names(best_fit_summary.get("coefficient_columns")),
        "sort_key": "abs_coef_scaled_desc",
        "top_n_requested": int(top_n_requested),
        "top_n_effective": int(len(top_rows)),
        "top_feature_names": [str(row["feature"]) for row in top_rows],
        "top_path_rows": top_rows,
        "active_feature_names": active_features,
        "n_features_total": feature_count,
        "n_features_active": int(len(active_features)),
        "n_features_inactive": (
            int(max(feature_count - len(active_features), 0)) if feature_count is not None else None
        ),
        "active_fraction": (float(len(active_features) / feature_count) if feature_count and feature_count > 0 else None),
        "active_sign_counts": {
            "positive": int(totals["positive_count"]),
            "negative": int(totals["negative_count"]),
            "zero": int(totals["zero_count"]),
        },
        "active_abs_coef_scaled_total": float(totals["abs_coef_scaled_total"]),
        "top_n_abs_coef_scaled_share": totals["top_n_abs_coef_scaled_share"],
        "active_threshold_abs_coef_scaled": float(ACTIVE_COEF_TOLERANCE),
        "parameterization": {
            "model_name": str(best_penalty.get("model_name") or model_name),
            "penalty_family": str(best_penalty.get("penalty_family") or model_name.removeprefix("glm_")),
            "penalty": str(best_penalty.get("penalty") or config.penalty),
            "solver": str(best_penalty.get("solver") or config.solver),
            "lambda": float(_safe_float(best_penalty.get("lambda")) or 1.0 / float(config.default_c)),
            "c": float(_safe_float(best_penalty.get("c")) or config.default_c),
            "l1_ratio": _safe_float(best_penalty.get("l1_ratio")),
        },
    }


def _fallback_fit_summary_from_tuning(
    model_name: str,
    *,
    tuning: Mapping[str, Any],
    top_n: int,
) -> dict[str, Any]:
    tuning_dict = dict(tuning)
    best_fit_summary = dict(tuning_dict.get("best_fit_summary") or {})
    if best_fit_summary:
        return best_fit_summary

    config = penalized_glm_config(model_name)
    best_penalty = dict(
        tuning_dict.get("best_penalty")
        or penalty_choice(
            model_name,
            lambda_value=tuning_dict.get("best_lambda"),
            c_value=tuning_dict.get("best_c", config.default_c),
            l1_ratio=tuning_dict.get("best_l1_ratio", config.default_l1_ratio),
        )
    )
    active_features = _clean_feature_names(tuning_dict.get("best_active_features"))
    active_rows = _mapping_rows(tuning_dict.get("best_active_coefficient_summary"))
    active_totals = _active_coefficient_totals_from_rows(active_rows, top_n=top_n)
    return {
        "model_name": str(best_penalty.get("model_name") or model_name),
        "penalty_family": str(best_penalty.get("penalty_family") or model_name.removeprefix("glm_")),
        "penalty": str(best_penalty.get("penalty") or config.penalty),
        "solver": str(best_penalty.get("solver") or config.solver),
        "lambda": float(_safe_float(best_penalty.get("lambda")) or 1.0 / float(config.default_c)),
        "c": float(_safe_float(best_penalty.get("c")) or config.default_c),
        "l1_ratio": _safe_float(best_penalty.get("l1_ratio")),
        "active_parameter_count": _safe_int(tuning_dict.get("best_active_parameter_count")) or int(len(active_features)),
        "active_features": active_features,
        "active_coefficient_totals": active_totals,
        "active_coefficient_summary": active_rows[: max(int(top_n), 0)],
        "coefficient_columns": _clean_feature_names(best_fit_summary.get("coefficient_columns")),
    }


def _tuning_summary(
    model_name: str,
    *,
    tuning: Mapping[str, Any] | None,
    top_n: int,
) -> dict[str, Any]:
    if not isinstance(tuning, Mapping):
        return {}
    tuning_dict = dict(tuning)
    config = penalized_glm_config(model_name)
    selected_penalty = penalty_choice(
        model_name,
        lambda_value=tuning_dict.get("best_lambda"),
        c_value=tuning_dict.get("best_c", config.default_c),
        l1_ratio=tuning_dict.get("best_l1_ratio", config.default_l1_ratio),
    )

    ranked_results = _rank_grid_results(_mapping_rows(tuning_dict.get("results")), selected_penalty=selected_penalty)
    annotated_folds = _annotate_fold_metrics(
        _mapping_rows(tuning_dict.get("fold_metrics")),
        selected_penalty=selected_penalty,
    )
    selected_result = next((row for row in ranked_results if row.get("matches_selected_penalty")), None)
    comparison_best = ranked_results[0] if ranked_results else {}
    runner_up = ranked_results[1] if len(ranked_results) > 1 else None

    selected_fold_counter: Counter[str] = Counter()
    for row in annotated_folds:
        if not bool(row.get("matches_selected_penalty")):
            continue
        for feature in _clean_feature_names(row.get("leading_features")):
            selected_fold_counter[feature] += 1

    selected_fold_features = [
        {"feature": feature, "fold_count": int(count)}
        for feature, count in selected_fold_counter.most_common(max(int(top_n), 0))
    ]
    top_candidates: list[dict[str, Any]] = []
    for row in ranked_results[:3]:
        top_candidates.append(
            {
                "tuning_rank": int(row.get("tuning_rank", 0)),
                "matches_selected_penalty": bool(row.get("matches_selected_penalty", False)),
                "lambda": _safe_float(row.get("lambda")),
                "c": _safe_float(row.get("c")),
                "l1_ratio": _safe_float(row.get("l1_ratio")),
                "log_loss": _safe_float(row.get("log_loss")),
                "brier": _safe_float(row.get("brier")),
                "accuracy": _safe_float(row.get("accuracy")),
                "auc": _safe_float(row.get("auc")),
                "active_parameter_count_mean": _safe_float(row.get("active_parameter_count_mean")),
            }
        )

    return {
        "model_name": str(model_name),
        "penalty_family": str(selected_penalty.get("penalty_family") or model_name.removeprefix("glm_")),
        "search_grid_size": int(len(_mapping_rows(tuning_dict.get("search_grid")))),
        "evaluated_parameter_count": int(len(ranked_results)),
        "evaluated_fold_rows": int(len(annotated_folds)),
        "evaluated_fold_count": int(len({str(row.get("fold")) for row in annotated_folds})),
        "best_parameterization": {
            "model_name": str(selected_penalty.get("model_name") or model_name),
            "penalty_family": str(selected_penalty.get("penalty_family") or model_name.removeprefix("glm_")),
            "lambda": float(selected_penalty.get("lambda")),
            "c": float(selected_penalty.get("c")),
            "l1_ratio": _safe_float(selected_penalty.get("l1_ratio")),
        },
        "selected_penalty_rank": _safe_int(selected_result.get("tuning_rank")) if selected_result else None,
        "best_candidate_metrics": {
            "log_loss": _safe_float(comparison_best.get("log_loss")),
            "brier": _safe_float(comparison_best.get("brier")),
            "accuracy": _safe_float(comparison_best.get("accuracy")),
            "auc": _safe_float(comparison_best.get("auc")),
        },
        "selected_penalty_metrics": {
            "log_loss": _safe_float(selected_result.get("log_loss")) if selected_result else None,
            "brier": _safe_float(selected_result.get("brier")) if selected_result else None,
            "accuracy": _safe_float(selected_result.get("accuracy")) if selected_result else None,
            "auc": _safe_float(selected_result.get("auc")) if selected_result else None,
        },
        "selected_penalty_active_parameter_count_mean": (
            _safe_float(selected_result.get("active_parameter_count_mean")) if selected_result else None
        ),
        "selected_penalty_active_parameter_count_min": (
            _safe_int(selected_result.get("active_parameter_count_min")) if selected_result else None
        ),
        "selected_penalty_active_parameter_count_max": (
            _safe_int(selected_result.get("active_parameter_count_max")) if selected_result else None
        ),
        "selected_penalty_active_feature_union": (
            _clean_feature_names(selected_result.get("active_feature_union")) if selected_result else []
        ),
        "selected_penalty_active_feature_frequency": (
            _mapping_rows(selected_result.get("active_feature_frequency"))[: max(int(top_n), 0)] if selected_result else []
        ),
        "selected_penalty_leading_coefficients": (
            _mapping_rows(selected_result.get("leading_coefficients"))[: max(int(top_n), 0)] if selected_result else []
        ),
        "selected_penalty_fold_leading_features": selected_fold_features,
        "runner_up_gap": (
            {
                "log_loss": (
                    None
                    if _safe_float(runner_up.get("log_loss")) is None or _safe_float(comparison_best.get("log_loss")) is None
                    else float(_safe_float(runner_up.get("log_loss")) - _safe_float(comparison_best.get("log_loss")))
                ),
                "brier": (
                    None
                    if _safe_float(runner_up.get("brier")) is None or _safe_float(comparison_best.get("brier")) is None
                    else float(_safe_float(runner_up.get("brier")) - _safe_float(comparison_best.get("brier")))
                ),
                "accuracy": (
                    None
                    if _safe_float(runner_up.get("accuracy")) is None
                    or _safe_float(comparison_best.get("accuracy")) is None
                    else float(_safe_float(comparison_best.get("accuracy")) - _safe_float(runner_up.get("accuracy")))
                ),
            }
            if runner_up is not None
            else {}
        ),
        "top_candidates": top_candidates,
    }


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
    ranked_results = _rank_grid_results(_mapping_rows(tuning_dict.get("results")), selected_penalty=choice)
    annotated_fold_metrics = _annotate_fold_metrics(_mapping_rows(tuning_dict.get("fold_metrics")), selected_penalty=choice)

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
        grid_results=ranked_results,
        fold_metrics=annotated_fold_metrics,
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
        tuning_summary = _tuning_summary(model_name, tuning=tuning, top_n=top_n)
        fit_summary = penalized_glm_fit_summary(model, top_n=top_n)
        if not fit_summary and isinstance(tuning, Mapping):
            fit_summary = _fallback_fit_summary_from_tuning(model_name, tuning=tuning, top_n=top_n)
        fit_summary = dict(fit_summary)

        path_metadata = {}
        if isinstance(fit_summary.get("coefficient_path_metadata"), Mapping):
            path_metadata = dict(fit_summary.get("coefficient_path_metadata") or {})
        if not path_metadata and isinstance(tuning, Mapping):
            path_metadata = _coefficient_path_metadata_from_tuning(model_name, tuning=tuning, top_n=top_n)
        if path_metadata:
            fit_summary["coefficient_path_metadata"] = path_metadata

        if not fit_summary.get("active_features") and isinstance(tuning, Mapping):
            fit_summary["active_features"] = _clean_feature_names(tuning.get("best_active_features"))
        if fit_summary.get("active_parameter_count") is None and isinstance(tuning, Mapping):
            fit_summary["active_parameter_count"] = _safe_int(tuning.get("best_active_parameter_count"))

        if "active_coefficient_summary" not in fit_summary and isinstance(tuning, Mapping):
            fit_summary["active_coefficient_summary"] = _mapping_rows(tuning.get("best_active_coefficient_summary"))

        if "active_coefficient_totals" not in fit_summary:
            fit_summary["active_coefficient_totals"] = _active_coefficient_totals_from_rows(
                _mapping_rows(fit_summary.get("active_coefficient_summary")),
                top_n=top_n,
            )

        if tuning_summary:
            fit_summary["tuning_summary"] = tuning_summary

        payloads[model_name] = {
            "fit_summary": fit_summary,
            "penalty": build_penalized_glm_penalty_selection(
                model_name,
                tuning=tuning,
                model=model,
            ),
            "coefficient_path_metadata": dict(path_metadata),
            "tuning_summary": tuning_summary,
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
