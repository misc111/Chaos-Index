from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from src.common.config import AppConfig, load_config
from src.common.utils import ensure_dir, stable_hash, to_json
from src.evaluation.metrics import metric_bundle, per_game_scores
from src.evaluation.calibration import calibration_alpha_beta
from src.evaluation.validation_classification import validate_logistic_probability_model
from src.features.leakage_checks import run_leakage_checks
from src.research.artifact_guardrails import require_mlb_tournament_path
from src.research.candidate_models import (
    BaseCandidateModel,
    DGLMMarginCandidate,
    GAMSplineCandidate,
    GLMMLogitCandidate,
    PenalizedLogitCandidate,
    VanillaGLMBinomialCandidate,
)
from src.research.mlb_targets import TargetDefinition, build_target_frames
from src.research.model_comparison import (
    FEATURE_POOL_FULL_SCREENED,
    FEATURE_POOL_PRODUCTION_MODEL_MAP,
    FEATURE_POOL_RESEARCH_BROAD,
    _select_feature_sets,
)
from src.services.train import load_features_dataframe
from src.training.feature_selection import select_feature_columns
from src.training.lambda_search import penalized_glm_search_grid
from src.training.model_feature_research import load_model_feature_map


CORE_LINEAR_FAMILIES = ("glm_vanilla", "glm_ridge", "glm_lasso", "glm_elastic_net")
EXTENSION_FAMILIES = ("glmm_logit", "dglm_margin", "gam_spline")
DEFAULT_NESTED_MODELS = CORE_LINEAR_FAMILIES + EXTENSION_FAMILIES
DEFAULT_STRUCTURED_SPEC_PATH = "configs/research/mlb_tournament_structured_glm.yaml"


@dataclass(frozen=True, slots=True)
class NestedTournamentResult:
    run_id: str
    artifact_root: Path
    summary_path: Path
    target_coverage_path: Path
    variant_leaderboard_path: Path
    family_champions_path: Path
    inter_family_leaderboard_path: Path
    current_best_models_path: Path


@dataclass(frozen=True, slots=True)
class FeatureVariant:
    variant_key: str
    display_name: str
    source: str
    features: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NestedCandidateSpec:
    target: TargetDefinition
    model_name: str
    display_name: str
    variant_key: str
    variant_display_name: str
    feature_source: str
    features: tuple[str, ...]
    param_grid: tuple[dict[str, Any], ...]
    builder: Callable[[dict[str, Any]], BaseCandidateModel]

    @property
    def candidate_key(self) -> str:
        return f"{self.target.target_name}::{self.model_name}::{self.variant_key}"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        numeric = float(value)
    except Exception:
        return None
    return numeric if np.isfinite(numeric) else None


def _safe_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    if isinstance(value, tuple):
        return [_json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if pd.isna(value) if not isinstance(value, (str, bytes, bool)) else False:
        return None
    return value


def _split_csv(raw_value: str | None, default: tuple[str, ...]) -> list[str]:
    raw = str(raw_value or "").strip()
    if not raw or raw.lower() in {"all", "*"}:
        return list(default)
    return [token.strip() for token in raw.split(",") if token.strip()]


def _time_ordered_split(
    df: pd.DataFrame,
    *,
    target_col: str,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df[df[target_col].notna()].copy().sort_values("start_time_utc").reset_index(drop=True)
    n_obs = len(work)
    if n_obs < 12:
        return work.iloc[:0].copy(), work.iloc[:0].copy(), work.iloc[:0].copy()
    train_end = int(round(train_fraction * n_obs))
    valid_end = int(round((train_fraction + validation_fraction) * n_obs))
    train_end = min(max(train_end, 1), max(n_obs - 2, 1))
    valid_end = min(max(valid_end, train_end + 1), n_obs - 1)
    return work.iloc[:train_end].copy(), work.iloc[train_end:valid_end].copy(), work.iloc[valid_end:].copy()


def _raw_features_for_pool(cfg: AppConfig, features_df: pd.DataFrame, *, feature_pool: str, feature_map_model: str) -> tuple[list[str], str]:
    all_raw_features = select_feature_columns(features_df)
    leakage_issues = run_leakage_checks(features_df, feature_columns=all_raw_features)
    if leakage_issues:
        raise RuntimeError(f"Leakage checks failed before nested tournament: {leakage_issues}")
    token = str(feature_pool or FEATURE_POOL_FULL_SCREENED).strip().lower()
    if token == FEATURE_POOL_PRODUCTION_MODEL_MAP:
        model_feature_map = load_model_feature_map(cfg.data.league)
        requested = [str(col) for col in model_feature_map.get(str(feature_map_model), []) if str(col).strip()]
        missing = [feature for feature in requested if feature not in all_raw_features]
        if missing:
            raise ValueError(f"Production feature map `{feature_map_model}` includes unavailable columns: {missing}")
        return requested, f"production_model_map:{feature_map_model}"
    if token in {FEATURE_POOL_FULL_SCREENED, FEATURE_POOL_RESEARCH_BROAD}:
        return all_raw_features, token
    raise ValueError(f"Unknown nested tournament feature pool `{feature_pool}`")


def _raw_features_for_target(target_df: pd.DataFrame, base_raw_features: list[str]) -> list[str]:
    target_features = select_feature_columns(target_df)
    combined = list(dict.fromkeys(list(base_raw_features) + target_features))
    leakage_issues = run_leakage_checks(target_df, feature_columns=combined)
    if leakage_issues:
        raise RuntimeError(f"Leakage checks failed after target/market enrichment: {leakage_issues}")
    return combined


def _load_features(cfg: AppConfig, *, feature_pool: str) -> pd.DataFrame:
    if str(feature_pool or "").strip().lower() == FEATURE_POOL_RESEARCH_BROAD:
        from src.common.research import resolve_research_paths

        research_paths = resolve_research_paths(cfg)
        return load_features_dataframe(str(research_paths.processed_dir))
    return load_features_dataframe(cfg.paths.processed_dir)


def _screened_feature_sets(fit_df: pd.DataFrame, *, target_col: str, raw_features: list[str]):
    work = fit_df.copy()
    work["home_win"] = pd.to_numeric(work[target_col], errors="coerce")
    return _select_feature_sets(work, raw_features)


def _resolve_spec_path(path_value: str | Path | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path if path.exists() else None


def _structured_variants(feature_sets: Any, *, spec_path: str | Path | None) -> list[FeatureVariant]:
    path = _resolve_spec_path(spec_path or DEFAULT_STRUCTURED_SPEC_PATH)
    screened_set = set(feature_sets.screened_features)
    screening_frame = getattr(feature_sets, "screening_frame", pd.DataFrame())
    screening_by_feature = (
        {
            str(row["feature"]): row
            for row in screening_frame.to_dict(orient="records")
            if str(row.get("feature") or "").strip()
        }
        if isinstance(screening_frame, pd.DataFrame) and not screening_frame.empty
        else {}
    )
    variants: list[FeatureVariant] = []

    def structured_feature(feature: str) -> str | None:
        if feature in screened_set:
            return feature
        row = screening_by_feature.get(feature)
        if not row:
            return None
        if str(row.get("reason") or "") == "constant_or_singleton_on_fit_window":
            return None
        retained = str(row.get("retained_as") or feature).strip()
        return retained if retained in screened_set else None

    if path is not None:
        payload = yaml.safe_load(path.read_text()) or {}
        slates = payload.get("slates") if isinstance(payload.get("slates"), dict) else {}
        for slate_name, slate_payload in slates.items():
            if not isinstance(slate_payload, dict):
                continue
            feature_order = [str(feature) for feature in slate_payload.get("feature_order", []) if str(feature).strip()]
            width_variants = slate_payload.get("width_variants") if isinstance(slate_payload.get("width_variants"), dict) else {}
            for width_name, width_payload in width_variants.items():
                count = int(width_payload.get("feature_count", 0) if isinstance(width_payload, dict) else 0)
                if count <= 0:
                    continue
                features = tuple(
                    dict.fromkeys(
                        resolved
                        for feature in feature_order[:count]
                        for resolved in [structured_feature(feature)]
                        if resolved
                    )
                )
                if features:
                    key = f"{slate_name}_{width_name}"
                    variants.append(
                        FeatureVariant(
                            variant_key=key,
                            display_name=key.replace("_", " ").title(),
                            source=f"structured:{path.name}",
                            features=features,
                        )
                    )

    ranked = list(feature_sets.ranking_frame["feature"]) if not feature_sets.ranking_frame.empty else list(feature_sets.screened_features)
    for count in (8, 12, 18):
        features = tuple(feature for feature in ranked[: min(count, len(ranked))] if feature in screened_set)
        if features:
            variants.append(
                FeatureVariant(
                    variant_key=f"screened_top_{count}",
                    display_name=f"Screened Top {count}",
                    source="screened_rank",
                    features=features,
                )
            )
    no_market = tuple(
        feature
        for feature in ranked
        if feature in screened_set
        and not any(token in feature.lower() for token in ("market", "vig", "odds", "price", "bookmaker"))
    )[:12]
    if no_market:
        variants.append(
            FeatureVariant(
                variant_key="no_market_top_12",
                display_name="No Market Top 12",
                source="screened_rank_no_market",
                features=no_market,
            )
        )

    deduped: dict[str, FeatureVariant] = {}
    for variant in variants:
        deduped.setdefault(variant.variant_key, variant)
    return list(deduped.values())


def _model_specs_for_variant(
    *,
    target: TargetDefinition,
    feature_sets: Any,
    variant: FeatureVariant,
    candidate_models: set[str] | None,
) -> list[NestedCandidateSpec]:
    specs: list[NestedCandidateSpec] = []

    def selected(name: str) -> bool:
        return candidate_models is None or name in candidate_models

    def optional_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        text = str(value).strip()
        if not text or text.lower() in {"none", "nan", "null"}:
            return None
        return float(value)

    if selected("glm_vanilla"):
        specs.append(
            NestedCandidateSpec(
                target=target,
                model_name="glm_vanilla",
                display_name="Vanilla GLM",
                variant_key=variant.variant_key,
                variant_display_name=variant.display_name,
                feature_source=variant.source,
                features=variant.features,
                param_grid=({},),
                builder=lambda params, features=variant.features: VanillaGLMBinomialCandidate(features=list(features)),
            )
        )
    for model_name, display_name, penalty, solver in (
        ("glm_ridge", "GLM Ridge", "l2", "lbfgs"),
        ("glm_lasso", "GLM Lasso", "l1", "saga"),
        ("glm_elastic_net", "Elastic Net GLM", "elasticnet", "saga"),
    ):
        if not selected(model_name):
            continue
        grid = tuple(penalized_glm_search_grid(model_name))
        specs.append(
            NestedCandidateSpec(
                target=target,
                model_name=model_name,
                display_name=display_name,
                variant_key=variant.variant_key,
                variant_display_name=variant.display_name,
                feature_source=variant.source,
                features=variant.features,
                param_grid=grid,
                builder=lambda params, model_name=model_name, display_name=display_name, penalty=penalty, solver=solver, features=variant.features: PenalizedLogitCandidate(
                    model_name=model_name,
                    display_name=display_name,
                    features=list(features),
                    penalty=penalty,
                    c=float(params["c"]),
                    l1_ratio=optional_float(params.get("l1_ratio")),
                    solver=solver,
                ),
            )
        )

    if selected("glmm_logit"):
        caps = sorted({cap for cap in (6, 10, 14) if cap <= len(feature_sets.glmm_features)}) or [len(feature_sets.glmm_features)]
        for cap in caps:
            features = tuple(feature_sets.glmm_features[:cap])
            specs.append(
                NestedCandidateSpec(
                    target=target,
                    model_name="glmm_logit",
                    display_name="GLMM Logit",
                    variant_key=f"glmm_cap_{cap}",
                    variant_display_name=f"GLMM Cap {cap}",
                    feature_source="grouped_effects",
                    features=features,
                    param_grid=({"feature_cap": cap},),
                    builder=lambda params, features=features: GLMMLogitCandidate(fixed_features=list(features)),
                )
            )
    if selected("dglm_margin") and target.target_name == "moneyline_home_win":
        caps = sorted({cap for cap in (6, 10, 14) if cap <= len(feature_sets.dglm_features)}) or [len(feature_sets.dglm_features)]
        for cap in caps:
            features = tuple(feature_sets.dglm_features[:cap])
            specs.append(
                NestedCandidateSpec(
                    target=target,
                    model_name="dglm_margin",
                    display_name="DGLM Margin",
                    variant_key=f"dglm_cap_{cap}",
                    variant_display_name=f"DGLM Cap {cap}",
                    feature_source="dispersion",
                    features=features,
                    param_grid=tuple({"feature_cap": cap, "iterations": iterations} for iterations in (1, 2)),
                    builder=lambda params, features=features: DGLMMarginCandidate(
                        features=list(features),
                        iterations=int(params["iterations"]),
                    ),
                )
            )
    if selected("gam_spline"):
        caps = sorted({cap for cap in (3, 5, 6) if cap <= len(feature_sets.gam_features)}) or [len(feature_sets.gam_features)]
        for cap in caps:
            spline_features = tuple(feature_sets.gam_features[:cap])
            linear_features = tuple(
                feature for feature in feature_sets.core_features if feature not in spline_features
            )[: max(4, cap + 2)]
            specs.append(
                NestedCandidateSpec(
                    target=target,
                    model_name="gam_spline",
                    display_name="GAM Spline",
                    variant_key=f"gam_cap_{cap}",
                    variant_display_name=f"GAM Cap {cap}",
                    feature_source="nonlinearity",
                    features=tuple(dict.fromkeys(linear_features + spline_features)),
                    param_grid=tuple(
                        {"feature_cap": cap, "n_knots": n_knots, "c": c}
                        for n_knots in (4, 5)
                        for c in (0.25, 0.5, 1.0)
                    ),
                    builder=lambda params, linear_features=linear_features, spline_features=spline_features: GAMSplineCandidate(
                        linear_features=list(linear_features),
                        spline_features=list(spline_features),
                        n_knots=int(params["n_knots"]),
                        c=float(params["c"]),
                    ),
                )
            )
    return specs


def _candidate_specs(
    *,
    target: TargetDefinition,
    feature_sets: Any,
    candidate_models: set[str] | None,
    structured_glm_spec_path: str | Path | None,
) -> list[NestedCandidateSpec]:
    variants = _structured_variants(feature_sets, spec_path=structured_glm_spec_path)
    specs: list[NestedCandidateSpec] = []
    for variant in variants:
        specs.extend(
            _model_specs_for_variant(
                target=target,
                feature_sets=feature_sets,
                variant=variant,
                candidate_models=candidate_models,
            )
        )
    return specs


def _params_text(params: dict[str, Any]) -> str:
    if not params:
        return "{}"
    return "; ".join(f"{key}={value}" for key, value in sorted(params.items()))


def _parse_params_text(params_text: Any) -> dict[str, Any]:
    if params_text is None:
        return {}
    if isinstance(params_text, float) and np.isnan(params_text):
        return {}
    text = str(params_text).strip()
    if not text or text == "{}" or text.lower() in {"nan", "none", "null"}:
        return {}
    parsed: dict[str, Any] = {}
    for token in text.split(";"):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        value_text = value.strip()
        if value_text.lower() in {"none", "nan", "null"}:
            parsed[key] = None
            continue
        try:
            parsed[key] = float(value_text)
        except ValueError:
            parsed[key] = value_text
    return parsed


def _time_series_splits(df: pd.DataFrame, *, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(df)
    if n < 24:
        return []
    split_count = max(2, int(n_splits))
    min_train = min(max(20, n // 2), max(n - 10, 10))
    validation_size = max(5, (n - min_train) // split_count)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(split_count):
        train_end = min_train + fold * validation_size
        valid_end = train_end + validation_size
        if fold == split_count - 1:
            valid_end = n
        if train_end >= n or valid_end <= train_end:
            continue
        splits.append((np.arange(0, train_end), np.arange(train_end, valid_end)))
    return splits


def _tune_candidate(
    spec: NestedCandidateSpec,
    *,
    fit_df: pd.DataFrame,
    cv_splits: int,
) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    splits = _time_series_splits(fit_df.sort_values("start_time_utc").reset_index(drop=True), n_splits=cv_splits)
    if not splits:
        return None, pd.DataFrame()
    ordered = fit_df.sort_values("start_time_utc").reset_index(drop=True)
    for params in spec.param_grid:
        for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
            tr = ordered.iloc[tr_idx].copy()
            va = ordered.iloc[va_idx].copy()
            if va[spec.target.target_col].nunique(dropna=True) < 2:
                continue
            try:
                model = spec.builder(dict(params))
                model.fit(tr, target_col=spec.target.target_col)
                p = model.predict_proba(va)
                metrics = metric_bundle(va[spec.target.target_col].astype(int).to_numpy(), p)
                rows.append(
                    {
                        "target_name": spec.target.target_name,
                        "candidate_key": spec.candidate_key,
                        "model_name": spec.model_name,
                        "variant_key": spec.variant_key,
                        "params": _params_text(dict(params)),
                        "fold": fold,
                        "status": "ok",
                        "mean_log_loss": float(metrics["log_loss"]),
                        "mean_brier": float(metrics["brier"]),
                        "mean_auc": float(metrics["auc"]),
                        "error": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "target_name": spec.target.target_name,
                        "candidate_key": spec.candidate_key,
                        "model_name": spec.model_name,
                        "variant_key": spec.variant_key,
                        "params": _params_text(dict(params)),
                        "fold": fold,
                        "status": "failed",
                        "mean_log_loss": np.nan,
                        "mean_brier": np.nan,
                        "mean_auc": np.nan,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    folds = pd.DataFrame(rows)
    ok = folds[folds["status"] == "ok"].copy() if not folds.empty else pd.DataFrame()
    if ok.empty:
        return None, folds
    summary = (
        ok.groupby("params", as_index=False)
        .agg(mean_log_loss=("mean_log_loss", "mean"), mean_brier=("mean_brier", "mean"), mean_auc=("mean_auc", "mean"))
        .sort_values(["mean_log_loss", "mean_brier", "mean_auc", "params"], ascending=[True, True, False, True])
    )
    params_text = str(summary.iloc[0]["params"])
    params = next((dict(params) for params in spec.param_grid if _params_text(dict(params)) == params_text), None)
    return params, folds


def _diagnostic_artifacts(
    *,
    y_true: np.ndarray,
    p_pred: np.ndarray,
    out_dir: Path,
    prefix: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    validation = validate_logistic_probability_model(
        y_true,
        p_pred,
        bins=10,
        plot_dir=out_dir / "plots",
        plot_prefix=prefix,
    )
    paths: dict[str, str] = {}
    for key in (
        "quantile_curve",
        "actual_vs_predicted_curve",
        "lift_curve",
        "lorenz_curve",
        "roc_curve",
        "operating_points",
        "tossup_sweep",
    ):
        value = validation.get(key)
        if isinstance(value, pd.DataFrame):
            path = out_dir / f"{prefix}_{key}.csv"
            value.to_csv(path, index=False)
            paths[key] = str(path)
    summaries = {
        "quantile_summary": validation.get("quantile_summary"),
        "actual_vs_predicted_summary": validation.get("actual_vs_predicted_summary"),
        "lift_summary": validation.get("lift_summary"),
        "lorenz_summary": validation.get("lorenz_summary"),
        "roc_summary": validation.get("roc_summary"),
        "tossup_summary": validation.get("tossup_summary"),
        "applicability_summary": validation.get("applicability_summary"),
    }
    summary_path = out_dir / f"{prefix}_summary.json"
    _safe_json(summary_path, summaries)
    paths["summary"] = str(summary_path)
    for plot_key, plot_path in dict(validation.get("plot_paths") or {}).items():
        paths[plot_key] = str(plot_path)
    return summaries, paths


def _metric_row(
    *,
    spec: NestedCandidateSpec,
    split: str,
    params: dict[str, Any],
    y_true: np.ndarray,
    p_pred: np.ndarray,
    fit_stats: Any,
    cv_folds: pd.DataFrame,
    diagnostic_summary: dict[str, Any],
    diagnostic_paths: dict[str, str],
) -> dict[str, Any]:
    metrics = metric_bundle(y_true, p_pred)
    calibration = calibration_alpha_beta(y_true, p_pred)
    quantile = dict(diagnostic_summary.get("quantile_summary") or {})
    lift = dict(diagnostic_summary.get("lift_summary") or {})
    lorenz = dict(diagnostic_summary.get("lorenz_summary") or {})
    roc = dict(diagnostic_summary.get("roc_summary") or {})
    applicability = dict(diagnostic_summary.get("applicability_summary") or {})
    cv_ok = cv_folds[cv_folds["status"] == "ok"].copy() if not cv_folds.empty else pd.DataFrame()
    cv_mean_log_loss = _safe_float(cv_ok["mean_log_loss"].mean()) if not cv_ok.empty else None
    validation_to_cv_delta = (float(metrics["log_loss"]) - cv_mean_log_loss) if cv_mean_log_loss is not None else None
    return {
        "target_name": spec.target.target_name,
        "target_col": spec.target.target_col,
        "market": spec.target.market,
        "split": split,
        "candidate_key": spec.candidate_key,
        "model_name": spec.model_name,
        "display_name": spec.display_name,
        "variant_key": spec.variant_key,
        "variant_display_name": spec.variant_display_name,
        "feature_source": spec.feature_source,
        "params": _params_text(params),
        "n_features": len(spec.features),
        "active_parameter_count": int(fit_stats.active_parameter_count) if fit_stats is not None else None,
        "log_loss": float(metrics["log_loss"]),
        "brier": float(metrics["brier"]),
        "accuracy": float(metrics["accuracy"]),
        "auc": _safe_float(metrics["auc"]),
        "ece": _safe_float(quantile.get("mean_abs_calibration_gap")),
        "calibration_alpha": _safe_float(calibration.get("calibration_alpha")),
        "calibration_beta": _safe_float(calibration.get("calibration_beta")),
        "max_abs_calibration_gap": _safe_float(quantile.get("max_abs_calibration_gap")),
        "top_vs_bottom_actual_lift_ratio": _safe_float(lift.get("top_vs_bottom_actual_lift_ratio")),
        "top_quantile_actual_lift": _safe_float(lift.get("top_quantile_actual_lift")),
        "normalized_gini": _safe_float(lorenz.get("normalized_gini")),
        "top_decile_event_capture": _safe_float(lorenz.get("top_decile_event_capture")),
        "roc_auc": _safe_float(roc.get("auroc")),
        "has_class_contrast": bool(applicability.get("has_class_contrast")),
        "cv_mean_log_loss": cv_mean_log_loss,
        "validation_to_cv_log_loss_delta": validation_to_cv_delta,
        "diagnostic_artifacts": diagnostic_paths,
        "fit_status": "ok",
        "fit_error": "",
    }


def _failed_row(spec: NestedCandidateSpec, *, split: str, error: str) -> dict[str, Any]:
    return {
        "target_name": spec.target.target_name,
        "target_col": spec.target.target_col,
        "market": spec.target.market,
        "split": split,
        "candidate_key": spec.candidate_key,
        "model_name": spec.model_name,
        "display_name": spec.display_name,
        "variant_key": spec.variant_key,
        "variant_display_name": spec.variant_display_name,
        "feature_source": spec.feature_source,
        "params": "",
        "n_features": len(spec.features),
        "active_parameter_count": None,
        "log_loss": np.nan,
        "brier": np.nan,
        "accuracy": np.nan,
        "auc": np.nan,
        "ece": np.nan,
        "calibration_alpha": np.nan,
        "calibration_beta": np.nan,
        "max_abs_calibration_gap": np.nan,
        "top_vs_bottom_actual_lift_ratio": np.nan,
        "top_quantile_actual_lift": np.nan,
        "normalized_gini": np.nan,
        "top_decile_event_capture": np.nan,
        "roc_auc": np.nan,
        "has_class_contrast": False,
        "cv_mean_log_loss": None,
        "validation_to_cv_log_loss_delta": None,
        "diagnostic_artifacts": {},
        "fit_status": "failed",
        "fit_error": error,
    }


def _evaluate_candidate(
    spec: NestedCandidateSpec,
    *,
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    params: dict[str, Any],
    split: str,
    cv_folds: pd.DataFrame,
    diagnostics_root: Path,
) -> tuple[dict[str, Any], pd.Series | None]:
    try:
        model = spec.builder(params)
        model.fit(fit_df, target_col=spec.target.target_col)
        p_pred = model.predict_proba(eval_df)
        y_true = eval_df[spec.target.target_col].astype(int).to_numpy()
        prefix = spec.candidate_key.replace("::", "__")
        diagnostic_summary, diagnostic_paths = _diagnostic_artifacts(
            y_true=y_true,
            p_pred=p_pred,
            out_dir=ensure_dir(diagnostics_root / spec.target.target_name / split / spec.model_name / spec.variant_key),
            prefix=prefix,
        )
        fit_stats = model.fit_statistics()
        row = _metric_row(
            spec=spec,
            split=split,
            params=params,
            y_true=y_true,
            p_pred=p_pred,
            fit_stats=fit_stats,
            cv_folds=cv_folds,
            diagnostic_summary=diagnostic_summary,
            diagnostic_paths=diagnostic_paths,
        )
        prediction = pd.Series(p_pred, name=spec.candidate_key)
        return row, prediction
    except Exception as exc:
        return _failed_row(spec, split=split, error=f"{type(exc).__name__}: {exc}"), None


def _gate_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if str(row.get("fit_status") or "") != "ok":
        reasons.append("fit_failed")
    if not bool(row.get("has_class_contrast")):
        reasons.append("no_class_contrast")
    ece = _safe_float(row.get("ece"))
    if ece is None or ece > 0.04:
        reasons.append("ece_gt_0.04")
    auc = _safe_float(row.get("auc"))
    if auc is None or auc < 0.50:
        reasons.append("auc_lt_0.50")
    alpha = _safe_float(row.get("calibration_alpha"))
    beta = _safe_float(row.get("calibration_beta"))
    if alpha is None or beta is None or abs(alpha) > 0.15 or abs(beta - 1.0) > 0.15:
        reasons.append("calibration_alpha_beta_outside_0.15")
    lift_ratio = _safe_float(row.get("top_vs_bottom_actual_lift_ratio"))
    if lift_ratio is None or lift_ratio <= 1.0:
        reasons.append("non_positive_top_bottom_lift")
    drift = _safe_float(row.get("validation_to_cv_log_loss_delta"))
    if drift is None or abs(drift) > 0.05:
        reasons.append("cv_validation_drift_gt_0.05")
    return reasons


def _select_family_champions(validation_rows: pd.DataFrame) -> pd.DataFrame:
    if validation_rows.empty:
        return pd.DataFrame()
    valid = validation_rows.copy()
    valid["gate_reasons"] = valid.apply(_gate_reasons, axis=1)
    valid["gate_passed"] = valid["gate_reasons"].map(lambda reasons: len(reasons) == 0)
    champions: list[dict[str, Any]] = []
    for (target_name, model_name), bucket in valid.groupby(["target_name", "model_name"], sort=True):
        passed = bucket[bucket["gate_passed"]].copy()
        fitted = bucket[bucket["fit_status"] == "ok"].copy()
        source = passed if not passed.empty else (fitted if not fitted.empty else bucket)
        ordered = source.sort_values(
            ["log_loss", "brier", "auc", "n_features", "active_parameter_count", "candidate_key"],
            ascending=[True, True, False, True, True, True],
            na_position="last",
        )
        row = ordered.iloc[0].to_dict()
        row["champion_status"] = "shortlist" if bool(row.get("gate_passed")) else "validation-blocked"
        row["family_champion"] = True
        champions.append(row)
    return pd.DataFrame(champions)


def _reason_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
    try:
        parsed = json.loads(text.replace("'", '"'))
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [token.strip() for token in text.strip("[]").split(",") if token.strip()]


def _write_validation_autopsy(
    *,
    artifact_root: Path,
    validation_leaderboard: pd.DataFrame,
    family_champions: pd.DataFrame,
    target_coverage: pd.DataFrame,
) -> dict[str, str]:
    autopsy_dir = ensure_dir(artifact_root / "validation_autopsy")
    champion_path = autopsy_dir / "family_champion_gate_autopsy.csv"
    gate_counts_path = autopsy_dir / "gate_failure_counts.csv"
    target_path = autopsy_dir / "target_gate_summary.csv"
    json_path = autopsy_dir / "validation_autopsy.json"
    markdown_path = autopsy_dir / "validation_autopsy.md"

    champion_rows: list[dict[str, Any]] = []
    gate_count_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    gate_counter: dict[tuple[str, str], int] = {}

    if not family_champions.empty:
        for row in family_champions.to_dict(orient="records"):
            reasons = _reason_list(row.get("gate_reasons"))
            target_name = str(row.get("target_name") or "")
            for reason in reasons:
                gate_counter[(target_name, reason)] = gate_counter.get((target_name, reason), 0) + 1
            champion_rows.append(
                {
                    "target_name": target_name,
                    "model_name": row.get("model_name"),
                    "variant_key": row.get("variant_key"),
                    "candidate_key": row.get("candidate_key"),
                    "champion_status": row.get("champion_status"),
                    "gate_passed": bool(row.get("gate_passed")),
                    "gate_failure_count": len(reasons),
                    "gate_reasons": ";".join(reasons),
                    "log_loss": row.get("log_loss"),
                    "brier": row.get("brier"),
                    "auc": row.get("auc"),
                    "ece": row.get("ece"),
                    "calibration_alpha": row.get("calibration_alpha"),
                    "calibration_beta": row.get("calibration_beta"),
                    "top_vs_bottom_actual_lift_ratio": row.get("top_vs_bottom_actual_lift_ratio"),
                    "validation_to_cv_log_loss_delta": row.get("validation_to_cv_log_loss_delta"),
                    "n_features": row.get("n_features"),
                    "active_parameter_count": row.get("active_parameter_count"),
                }
            )

    for (target_name, reason), count in sorted(gate_counter.items()):
        gate_count_rows.append({"target_name": target_name, "gate_reason": reason, "family_champion_count": count})

    if not target_coverage.empty:
        champion_df = pd.DataFrame(champion_rows)
        for coverage in target_coverage.to_dict(orient="records"):
            target_name = str(coverage.get("target_name") or "")
            target_champions = champion_df[champion_df["target_name"] == target_name] if not champion_df.empty else pd.DataFrame()
            target_rows.append(
                {
                    "target_name": target_name,
                    "coverage_status": coverage.get("status"),
                    "usable_rows": coverage.get("usable_rows"),
                    "train_rows": coverage.get("train_rows"),
                    "validation_rows": coverage.get("validation_rows"),
                    "test_rows": coverage.get("test_rows"),
                    "line_rows": coverage.get("line_rows"),
                    "family_champions": int(len(target_champions)),
                    "shortlisted_family_champions": int((target_champions.get("champion_status") == "shortlist").sum()) if not target_champions.empty else 0,
                    "validation_blocked_family_champions": int((target_champions.get("champion_status") == "validation-blocked").sum()) if not target_champions.empty else 0,
                }
            )

    champion_autopsy = pd.DataFrame(champion_rows)
    gate_counts = pd.DataFrame(gate_count_rows)
    target_summary = pd.DataFrame(target_rows)
    champion_autopsy.to_csv(champion_path, index=False)
    gate_counts.to_csv(gate_counts_path, index=False)
    target_summary.to_csv(target_path, index=False)

    payload = {
        "target_summary": target_rows,
        "gate_failure_counts": gate_count_rows,
        "family_champion_gate_autopsy": champion_rows,
        "variant_rows": int(len(validation_leaderboard)),
        "family_champion_rows": int(len(family_champions)),
    }
    _safe_json(json_path, payload)
    markdown_path.write_text(
        "\n".join(
            [
                "# MLB Nested Tournament Validation Autopsy",
                "",
                "## Target Summary",
                target_summary.to_string(index=False) if not target_summary.empty else "No target rows.",
                "",
                "## Family Champion Gate Failures",
                gate_counts.to_string(index=False) if not gate_counts.empty else "No gate failures.",
                "",
                "## Champion Details",
                champion_autopsy[
                    [
                        "target_name",
                        "model_name",
                        "variant_key",
                        "champion_status",
                        "gate_failure_count",
                        "gate_reasons",
                        "log_loss",
                        "auc",
                        "ece",
                    ]
                ].to_string(index=False)
                if not champion_autopsy.empty
                else "No family champions.",
            ]
        )
        + "\n"
    )
    return {
        "validation_autopsy_dir": str(autopsy_dir),
        "validation_autopsy_json": str(json_path),
        "validation_autopsy_report": str(markdown_path),
        "family_champion_gate_autopsy_path": str(champion_path),
        "gate_failure_counts_path": str(gate_counts_path),
        "target_gate_summary_path": str(target_path),
    }


def _bootstrap_against_best(predictions: pd.DataFrame, leaderboard: pd.DataFrame, *, random_seed: int, n_bootstrap: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if leaderboard.empty or "target_name" not in leaderboard.columns:
        return pd.DataFrame(rows)
    rng = np.random.default_rng(random_seed)
    for target_name, bucket in leaderboard.groupby("target_name", sort=True):
        ok = bucket[bucket["fit_status"] == "ok"].copy()
        if ok.empty:
            continue
        best = ok.sort_values(["log_loss", "brier", "auc"], ascending=[True, True, False]).iloc[0]
        best_key = str(best["candidate_key"])
        y_col = f"{target_name}__actual"
        if y_col not in predictions.columns or best_key not in predictions.columns:
            continue
        y = predictions[y_col].dropna().astype(int)
        best_scores = per_game_scores(y.to_numpy(), predictions.loc[y.index, best_key].to_numpy())
        for _, row in ok.iterrows():
            candidate_key = str(row["candidate_key"])
            if candidate_key == best_key or candidate_key not in predictions.columns:
                continue
            other_scores = per_game_scores(y.to_numpy(), predictions.loc[y.index, candidate_key].to_numpy())
            diffs = []
            n_obs = len(best_scores)
            for _ in range(max(100, int(n_bootstrap))):
                sample_idx = rng.integers(0, n_obs, size=n_obs)
                diffs.append(float(other_scores.iloc[sample_idx]["log_loss"].mean() - best_scores.iloc[sample_idx]["log_loss"].mean()))
            array = np.asarray(diffs, dtype=float)
            rows.append(
                {
                    "target_name": target_name,
                    "reference_candidate_key": best_key,
                    "comparison_candidate_key": candidate_key,
                    "delta_log_loss_mean": float(array.mean()),
                    "delta_log_loss_p025": float(np.quantile(array, 0.025)),
                    "delta_log_loss_p975": float(np.quantile(array, 0.975)),
                    "delta_log_loss_prob_reference_better": float(np.mean(array > 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _target_prediction_frame(eval_df: pd.DataFrame, *, target: TargetDefinition) -> pd.DataFrame:
    columns = [column for column in ("game_id", "game_date_utc", "start_time_utc", "home_team", "away_team") if column in eval_df.columns]
    out = eval_df[columns].copy().reset_index(drop=True)
    out[f"{target.target_name}__actual"] = eval_df[target.target_col].reset_index(drop=True)
    return out


def _evaluate_validation_lane(
    *,
    target_result: Any,
    model_name: str,
    raw_features: list[str],
    cv_splits: int,
    train_fraction: float,
    validation_fraction: float,
    structured_glm_spec_path: str | Path | None,
    diagnostics_root: Path,
    lane_root: Path,
) -> dict[str, Any]:
    target = target_result.definition
    target_df = target_result.frame
    train_df, validation_df, test_df = _time_ordered_split(
        target_df,
        target_col=target.target_col,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    variant_rows: list[dict[str, Any]] = []
    cv_frames: list[pd.DataFrame] = []
    if train_df.empty or validation_df.empty or test_df.empty:
        path = ensure_dir(lane_root) / "validation_leaderboard.csv"
        pd.DataFrame().to_csv(path, index=False)
        return {
            "target_name": target.target_name,
            "model_name": model_name,
            "validation_leaderboard_path": str(path),
            "variant_rows": variant_rows,
            "cv_folds": cv_frames,
            "status": "coverage-blocked",
        }
    feature_sets = _screened_feature_sets(train_df, target_col=target.target_col, raw_features=raw_features)
    specs = _candidate_specs(
        target=target,
        feature_sets=feature_sets,
        candidate_models={model_name},
        structured_glm_spec_path=structured_glm_spec_path,
    )
    for spec in specs:
        best_params, cv_folds = _tune_candidate(spec, fit_df=train_df, cv_splits=cv_splits)
        if not cv_folds.empty:
            cv_frames.append(cv_folds)
        if best_params is None:
            variant_rows.append(_failed_row(spec, split="validation", error="no_successful_cv_folds"))
            continue
        row, _ = _evaluate_candidate(
            spec,
            fit_df=train_df,
            eval_df=validation_df,
            params=best_params,
            split="validation",
            cv_folds=cv_folds,
            diagnostics_root=diagnostics_root,
        )
        variant_rows.append(row)
    path = ensure_dir(lane_root) / "validation_leaderboard.csv"
    pd.DataFrame(variant_rows).to_csv(path, index=False)
    return {
        "target_name": target.target_name,
        "model_name": model_name,
        "validation_leaderboard_path": str(path),
        "variant_rows": variant_rows,
        "cv_folds": cv_frames,
        "status": "ok",
    }


def run_mlb_nested_tournament(
    cfg: AppConfig,
    *,
    run_id: str | None = None,
    targets: str | None = None,
    candidate_models: str | None = None,
    feature_pool: str = FEATURE_POOL_FULL_SCREENED,
    feature_map_model: str = "glm_ridge",
    structured_glm_spec_path: str | None = DEFAULT_STRUCTURED_SPEC_PATH,
    train_fraction: float = 0.40,
    validation_fraction: float = 0.30,
    bootstrap_samples: int = 200,
) -> NestedTournamentResult:
    features_df = _load_features(cfg, feature_pool=feature_pool)
    raw_features, feature_pool_note = _raw_features_for_pool(
        cfg,
        features_df,
        feature_pool=feature_pool,
        feature_map_model=feature_map_model,
    )
    selected_models = set(_split_csv(candidate_models, DEFAULT_NESTED_MODELS)) if candidate_models else None
    target_results = build_target_frames(
        features_df,
        db_path=cfg.paths.db_path,
        league=cfg.data.league,
        targets=targets,
    )
    payload_for_hash = {
        "targets": [result.definition.target_name for result in target_results],
        "candidate_models": sorted(selected_models) if selected_models else list(DEFAULT_NESTED_MODELS),
        "feature_pool": feature_pool,
        "structured_glm_spec_path": structured_glm_spec_path,
        "bootstrap_samples": int(bootstrap_samples),
    }
    resolved_run_id = run_id or f"{_utc_stamp()}_{stable_hash(payload_for_hash)[:8]}"
    artifact_root = require_mlb_tournament_path(
        ensure_dir(Path(cfg.paths.artifacts_dir) / "reports" / "mlb" / "tournament" / "nested" / resolved_run_id),
        cfg.paths.artifacts_dir,
        purpose="MLB nested model tournament artifacts",
    )
    diagnostics_root = ensure_dir(artifact_root / "diagnostics")

    coverage_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    cv_frames: list[pd.DataFrame] = []
    final_prediction_frames: list[pd.DataFrame] = []

    for target_result in target_results:
        target = target_result.definition
        coverage = dict(target_result.coverage)
        target_df = target_result.frame
        train_df, validation_df, test_df = _time_ordered_split(
            target_df,
            target_col=target.target_col,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
        coverage.update(
            {
                "train_rows": int(len(train_df)),
                "validation_rows": int(len(validation_df)),
                "test_rows": int(len(test_df)),
            }
        )
        if coverage.get("status") != "ok" or train_df.empty or validation_df.empty or test_df.empty:
            coverage["status"] = "coverage-blocked"
            coverage["reason"] = coverage.get("reason") or "insufficient_target_split_rows"
            coverage_rows.append(coverage)
            continue
        coverage_rows.append(coverage)

        target_raw_features = _raw_features_for_target(target_df, raw_features)
        feature_sets = _screened_feature_sets(train_df, target_col=target.target_col, raw_features=target_raw_features)
        specs = _candidate_specs(
            target=target,
            feature_sets=feature_sets,
            candidate_models=selected_models,
            structured_glm_spec_path=structured_glm_spec_path,
        )
        for spec in specs:
            best_params, cv_folds = _tune_candidate(spec, fit_df=train_df, cv_splits=int(cfg.modeling.cv_splits))
            if not cv_folds.empty:
                cv_frames.append(cv_folds)
            if best_params is None:
                variant_rows.append(_failed_row(spec, split="validation", error="no_successful_cv_folds"))
                continue
            row, _ = _evaluate_candidate(
                spec,
                fit_df=train_df,
                eval_df=validation_df,
                params=best_params,
                split="validation",
                cv_folds=cv_folds,
                diagnostics_root=diagnostics_root,
            )
            variant_rows.append(row)

    target_coverage = pd.DataFrame(coverage_rows)
    validation_leaderboard = pd.DataFrame(variant_rows)
    family_champions = _select_family_champions(validation_leaderboard)
    inter_rows: list[dict[str, Any]] = []
    if not family_champions.empty:
        for _, champion in family_champions.iterrows():
            if champion.get("champion_status") == "coverage-blocked":
                continue
            if str(champion.get("fit_status") or "") != "ok":
                continue
            target_name = str(champion["target_name"])
            target_result = next(result for result in target_results if result.definition.target_name == target_name)
            target = target_result.definition
            target_df = target_result.frame
            train_df, validation_df, test_df = _time_ordered_split(
                target_df,
                target_col=target.target_col,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
            )
            fit_plus_validation = pd.concat([train_df, validation_df], ignore_index=True).sort_values("start_time_utc")
            target_raw_features = _raw_features_for_target(target_df, raw_features)
            feature_sets = _screened_feature_sets(fit_plus_validation, target_col=target.target_col, raw_features=target_raw_features)
            specs = _candidate_specs(
                target=target,
                feature_sets=feature_sets,
                candidate_models={str(champion["model_name"])},
                structured_glm_spec_path=structured_glm_spec_path,
            )
            spec = next((item for item in specs if item.candidate_key == str(champion["candidate_key"])), None)
            if spec is None:
                spec = next((item for item in specs if item.variant_key == str(champion["variant_key"])), None)
            if spec is None:
                continue
            params = _parse_params_text(champion.get("params"))
            cv_folds = pd.DataFrame()
            row, prediction = _evaluate_candidate(
                spec,
                fit_df=fit_plus_validation,
                eval_df=test_df,
                params=params,
                split="final_holdout",
                cv_folds=cv_folds,
                diagnostics_root=diagnostics_root,
            )
            row["intra_family_status"] = champion.get("champion_status")
            row["intra_family_gate_reasons"] = champion.get("gate_reasons")
            inter_rows.append(row)
            if prediction is not None:
                base = _target_prediction_frame(test_df, target=target)
                base[spec.candidate_key] = prediction.reset_index(drop=True)
                final_prediction_frames.append(base)

    inter_family = pd.DataFrame(inter_rows)
    if not inter_family.empty:
        inter_family = inter_family.sort_values(
            ["target_name", "log_loss", "brier", "auc", "candidate_key"],
            ascending=[True, True, True, False, True],
            na_position="last",
        ).reset_index(drop=True)
        inter_family["target_rank"] = inter_family.groupby("target_name").cumcount() + 1
    else:
        inter_family = pd.DataFrame(
            columns=[
                "target_name",
                "candidate_key",
                "model_name",
                "variant_key",
                "log_loss",
                "brier",
                "auc",
                "fit_status",
                "target_rank",
            ]
        )

    predictions = pd.DataFrame()
    if final_prediction_frames:
        predictions = final_prediction_frames[0]
        for frame in final_prediction_frames[1:]:
            merge_cols = [column for column in ("game_id", "game_date_utc", "start_time_utc", "home_team", "away_team") if column in predictions.columns and column in frame.columns]
            predictions = predictions.merge(frame, on=merge_cols, how="outer")
    bootstrap = _bootstrap_against_best(
        predictions,
        inter_family,
        random_seed=int(cfg.modeling.random_seed),
        n_bootstrap=int(bootstrap_samples),
    )

    target_coverage_path = artifact_root / "target_coverage.csv"
    variant_leaderboard_path = artifact_root / "variant_leaderboard.csv"
    family_champions_path = artifact_root / "family_champions.csv"
    inter_family_path = artifact_root / "inter_family_leaderboard.csv"
    predictions_path = artifact_root / "final_holdout_predictions.csv"
    cv_path = artifact_root / "cv_folds.csv"
    bootstrap_path = artifact_root / "bootstrap.csv"
    summary_path = artifact_root / "nested_tournament_summary.json"
    report_path = artifact_root / "nested_tournament_summary.md"
    current_best_path = Path(cfg.paths.artifacts_dir) / "reports" / "mlb" / "current_best_models.json"
    validation_autopsy_paths = _write_validation_autopsy(
        artifact_root=artifact_root,
        validation_leaderboard=validation_leaderboard,
        family_champions=family_champions,
        target_coverage=target_coverage,
    )

    target_coverage.to_csv(target_coverage_path, index=False)
    validation_leaderboard.to_csv(variant_leaderboard_path, index=False)
    family_champions.to_csv(family_champions_path, index=False)
    inter_family.to_csv(inter_family_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    (pd.concat(cv_frames, ignore_index=True) if cv_frames else pd.DataFrame()).to_csv(cv_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)

    champion_rows = []
    for target_name, bucket in inter_family.groupby("target_name", sort=True) if not inter_family.empty else []:
        top = bucket.sort_values(["target_rank"]).iloc[0].to_dict()
        champion_rows.append(top)
    current_best = {
        "league": str(cfg.data.league).upper(),
        "as_of_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_kind": "nested_mlb_model_tournament",
        "report_slug": resolved_run_id,
        "ranking_rule": {
            "primary": "validation_log_loss_inside_family_then_final_holdout_log_loss_between_family_champions",
            "gates": [
                "fit_status_ok",
                "class_contrast",
                "ece_lte_0.04",
                "calibration_alpha_abs_lte_0.15_and_beta_within_0.15_of_1.0",
                "auc_gte_0.50",
                "positive_top_bottom_lift",
                "cv_validation_drift_lte_0.05",
            ],
        },
        "target_champions": champion_rows,
        "top_models": champion_rows,
        "artifact_paths": {
            "summary_path": str(summary_path),
            "variant_leaderboard_path": str(variant_leaderboard_path),
            "family_champions_path": str(family_champions_path),
            "inter_family_leaderboard_path": str(inter_family_path),
            "target_coverage_path": str(target_coverage_path),
            **validation_autopsy_paths,
        },
    }
    ensure_dir(current_best_path.parent)
    _safe_json(current_best_path, current_best)

    summary_payload = {
        "league": str(cfg.data.league).upper(),
        "run_id": resolved_run_id,
        "generated_at_utc": current_best["as_of_utc"],
        "feature_pool_note": feature_pool_note,
        "targets": target_coverage.to_dict(orient="records"),
        "family_champions": family_champions.to_dict(orient="records"),
        "inter_family_leaderboard": inter_family.to_dict(orient="records"),
        "artifacts": current_best["artifact_paths"]
        | {
            "report_path": str(report_path),
            "predictions_path": str(predictions_path),
            "cv_path": str(cv_path),
            "bootstrap_path": str(bootstrap_path),
            "current_best_models_path": str(current_best_path),
        },
    }
    _safe_json(summary_path, summary_payload)

    report_lines = [
        "# MLB Nested Model Tournament",
        "",
        "This report uses nested selection: intra-family variants are selected on validation evidence before family champions spend the final holdout.",
        "",
        "## Target Coverage",
        target_coverage.to_string(index=False) if not target_coverage.empty else "No target coverage rows.",
        "",
        "## Family Champions",
        family_champions[["target_name", "model_name", "variant_key", "champion_status", "log_loss", "brier", "auc", "gate_reasons"]].to_string(index=False)
        if not family_champions.empty
        else "No family champions.",
        "",
        "## Inter-Family Final Holdout",
        inter_family[["target_name", "target_rank", "model_name", "variant_key", "log_loss", "brier", "auc", "intra_family_status"]].to_string(index=False)
        if not inter_family.empty
        else "No inter-family rows.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n")

    return NestedTournamentResult(
        run_id=resolved_run_id,
        artifact_root=artifact_root,
        summary_path=summary_path,
        target_coverage_path=target_coverage_path,
        variant_leaderboard_path=variant_leaderboard_path,
        family_champions_path=family_champions_path,
        inter_family_leaderboard_path=inter_family_path,
        current_best_models_path=current_best_path,
    )


def run_mlb_parallel_nested_tournament(
    cfg: AppConfig,
    *,
    run_id: str | None = None,
    targets: str | None = None,
    candidate_models: str | None = None,
    feature_pool: str = FEATURE_POOL_FULL_SCREENED,
    feature_map_model: str = "glm_ridge",
    structured_glm_spec_path: str | None = DEFAULT_STRUCTURED_SPEC_PATH,
    train_fraction: float = 0.40,
    validation_fraction: float = 0.30,
    bootstrap_samples: int = 200,
    max_workers: int = 4,
) -> NestedTournamentResult:
    features_df = _load_features(cfg, feature_pool=feature_pool)
    raw_features, feature_pool_note = _raw_features_for_pool(
        cfg,
        features_df,
        feature_pool=feature_pool,
        feature_map_model=feature_map_model,
    )
    selected_models = set(_split_csv(candidate_models, DEFAULT_NESTED_MODELS)) if candidate_models else set(DEFAULT_NESTED_MODELS)
    target_results = build_target_frames(
        features_df,
        db_path=cfg.paths.db_path,
        league=cfg.data.league,
        targets=targets,
    )
    payload_for_hash = {
        "targets": [result.definition.target_name for result in target_results],
        "candidate_models": sorted(selected_models),
        "feature_pool": feature_pool,
        "structured_glm_spec_path": structured_glm_spec_path,
        "bootstrap_samples": int(bootstrap_samples),
        "parallel": True,
    }
    resolved_run_id = run_id or f"{_utc_stamp()}_{stable_hash(payload_for_hash)[:8]}_parallel"
    artifact_root = require_mlb_tournament_path(
        ensure_dir(Path(cfg.paths.artifacts_dir) / "reports" / "mlb" / "tournament" / "nested" / resolved_run_id),
        cfg.paths.artifacts_dir,
        purpose="MLB parallel nested model tournament artifacts",
    )
    diagnostics_root = ensure_dir(artifact_root / "diagnostics")
    lane_runs_root = ensure_dir(artifact_root / "lane_runs")

    coverage_rows: list[dict[str, Any]] = []
    eligible_targets: list[Any] = []
    for target_result in target_results:
        target = target_result.definition
        coverage = dict(target_result.coverage)
        train_df, validation_df, test_df = _time_ordered_split(
            target_result.frame,
            target_col=target.target_col,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
        coverage.update({"train_rows": int(len(train_df)), "validation_rows": int(len(validation_df)), "test_rows": int(len(test_df))})
        if coverage.get("status") != "ok" or train_df.empty or validation_df.empty or test_df.empty:
            coverage["status"] = "coverage-blocked"
            coverage["reason"] = coverage.get("reason") or "insufficient_target_split_rows"
        else:
            eligible_targets.append(target_result)
        coverage_rows.append(coverage)

    lane_results: list[dict[str, Any]] = []
    lane_jobs: list[dict[str, Any]] = []
    for target_result in eligible_targets:
        for model_name in sorted(selected_models):
            lane_jobs.append({"target_result": target_result, "model_name": model_name})
    worker_count = max(1, min(int(max_workers), max(len(lane_jobs), 1)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _evaluate_validation_lane,
                target_result=job["target_result"],
                model_name=job["model_name"],
                raw_features=_raw_features_for_target(job["target_result"].frame, raw_features),
                cv_splits=int(cfg.modeling.cv_splits),
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                structured_glm_spec_path=structured_glm_spec_path,
                diagnostics_root=diagnostics_root,
                lane_root=lane_runs_root / job["target_result"].definition.target_name / job["model_name"],
            )
            for job in lane_jobs
        ]
        for future in as_completed(futures):
            lane_results.append(future.result())

    variant_rows = [row for lane in lane_results for row in lane.get("variant_rows", [])]
    cv_frames = [frame for lane in lane_results for frame in lane.get("cv_folds", [])]
    target_coverage = pd.DataFrame(coverage_rows)
    validation_leaderboard = pd.DataFrame(variant_rows)
    family_champions = _select_family_champions(validation_leaderboard)

    inter_rows: list[dict[str, Any]] = []
    final_prediction_frames: list[pd.DataFrame] = []
    if not family_champions.empty:
        for _, champion in family_champions.iterrows():
            if str(champion.get("fit_status") or "") != "ok":
                continue
            if str(champion.get("champion_status") or "") != "shortlist":
                continue
            target_name = str(champion["target_name"])
            target_result = next(result for result in target_results if result.definition.target_name == target_name)
            target = target_result.definition
            train_df, validation_df, test_df = _time_ordered_split(
                target_result.frame,
                target_col=target.target_col,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
            )
            fit_plus_validation = pd.concat([train_df, validation_df], ignore_index=True).sort_values("start_time_utc")
            target_raw_features = _raw_features_for_target(target_result.frame, raw_features)
            feature_sets = _screened_feature_sets(fit_plus_validation, target_col=target.target_col, raw_features=target_raw_features)
            specs = _candidate_specs(
                target=target,
                feature_sets=feature_sets,
                candidate_models={str(champion["model_name"])},
                structured_glm_spec_path=structured_glm_spec_path,
            )
            spec = next((item for item in specs if item.candidate_key == str(champion["candidate_key"])), None)
            if spec is None:
                spec = next((item for item in specs if item.variant_key == str(champion["variant_key"])), None)
            if spec is None:
                continue
            row, prediction = _evaluate_candidate(
                spec,
                fit_df=fit_plus_validation,
                eval_df=test_df,
                params=_parse_params_text(champion.get("params")),
                split="final_holdout",
                cv_folds=pd.DataFrame(),
                diagnostics_root=diagnostics_root,
            )
            row["intra_family_status"] = champion.get("champion_status")
            row["intra_family_gate_reasons"] = champion.get("gate_reasons")
            inter_rows.append(row)
            if prediction is not None:
                base = _target_prediction_frame(test_df, target=target)
                base[spec.candidate_key] = prediction.reset_index(drop=True)
                final_prediction_frames.append(base)

    inter_family = pd.DataFrame(inter_rows)
    if not inter_family.empty:
        inter_family = inter_family.sort_values(
            ["target_name", "log_loss", "brier", "auc", "candidate_key"],
            ascending=[True, True, True, False, True],
            na_position="last",
        ).reset_index(drop=True)
        inter_family["target_rank"] = inter_family.groupby("target_name").cumcount() + 1
    else:
        inter_family = pd.DataFrame(
            columns=[
                "target_name",
                "candidate_key",
                "model_name",
                "variant_key",
                "log_loss",
                "brier",
                "auc",
                "fit_status",
                "target_rank",
            ]
        )

    predictions = pd.DataFrame()
    if final_prediction_frames:
        predictions = final_prediction_frames[0]
        for frame in final_prediction_frames[1:]:
            merge_cols = [column for column in ("game_id", "game_date_utc", "start_time_utc", "home_team", "away_team") if column in predictions.columns and column in frame.columns]
            predictions = predictions.merge(frame, on=merge_cols, how="outer")
    bootstrap = _bootstrap_against_best(
        predictions,
        inter_family,
        random_seed=int(cfg.modeling.random_seed),
        n_bootstrap=int(bootstrap_samples),
    )

    target_coverage_path = artifact_root / "target_coverage.csv"
    variant_leaderboard_path = artifact_root / "variant_leaderboard.csv"
    family_champions_path = artifact_root / "family_champions.csv"
    inter_family_path = artifact_root / "inter_family_leaderboard.csv"
    predictions_path = artifact_root / "final_holdout_predictions.csv"
    cv_path = artifact_root / "cv_folds.csv"
    bootstrap_path = artifact_root / "bootstrap.csv"
    summary_path = artifact_root / "nested_tournament_summary.json"
    report_path = artifact_root / "nested_tournament_summary.md"
    current_best_path = Path(cfg.paths.artifacts_dir) / "reports" / "mlb" / "current_best_models.json"
    validation_autopsy_paths = _write_validation_autopsy(
        artifact_root=artifact_root,
        validation_leaderboard=validation_leaderboard,
        family_champions=family_champions,
        target_coverage=target_coverage,
    )

    target_coverage.to_csv(target_coverage_path, index=False)
    validation_leaderboard.to_csv(variant_leaderboard_path, index=False)
    family_champions.to_csv(family_champions_path, index=False)
    inter_family.to_csv(inter_family_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    (pd.concat(cv_frames, ignore_index=True) if cv_frames else pd.DataFrame()).to_csv(cv_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)

    champion_rows = [bucket.sort_values(["target_rank"]).iloc[0].to_dict() for _, bucket in inter_family.groupby("target_name", sort=True)] if not inter_family.empty else []
    blocked_champions = (
        family_champions[family_champions["champion_status"] != "shortlist"].to_dict(orient="records")
        if not family_champions.empty and "champion_status" in family_champions.columns
        else []
    )
    current_best = {
        "league": str(cfg.data.league).upper(),
        "as_of_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_kind": "parallel_nested_mlb_model_tournament",
        "report_slug": resolved_run_id,
        "target_champions": champion_rows,
        "blocked_family_champions": blocked_champions,
        "top_models": champion_rows,
        "artifact_paths": {
            "summary_path": str(summary_path),
            "variant_leaderboard_path": str(variant_leaderboard_path),
            "family_champions_path": str(family_champions_path),
            "inter_family_leaderboard_path": str(inter_family_path),
            "target_coverage_path": str(target_coverage_path),
            **validation_autopsy_paths,
        },
    }
    ensure_dir(current_best_path.parent)
    _safe_json(current_best_path, current_best)
    summary_payload = {
        "league": str(cfg.data.league).upper(),
        "run_id": resolved_run_id,
        "generated_at_utc": current_best["as_of_utc"],
        "orchestration": "parallel_intra_family_then_central_final_holdout",
        "max_workers": worker_count,
        "feature_pool_note": feature_pool_note,
        "targets": target_coverage.to_dict(orient="records"),
        "lane_runs": [{key: lane[key] for key in ("target_name", "model_name", "validation_leaderboard_path", "status")} for lane in lane_results],
        "family_champions": family_champions.to_dict(orient="records"),
        "inter_family_leaderboard": inter_family.to_dict(orient="records"),
        "artifacts": current_best["artifact_paths"]
        | {
            "report_path": str(report_path),
            "predictions_path": str(predictions_path),
            "cv_path": str(cv_path),
            "bootstrap_path": str(bootstrap_path),
            "current_best_models_path": str(current_best_path),
        },
    }
    _safe_json(summary_path, summary_payload)
    report_path.write_text(
        "\n".join(
            [
                "# MLB Parallel Nested Model Tournament",
                "",
                "Intra-family validation lanes were evaluated independently before central final-holdout ranking.",
                "",
                "## Target Coverage",
                target_coverage.to_string(index=False) if not target_coverage.empty else "No target coverage rows.",
                "",
                "## Family Champions",
                family_champions[["target_name", "model_name", "variant_key", "champion_status", "log_loss", "brier", "auc", "gate_reasons"]].to_string(index=False)
                if not family_champions.empty
                else "No family champions.",
                "",
                "## Inter-Family Final Holdout",
                inter_family[["target_name", "target_rank", "model_name", "variant_key", "log_loss", "brier", "auc", "intra_family_status"]].to_string(index=False)
                if not inter_family.empty
                else "No inter-family rows.",
            ]
        )
        + "\n"
    )
    return NestedTournamentResult(
        run_id=resolved_run_id,
        artifact_root=artifact_root,
        summary_path=summary_path,
        target_coverage_path=target_coverage_path,
        variant_leaderboard_path=variant_leaderboard_path,
        family_champions_path=family_champions_path,
        inter_family_leaderboard_path=inter_family_path,
        current_best_models_path=current_best_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the nested all-target MLB model tournament.")
    parser.add_argument("--config", default="configs/mlb.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--targets", default="all")
    parser.add_argument("--candidate-models", default=None)
    parser.add_argument("--feature-pool", default=FEATURE_POOL_FULL_SCREENED)
    parser.add_argument("--feature-map-model", default="glm_ridge")
    parser.add_argument("--structured-glm-spec", default=DEFAULT_STRUCTURED_SPEC_PATH)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = run_mlb_nested_tournament(
        cfg,
        run_id=args.run_id,
        targets=args.targets,
        candidate_models=args.candidate_models,
        feature_pool=args.feature_pool,
        feature_map_model=args.feature_map_model,
        structured_glm_spec_path=args.structured_glm_spec,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    print(f"MLB_NESTED_TOURNAMENT::{result.run_id}::{result.artifact_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
