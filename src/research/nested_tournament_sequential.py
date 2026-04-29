"""Sequential orchestration for the nested MLB model tournament."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import AppConfig
from src.common.utils import ensure_dir, stable_hash
from src.research.artifact_guardrails import require_mlb_tournament_path
from src.research.mlb_targets import build_target_frames
from src.research.model_comparison import FEATURE_POOL_FULL_SCREENED
from src.research.nested_tournament_candidates import _candidate_specs, _parse_params_text
from src.research.nested_tournament_contracts import (
    DEFAULT_NESTED_MODELS,
    DEFAULT_STRUCTURED_SPEC_PATH,
    NestedTournamentResult,
    _split_csv,
    _time_ordered_split,
    _utc_stamp,
)
from src.research.nested_tournament_evaluation import _evaluate_candidate, _failed_row, _tune_candidate
from src.research.nested_tournament_features import (
    _feature_coverage_rows,
    _load_features,
    _raw_features_for_pool,
    _raw_features_for_target,
    _screened_feature_sets,
    _target_prediction_frame,
)
from src.research.nested_tournament_governance import _select_family_champions
from src.research.nested_tournament_reporting import (
    _bootstrap_against_best,
    nested_tournament_artifact_paths,
    write_nested_tournament_artifacts,
)

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
    feature_coverage_rows: list[dict[str, Any]] = []

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
        target_raw_features = _raw_features_for_target(target_df, raw_features)
        feature_coverage_rows.extend(
            _feature_coverage_rows(
                target=target,
                split_frames={"train": train_df, "validation": validation_df, "final_holdout": test_df},
                features=target_raw_features,
            )
        )
        if coverage.get("status") != "ok" or train_df.empty or validation_df.empty or test_df.empty:
            coverage["status"] = "coverage-blocked"
            coverage["reason"] = coverage.get("reason") or "insufficient_target_split_rows"
            coverage_rows.append(coverage)
            continue
        coverage_rows.append(coverage)

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
            if str(champion.get("fit_status") or "") != "ok":
                continue
            if str(champion.get("champion_status") or "") != "shortlist":
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

    paths = nested_tournament_artifact_paths(
        artifact_root=artifact_root,
        artifacts_dir=cfg.paths.artifacts_dir,
    )
    return write_nested_tournament_artifacts(
        paths=paths,
        artifact_root=artifact_root,
        league=cfg.data.league,
        resolved_run_id=resolved_run_id,
        source_kind="nested_mlb_model_tournament",
        target_coverage=target_coverage,
        validation_leaderboard=validation_leaderboard,
        family_champions=family_champions,
        inter_family=inter_family,
        predictions=predictions,
        cv_frames=cv_frames,
        bootstrap=bootstrap,
        feature_coverage_rows=feature_coverage_rows,
        feature_pool_note=feature_pool_note,
        report_title="MLB Nested Model Tournament",
        report_intro=(
            "This report uses nested selection: intra-family variants are selected on validation "
            "evidence before family champions spend the final holdout."
        ),
        ranking_rule={
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
    )
