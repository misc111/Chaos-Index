"""Parallel intra-family orchestration for nested MLB tournaments."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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
    _safe_json,
    _split_csv,
    _time_ordered_split,
    _utc_stamp,
)
from src.research.nested_tournament_evaluation import _evaluate_candidate, _evaluate_validation_lane
from src.research.nested_tournament_features import (
    _feature_coverage_rows,
    _feature_coverage_summary,
    _load_features,
    _raw_features_for_pool,
    _raw_features_for_target,
    _screened_feature_sets,
    _target_prediction_frame,
)
from src.research.nested_tournament_governance import _select_family_champions
from src.research.nested_tournament_reporting import _bootstrap_against_best, _write_validation_autopsy

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
    feature_coverage_rows: list[dict[str, Any]] = []
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
        target_raw_features = _raw_features_for_target(target_result.frame, raw_features)
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
    feature_coverage_path = artifact_root / "feature_coverage_by_split.csv"
    feature_coverage_json_path = artifact_root / "feature_coverage_by_split.json"
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
    feature_coverage = pd.DataFrame(feature_coverage_rows)
    feature_coverage.to_csv(feature_coverage_path, index=False)
    _safe_json(
        feature_coverage_json_path,
        {
            "rows": feature_coverage.to_dict(orient="records"),
            "summary": _feature_coverage_summary(feature_coverage_rows),
        },
    )

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
            "feature_coverage_by_split_path": str(feature_coverage_path),
            "feature_coverage_by_split_json": str(feature_coverage_json_path),
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
