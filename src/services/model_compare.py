"""Standalone research comparison service for candidate predictive models."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.common.config import AppConfig
from src.common.logging import get_logger
from src.common.time import utc_now_iso
from src.common.utils import ensure_dir
from src.governance.evidence import (
    BETTING_OVERLAY,
    BETTING_OVERLAY_LANE,
    GOVERNANCE_CONTRACT_VERSION,
    THEORY_COMPATIBLE_ENGINEERING_SUPPORT,
    model_evidence_fields,
    target_scope_fields,
)
from src.research.artifact_guardrails import require_mlb_report_path
from src.research.model_comparison import run_candidate_model_comparison
from src.storage.db import Database
from src.storage.tracker import RunTracker

logger = get_logger(__name__)

_MLB_REPORT_LANE = "mlb"
_MLB_LATEST_MATERIAL_ARTIFACT_KEYS = (
    "report_path",
    "summary_path",
    "candidate_scorecards_path",
    "candidate_scorecards_contract_path",
    "recommendation_path",
    "recommendation_surface_path",
    "leaderboard_json_path",
)
_CURRENT_BEST_MODELS_FILE_NAME = "current_best_models.json"


def _load_summary_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def current_best_models_path(cfg: AppConfig) -> Path:
    output_dir = require_mlb_report_path(
        ensure_dir(Path(cfg.paths.artifacts_dir) / "reports" / _MLB_REPORT_LANE),
        cfg.paths.artifacts_dir,
        purpose="MLB current best models artifact",
    )
    return output_dir / _CURRENT_BEST_MODELS_FILE_NAME


def _market_name_for_target(target_name: str | None) -> str | None:
    token = str(target_name or "").strip().lower()
    if not token:
        return None
    if token.startswith("moneyline"):
        return "moneyline"
    if token.startswith("runline"):
        return "runline"
    if token.startswith("totals"):
        return "totals"
    return token


def _comparison_ranking_rule() -> dict[str, Any]:
    return {
        "primary": "final_holdout_log_loss",
        "secondary": "final_holdout_brier",
        "tertiary": "final_holdout_auc_desc",
        "gates": [],
        "notes": "Lower log loss wins, then lower Brier, then higher AUC. This is the candidate comparison contract.",
    }


def _tournament_ranking_rule() -> dict[str, Any]:
    return {
        "primary": "log_loss",
        "secondary": "brier",
        "tertiary": "calibration_flag_asc",
        "quaternary": "stability_flag_asc",
        "notes": "Tournament ranking starts with scoring, then prefers cleaner calibration and stability. ROI is a betting overlay and does not participate in theory-lane ordering.",
    }


def _comparison_top_models(summary_payload: dict[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    recommended_model = str(summary_payload.get("promotion_decision", {}).get("recommended_model") or "")
    for record in list(summary_payload.get("candidate_scorecards", [])):
        validation_metrics = dict(record.get("validation_metrics") or {})
        stability_metrics = dict(record.get("stability_metrics") or {})
        calibration_summary = dict(record.get("calibration_summary") or {})
        complement_summary = dict(record.get("complement_summary") or {})
        target_name = str(record.get("target_name") or summary_payload.get("target_name") or "moneyline_home_win")
        rows.append(
            {
                "rank": stability_metrics.get("final_holdout_rank"),
                "model_name": str(record.get("model_name") or ""),
                "display_name": str(complement_summary.get("display_name") or record.get("model_name") or ""),
                "target_name": target_name,
                "market": _market_name_for_target(target_name),
                "family": complement_summary.get("family"),
                "governance_note": complement_summary.get("governance_note"),
                "recommendation_tier": complement_summary.get("recommendation_tier"),
                "recommended_for_next_stage": bool(complement_summary.get("recommended_for_next_stage")),
                "is_recommended_model": str(record.get("model_name") or "") == recommended_model,
                "final_holdout_log_loss": _safe_float(validation_metrics.get("final_holdout_log_loss")),
                "final_holdout_brier": _safe_float(validation_metrics.get("final_holdout_brier")),
                "final_holdout_auc": _safe_float(validation_metrics.get("final_holdout_auc")),
                "final_holdout_ece": _safe_float(validation_metrics.get("final_holdout_ece")),
                "validation_log_loss": _safe_float(validation_metrics.get("validation_log_loss")),
                "validation_brier": _safe_float(validation_metrics.get("validation_brier")),
                "validation_auc": _safe_float(validation_metrics.get("validation_auc")),
                **model_evidence_fields(
                    str(record.get("model_name") or ""),
                    target_name=target_name,
                    target_col="home_win",
                    market=_market_name_for_target(target_name),
                ),
            }
        )

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        rank = row.get("rank")
        rank_bucket = 0 if isinstance(rank, int) else 1
        return (
            rank_bucket,
            rank if isinstance(rank, int) else 10**9,
            row.get("final_holdout_log_loss") if row.get("final_holdout_log_loss") is not None else float("inf"),
            row.get("final_holdout_brier") if row.get("final_holdout_brier") is not None else float("inf"),
            -(row.get("final_holdout_auc") if row.get("final_holdout_auc") is not None else float("-inf")),
            row.get("model_name") or "",
        )

    ordered = sorted(rows, key=sort_key)
    for index, row in enumerate(ordered, start=1):
        if not isinstance(row.get("rank"), int):
            row["rank"] = index
    return ordered[:limit]


def _build_current_best_models_payload_from_comparison(
    *,
    summary_payload: dict[str, Any],
    canonical_output_dir: Path,
) -> dict[str, Any]:
    decision = dict(summary_payload.get("promotion_decision") or {})
    metadata = dict(summary_payload.get("metadata") or {})
    artifacts = dict(summary_payload.get("artifacts") or {})
    evidence = dict(decision.get("evidence") or {})
    execution_metadata = dict(metadata.get("execution_metadata") or evidence.get("execution_metadata") or {})
    top_models = _comparison_top_models(summary_payload)
    benchmark = dict(evidence.get("intercept_only_benchmark") or {})
    target_name = str(summary_payload.get("target_name") or "moneyline_home_win")

    return {
        "governance_contract_version": GOVERNANCE_CONTRACT_VERSION,
        "evidence_scope": "candidate_model_comparison",
        "theory_governance": THEORY_COMPATIBLE_ENGINEERING_SUPPORT,
        "target_scope": target_scope_fields(
            league=str(summary_payload.get("league") or "MLB"),
            target_name=target_name,
            target_col="home_win",
            market=_market_name_for_target(target_name),
        ),
        "league": str(summary_payload.get("league") or "MLB").upper(),
        "as_of_utc": utc_now_iso(),
        "question_scope": "latest_completed_run",
        "source_kind": "candidate_model_comparison",
        "source_status": str(decision.get("status") or "candidate_screen_complete"),
        "report_slug": str(summary_payload.get("report_slug") or ""),
        "experiment_run_id": metadata.get("experiment_run_id"),
        "canonical_artifact_root": str(canonical_output_dir),
        "production_grade": execution_metadata.get("production_grade"),
        "execution_scope": execution_metadata.get("execution_data_scope"),
        "target_name": target_name,
        "market": _market_name_for_target(target_name),
        "ranking_rule": _comparison_ranking_rule(),
        "recommended_model": str(decision.get("recommended_model") or ""),
        "recommended_display_name": str(metadata.get("recommended_display_name") or decision.get("recommended_model") or ""),
        "best_candidate_model": top_models[0]["model_name"] if top_models else None,
        "best_candidate_display_name": top_models[0]["display_name"] if top_models else None,
        "baseline_model": str(decision.get("baseline_model") or "intercept_only"),
        "promotion_ready": evidence.get("promotion_ready"),
        "why_this_won": str(decision.get("rationale") or ""),
        "closest_challenger": top_models[1] if len(top_models) > 1 else None,
        "benchmark": {
            "model_name": benchmark.get("model_name"),
            "display_name": benchmark.get("display_name"),
            "log_loss": _safe_float(benchmark.get("log_loss")),
            "brier": _safe_float(benchmark.get("brier")),
            "auc": _safe_float(benchmark.get("auc")),
            "ece": _safe_float(benchmark.get("ece")),
        },
        "top_models": top_models,
        "artifact_paths": {
            "report_path": artifacts.get("report_path"),
            "summary_path": artifacts.get("summary_path"),
            "recommendation_path": artifacts.get("recommendation_path"),
            "recommendation_surface_path": artifacts.get("recommendation_surface_path"),
            "leaderboard_json_path": artifacts.get("leaderboard_json_path"),
        },
    }


def write_current_best_models_from_comparison(
    cfg: AppConfig,
    *,
    summary_payload: dict[str, Any],
    canonical_output_dir: Path | None = None,
) -> Path | None:
    league = str(summary_payload.get("league") or cfg.data.league).upper()
    if league != "MLB":
        return None
    output_dir = canonical_output_dir or current_best_models_path(cfg).parent
    payload = _build_current_best_models_payload_from_comparison(
        summary_payload=summary_payload,
        canonical_output_dir=output_dir,
    )
    output_path = output_dir / _CURRENT_BEST_MODELS_FILE_NAME
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def write_current_best_models_from_tournament(
    cfg: AppConfig,
    *,
    run_id: str,
    artifact_root: Path,
    summary_path: Path,
    leaderboard_rows: list[dict[str, Any]],
    recommendation_rows: list[dict[str, Any]],
    recommendation_path: Path,
    production_grade: bool | None = False,
    execution_scope: str | None = "bounded_mlb_feature_slice",
) -> Path:
    output_path = current_best_models_path(cfg)
    top_rows = list(leaderboard_rows)[:5]
    best_recommendation = recommendation_rows[0] if recommendation_rows else {}
    recommended_model = str(best_recommendation.get("model_name") or (top_rows[0].get("model_name") if top_rows else ""))
    recommended_display_name = str(
        best_recommendation.get("display_name") or (top_rows[0].get("display_name") if top_rows else recommended_model)
    )
    payload = {
        "governance_contract_version": GOVERNANCE_CONTRACT_VERSION,
        "evidence_scope": "mlb_model_tournament",
        "theory_governance": THEORY_COMPATIBLE_ENGINEERING_SUPPORT,
        "target_scope": target_scope_fields(
            league="MLB",
            target_name="moneyline_home_win",
            target_col="home_win",
            market="moneyline",
        ),
        "league": "MLB",
        "as_of_utc": utc_now_iso(),
        "question_scope": "latest_completed_tournament",
        "source_kind": "mlb_model_tournament",
        "source_status": "complete",
        "run_id": run_id,
        "canonical_artifact_root": str(output_path.parent),
        "artifact_root": str(artifact_root),
        "production_grade": production_grade,
        "execution_scope": execution_scope,
        "target_name": "moneyline_home_win",
        "market": "moneyline",
        "ranking_rule": _tournament_ranking_rule(),
        "recommended_model": recommended_model,
        "recommended_display_name": recommended_display_name,
        "best_candidate_model": top_rows[0].get("model_name") if top_rows else None,
        "best_candidate_display_name": top_rows[0].get("display_name") if top_rows else None,
        "baseline_model": "intercept_only",
        "promotion_ready": False,
        "why_this_won": str(best_recommendation.get("rationale") or "Top tournament leaderboard row under the current scoring contract."),
        "closest_challenger": top_rows[1] if len(top_rows) > 1 else None,
        "top_models": top_rows,
        "recommendations": recommendation_rows[:5],
        "artifact_paths": {
            "summary_path": str(summary_path),
            "recommendation_path": str(recommendation_path),
            "leaderboard_json_path": str(artifact_root / "leaderboard.json"),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def _promotion_review_top_models(promotion_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in list(promotion_payload.get("candidate_scorecards") or []):
        if not isinstance(record, dict):
            continue
        validation_metrics = dict(record.get("validation_metrics") or {})
        stability_metrics = dict(record.get("stability_metrics") or {})
        calibration_summary = dict(record.get("calibration_summary") or {})
        complement_summary = dict(record.get("complement_summary") or {})
        target_name = str(record.get("target_name") or "moneyline_home_win")
        model_name = str(record.get("model_name") or "")
        rows.append(
            {
                "rank": stability_metrics.get("scorecard_rank"),
                "model_name": model_name,
                "display_name": str(complement_summary.get("display_name") or model_name),
                "target_name": target_name,
                "market": _market_name_for_target(target_name),
                "family": complement_summary.get("family"),
                "governance_note": complement_summary.get("governance_note"),
                "promotion_role": complement_summary.get("promotion_role"),
                "strategy": validation_metrics.get("strategy") or promotion_payload.get("strategy"),
                "mean_ending_bankroll": _safe_float(validation_metrics.get("mean_ending_bankroll")),
                "mean_net_profit": _safe_float(validation_metrics.get("mean_net_profit")),
                "mean_roi": _safe_float(validation_metrics.get("mean_roi")),
                "median_roi": _safe_float(validation_metrics.get("median_roi")),
                "mean_log_loss": _safe_float(validation_metrics.get("mean_log_loss")),
                "mean_brier": _safe_float(validation_metrics.get("mean_brier")),
                "mean_auc": _safe_float(validation_metrics.get("mean_auc")),
                "mean_ece": _safe_float(calibration_summary.get("mean_ece")),
                "bet_count": validation_metrics.get("bet_count"),
                "profitable_folds": stability_metrics.get("profitable_folds"),
                "profit_winning_folds": stability_metrics.get("profit_winning_folds"),
                "all_integrity_checks": bool(stability_metrics.get("all_integrity_checks")),
                **model_evidence_fields(
                    model_name,
                    target_name=target_name,
                    target_col="home_win",
                    market=_market_name_for_target(target_name),
                ),
            }
        )

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        rank = row.get("rank")
        rank_bucket = 0 if isinstance(rank, int) else 1
        return (
            rank_bucket,
            rank if isinstance(rank, int) else 10**9,
            -(row.get("mean_ending_bankroll") if row.get("mean_ending_bankroll") is not None else float("-inf")),
            row.get("model_name") or "",
        )

    ordered = sorted(rows, key=sort_key)
    for index, row in enumerate(ordered, start=1):
        if not isinstance(row.get("rank"), int):
            row["rank"] = index
    return ordered[:5]


def write_current_best_models_from_promotion_review(
    cfg: AppConfig,
    *,
    run_id: str,
    promotion_payload: dict[str, Any],
    canonical_output_dir: Path | None = None,
) -> Path | None:
    league = str(promotion_payload.get("league") or cfg.data.league).upper()
    if league != "MLB":
        return None
    output_path = current_best_models_path(cfg)
    output_dir = canonical_output_dir or output_path.parent
    top_models = _promotion_review_top_models(promotion_payload)
    artifacts = dict(promotion_payload.get("artifacts") or {})
    decision = dict(promotion_payload.get("promotion_decision") or {})
    target_name = str(decision.get("target_name") or "moneyline_home_win")
    payload = {
        "governance_contract_version": GOVERNANCE_CONTRACT_VERSION,
        "evidence_scope": "research_desk_promotion_review",
        "theory_governance": BETTING_OVERLAY,
        "governance_lane": BETTING_OVERLAY_LANE,
        "target_scope": target_scope_fields(
            league=league,
            target_name=target_name,
            target_col="home_win",
            market=_market_name_for_target(target_name),
        ),
        "league": league,
        "as_of_utc": utc_now_iso(),
        "question_scope": "latest_promotion_review",
        "source_kind": "research_desk_promotion_review",
        "source_status": str(promotion_payload.get("status") or "rejected"),
        "run_id": run_id,
        "canonical_artifact_root": str(output_dir),
        "production_grade": True,
        "execution_scope": "full_immutable_pregame_mlb_research_backtest",
        "target_name": target_name,
        "market": _market_name_for_target(target_name),
        "ranking_rule": {
            "primary": "mean_ending_bankroll",
            "secondary": "mean_net_profit",
            "tertiary": "integrity_and_calibration_gates",
            "notes": "Promotion review applies betting-overlay economics only after immutable pregame integrity checks; model-family theory labels are carried per row.",
        },
        "recommended_model": str(promotion_payload.get("active_model_name") or decision.get("recommended_model") or ""),
        "recommended_display_name": str(
            next(
                (
                    row.get("display_name")
                    for row in top_models
                    if row.get("model_name") == (promotion_payload.get("active_model_name") or decision.get("recommended_model"))
                ),
                promotion_payload.get("active_model_name") or decision.get("recommended_model") or "",
            )
        ),
        "candidate_model_name": str(promotion_payload.get("candidate_model_name") or ""),
        "incumbent_model_name": str(promotion_payload.get("incumbent_model_name") or ""),
        "best_candidate_model": str(promotion_payload.get("candidate_model_name") or ""),
        "baseline_model": str(promotion_payload.get("incumbent_model_name") or promotion_payload.get("baseline_model") or ""),
        "promotion_ready": bool(promotion_payload.get("promoted")),
        "promoted": bool(promotion_payload.get("promoted")),
        "why_this_won": str(promotion_payload.get("reason_summary") or decision.get("rationale") or ""),
        "failed_reasons": list(promotion_payload.get("failed_reasons") or []),
        "gates": dict(promotion_payload.get("gates") or {}),
        "policy": dict(promotion_payload.get("policy") or {}),
        "top_models": top_models,
        "promotion_decision": decision,
        "artifact_paths": {
            "report_path": artifacts.get("report_path"),
            "scorecard_csv": artifacts.get("scorecard_csv"),
            "scorecard_contract_json": artifacts.get("scorecard_contract_json"),
            "fold_metrics_path": artifacts.get("fold_metrics_path"),
            "promotion_path": artifacts.get("promotion_path"),
            "market_truth_summary_path": artifacts.get("market_truth_summary_path"),
            "market_truth_predictions_path": artifacts.get("market_truth_predictions_path"),
            "market_truth_calibration_path": artifacts.get("market_truth_calibration_path"),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    latest_path = output_dir / "promotion_review_latest.json"
    manifest_path = output_dir / "promotion_review_latest_manifest.json"
    latest_payload = {
        "league": league,
        "report_lane": _MLB_REPORT_LANE,
        "canonical_artifact_root": str(output_dir),
        "generated_at_utc": utc_now_iso(),
        "source_kind": "research_desk_promotion_review",
        "source_status": payload["source_status"],
        "run_id": run_id,
        "candidate_model_name": payload["candidate_model_name"],
        "incumbent_model_name": payload["incumbent_model_name"],
        "active_model_name": payload["recommended_model"],
        "promoted": payload["promoted"],
        "reason_summary": payload["why_this_won"],
        "current_best_models_path": str(output_path),
        "material_artifacts": payload["artifact_paths"],
    }
    manifest_payload = {
        "league": league,
        "report_lane": _MLB_REPORT_LANE,
        "canonical_artifact_root": str(output_dir),
        "generated_at_utc": utc_now_iso(),
        "source_kind": "research_desk_promotion_review",
        "run_id": run_id,
        "current_best_models_path": str(output_path),
        "service_output_path": str(latest_path),
        "material_artifacts": payload["artifact_paths"],
    }
    latest_path.write_text(json.dumps(latest_payload, indent=2, sort_keys=True) + "\n")
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n")
    return output_path


def load_current_best_models(cfg: AppConfig) -> dict[str, Any]:
    path = current_best_models_path(cfg)
    if path.exists():
        return json.loads(path.read_text())

    db = Database(cfg.paths.db_path)
    db.init_schema()
    rows = db.query(
        """
        SELECT summary_json
        FROM experiment_runs
        WHERE league = ?
        ORDER BY COALESCE(completed_at_utc, started_at_utc) DESC
        LIMIT 1
        """,
        (str(cfg.data.league).upper(),),
    )
    if not rows:
        raise FileNotFoundError(f"No current best models artifact or experiment runs found for league {cfg.data.league!r}.")
    summary_payload = json.loads(rows[0]["summary_json"])
    payload = _build_current_best_models_payload_from_comparison(
        summary_payload=summary_payload,
        canonical_output_dir=path.parent,
    )
    return payload


def _artifact_sources(result, artifacts: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_path": artifacts.get("report_path") or getattr(result, "report_path", None),
        "summary_path": artifacts.get("summary_path") or getattr(result, "summary_path", None),
        "validation_metrics_path": artifacts.get("validation_metrics_path")
        or getattr(result, "validation_metrics_path", None),
        "test_metrics_path": artifacts.get("test_metrics_path") or getattr(result, "test_metrics_path", None),
        "bootstrap_path": artifacts.get("bootstrap_path") or getattr(result, "bootstrap_path", None),
        "fit_stats_path": artifacts.get("fit_stats_path") or getattr(result, "fit_stats_path", None),
        "cv_path": artifacts.get("cv_path") or getattr(result, "cv_path", None),
        "screening_path": artifacts.get("screening_path"),
        "nonlinearity_path": artifacts.get("nonlinearity_path"),
        "validation_predictions_path": artifacts.get("validation_predictions_path"),
        "test_predictions_path": artifacts.get("test_predictions_path"),
        "candidate_scorecards_path": artifacts.get("candidate_scorecards_path")
        or getattr(result, "candidate_scorecards_path", None),
        "candidate_scorecards_contract_path": artifacts.get("candidate_scorecards_contract_path")
        or getattr(result, "candidate_scorecards_contract_path", None),
        "leaderboard_path": artifacts.get("leaderboard_path") or getattr(result, "leaderboard_path", None),
        "leaderboard_json_path": artifacts.get("leaderboard_json_path")
        or getattr(result, "leaderboard_json_path", None),
        "recommendation_path": artifacts.get("recommendation_path") or getattr(result, "recommendation_path", None),
        "recommendation_surface_path": artifacts.get("recommendation_surface_path")
        or getattr(result, "recommendation_surface_path", None),
        "artifact_manifest_path": artifacts.get("artifact_manifest_path")
        or getattr(result, "artifact_manifest_path", None),
    }


def _copy_to_mlb_report_lane(*, source: Any, output_dir: Path) -> Path:
    source_path = Path(source)
    target_path = output_dir / source_path.name
    if source_path.expanduser().resolve() == target_path.expanduser().resolve():
        if not source_path.exists():
            raise FileNotFoundError(f"MLB report material artifact does not exist: {source_path}")
        return source_path
    if source_path.exists():
        ensure_dir(target_path.parent)
        shutil.copy2(source_path, target_path)
        return target_path
    if target_path.exists():
        return target_path
    raise FileNotFoundError(f"MLB report material artifact does not exist: {source_path}")


def _canonicalize_mlb_report_artifacts(
    cfg: AppConfig,
    *,
    result,
    summary_payload: dict[str, Any],
) -> tuple[Path, dict[str, str]] | None:
    league = str(summary_payload.get("league") or cfg.data.league).upper()
    if league != "MLB":
        return None

    output_dir = require_mlb_report_path(
        ensure_dir(Path(cfg.paths.artifacts_dir) / "reports" / _MLB_REPORT_LANE),
        cfg.paths.artifacts_dir,
        purpose="MLB latest candidate comparison artifacts",
    )
    artifacts = summary_payload.setdefault("artifacts", {})
    canonical_artifacts: dict[str, str] = {}
    for key, source in _artifact_sources(result, artifacts).items():
        if source is None or str(source) == "":
            continue
        canonical_artifacts[key] = str(_copy_to_mlb_report_lane(source=source, output_dir=output_dir))

    required_latest_keys = {
        "report_path",
        "summary_path",
        "candidate_scorecards_path",
        "candidate_scorecards_contract_path",
        "recommendation_path",
    }
    missing = sorted(required_latest_keys.difference(canonical_artifacts))
    if missing:
        raise RuntimeError(f"MLB latest service output is missing material artifact paths: {', '.join(missing)}")
    artifacts.update(canonical_artifacts)
    return output_dir, canonical_artifacts


def _persist_candidate_comparison_result(
    cfg: AppConfig,
    *,
    result,
    summary_payload: dict[str, Any],
    tracker_run_id: str,
) -> tuple[str, Path]:
    db = Database(cfg.paths.db_path)
    db.init_schema()

    decision = summary_payload.get("promotion_decision", {})
    artifacts = summary_payload.get("artifacts", {})
    metadata = summary_payload.get("metadata", {})

    league = str(summary_payload.get("league") or cfg.data.league).upper()
    run_id = f"{league.lower()}::candidate_compare::{result.report_slug}"
    summary_json = json.dumps(summary_payload, sort_keys=True)
    started_at = str(metadata.get("generated_at_utc") or utc_now_iso())
    completed_at = utc_now_iso()

    db.execute(
        """
        INSERT OR REPLACE INTO experiment_runs(
          run_id, league, profile_key, brief_id, brief_key, incumbent_model_name, candidate_model_name,
          report_slug, report_path, scorecard_path, fold_metrics_path, promotion_path, status,
          auto_promote, started_at_utc, completed_at_utc, summary_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            league,
            "default",
            None,
            "candidate_model_comparison",
            str(decision.get("baseline_model") or "intercept_only"),
            str(decision.get("recommended_model") or result.recommendation_model),
            str(result.report_slug),
            str(artifacts.get("report_path") or result.report_path),
            str(artifacts.get("candidate_scorecards_path") or result.summary_path),
            str(artifacts.get("cv_path") or result.validation_metrics_path),
            str(artifacts.get("recommendation_path") or result.summary_path),
            "candidate_screen_complete",
            0,
            started_at,
            completed_at,
            summary_json,
        ),
    )

    tracker = RunTracker(cfg.paths.artifacts_dir)
    service_output_path = Path(
        tracker.log_artifact(
            tracker_run_id,
            "candidate_comparison_service",
            {
                "experiment_run_id": run_id,
                "league": league,
                "report_slug": result.report_slug,
                "recommended_model": str(decision.get("recommended_model") or result.recommendation_model),
                "status": "candidate_screen_complete",
                "summary_path": str(artifacts.get("summary_path") or result.summary_path),
            },
        )
    )
    return run_id, service_output_path


def _write_mlb_service_output(
    cfg: AppConfig,
    *,
    result,
    summary_payload: dict[str, Any],
    tracker_output_path: Path,
    canonical_output_dir: Path,
    canonical_artifacts: dict[str, str],
) -> tuple[Path, Path, dict[str, str]] | None:
    league = str(summary_payload.get("league") or cfg.data.league).upper()
    if league != "MLB":
        return None

    promotion_decision = summary_payload.get("promotion_decision", {})
    candidate_scorecards = summary_payload.get("candidate_scorecards", [])
    artifacts = summary_payload.get("artifacts", {})
    metadata = summary_payload.get("metadata", {})

    output_path = canonical_output_dir / "candidate_model_comparison_latest.json"
    manifest_path = canonical_output_dir / "candidate_model_comparison_latest_manifest.json"
    material_artifacts = {
        key: canonical_artifacts[key]
        for key in _MLB_LATEST_MATERIAL_ARTIFACT_KEYS
        if key in canonical_artifacts
    }
    payload = {
        "league": league,
        "report_lane": _MLB_REPORT_LANE,
        "canonical_artifact_root": str(canonical_output_dir),
        "report_slug": str(result.report_slug),
        "generated_at_utc": utc_now_iso(),
        "status": str(promotion_decision.get("status") or "candidate_screen_complete"),
        "recommended_model": str(promotion_decision.get("recommended_model") or result.recommendation_model),
        "recommended_display_name": str(metadata.get("recommended_display_name") or result.recommendation_display_name),
        "rationale": str(promotion_decision.get("rationale") or ""),
        "report_path": material_artifacts["report_path"],
        "summary_path": material_artifacts["summary_path"],
        "recommendation_path": material_artifacts["recommendation_path"],
        "recommendation_surface_path": str(material_artifacts.get("recommendation_surface_path") or ""),
        "candidate_scorecards_path": material_artifacts["candidate_scorecards_contract_path"],
        "leaderboard_path": str(
            material_artifacts.get("leaderboard_json_path") or artifacts.get("leaderboard_path") or ""
        ),
        "tracker_service_output_path": str(tracker_output_path),
        "promotion_decision": promotion_decision,
        "recommendation_surface": metadata.get("recommendation_surface"),
        "execution_metadata": metadata.get("execution_metadata", {}),
        "top_scorecards": list(candidate_scorecards)[:3],
        "material_artifacts": material_artifacts,
        "manifest_path": str(manifest_path),
        "current_best_models_path": str(canonical_output_dir / _CURRENT_BEST_MODELS_FILE_NAME),
    }
    manifest_payload = {
        "league": league,
        "report_lane": _MLB_REPORT_LANE,
        "canonical_artifact_root": str(canonical_output_dir),
        "report_slug": str(result.report_slug),
        "generated_at_utc": utc_now_iso(),
        "material_artifacts": material_artifacts,
        "service_output_path": str(output_path),
        "tracker_service_output_path": str(tracker_output_path),
        "current_best_models_path": str(canonical_output_dir / _CURRENT_BEST_MODELS_FILE_NAME),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n")
    return output_path, manifest_path, material_artifacts


def compare_candidate_models(
    cfg: AppConfig,
    *,
    report_slug: str | None = None,
    bootstrap_samples: int = 1000,
    candidate_models: list[str] | None = None,
    feature_pool: str = "full_screened",
    feature_map_model: str = "glm_ridge",
    structured_glm_spec_path: str | None = None,
    structured_glm_slate: str | None = None,
    structured_glm_width_variant: str | None = None,
    execution_metadata: dict[str, Any] | None = None,
):
    tracker = RunTracker(cfg.paths.artifacts_dir)
    tracker_run_id = tracker.start_run(
        "candidate_compare",
        {
            "league": str(cfg.data.league).upper(),
            "report_slug": report_slug or "",
            "candidate_models": list(candidate_models or []),
            "feature_pool": feature_pool,
            "feature_map_model": feature_map_model,
        },
    )
    result = run_candidate_model_comparison(
        cfg,
        report_slug=report_slug,
        bootstrap_samples=bootstrap_samples,
        candidate_models=candidate_models,
        feature_pool=feature_pool,
        feature_map_model=feature_map_model,
        structured_glm_spec_path=structured_glm_spec_path,
        structured_glm_slate=structured_glm_slate,
        structured_glm_width_variant=structured_glm_width_variant,
        execution_metadata=execution_metadata,
    )
    summary_payload = _load_summary_payload(result.summary_path)
    mlb_canonical = _canonicalize_mlb_report_artifacts(
        cfg,
        result=result,
        summary_payload=summary_payload,
    )
    experiment_run_id, service_output_path = _persist_candidate_comparison_result(
        cfg,
        result=result,
        summary_payload=summary_payload,
        tracker_run_id=tracker_run_id,
    )
    tracker.log_metrics(
        tracker_run_id,
        {
            "candidate_count": int(len(summary_payload.get("candidate_scorecards", []))),
            "recommended_model": str(
                summary_payload.get("promotion_decision", {}).get("recommended_model") or result.recommendation_model
            ),
            "status": "candidate_screen_complete",
        },
    )
    tracker.end_run(tracker_run_id)
    summary_payload.setdefault("artifacts", {})
    summary_payload["artifacts"]["service_output_path"] = str(service_output_path)
    if mlb_canonical is not None:
        mlb_output_dir, canonical_artifacts = mlb_canonical
        mlb_outputs = _write_mlb_service_output(
            cfg,
            result=result,
            summary_payload=summary_payload,
            tracker_output_path=service_output_path,
            canonical_output_dir=mlb_output_dir,
            canonical_artifacts=canonical_artifacts,
        )
    else:
        mlb_outputs = None
    summary_payload.setdefault("metadata", {})
    summary_payload["metadata"]["experiment_run_id"] = experiment_run_id
    if mlb_outputs is not None:
        mlb_service_output_path, mlb_manifest_path, material_artifacts = mlb_outputs
        summary_payload["artifacts"]["mlb_service_output_path"] = str(mlb_service_output_path)
        summary_payload["artifacts"]["mlb_service_manifest_path"] = str(mlb_manifest_path)
        summary_payload["artifacts"]["mlb_latest_material_artifacts"] = material_artifacts
        current_best_models_output_path = write_current_best_models_from_comparison(
            cfg,
            summary_payload=summary_payload,
            canonical_output_dir=mlb_service_output_path.parent,
        )
        if current_best_models_output_path is not None:
            summary_payload["artifacts"]["current_best_models_path"] = str(current_best_models_output_path)
    else:
        mlb_service_output_path = None
    summary_text = json.dumps(summary_payload, indent=2, sort_keys=True) + "\n"
    result.summary_path.write_text(summary_text)
    canonical_summary_path = Path(summary_payload["artifacts"].get("summary_path") or result.summary_path)
    if canonical_summary_path.expanduser().resolve() != result.summary_path.expanduser().resolve():
        canonical_summary_path.write_text(summary_text)
    if hasattr(result, "service_output_path"):
        result.service_output_path = mlb_service_output_path or service_output_path
    logger.info(
        "Candidate model comparison complete | league=%s recommendation=%s report=%s",
        result.league,
        result.recommendation_model,
        result.report_path,
    )
    return result
