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


def _load_summary_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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

    output_dir = ensure_dir(Path(cfg.paths.artifacts_dir) / "reports" / _MLB_REPORT_LANE)
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
    if mlb_outputs is not None:
        mlb_service_output_path, mlb_manifest_path, material_artifacts = mlb_outputs
        summary_payload["artifacts"]["mlb_service_output_path"] = str(mlb_service_output_path)
        summary_payload["artifacts"]["mlb_service_manifest_path"] = str(mlb_manifest_path)
        summary_payload["artifacts"]["mlb_latest_material_artifacts"] = material_artifacts
    else:
        mlb_service_output_path = None
    summary_payload.setdefault("metadata", {})
    summary_payload["metadata"]["experiment_run_id"] = experiment_run_id
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
