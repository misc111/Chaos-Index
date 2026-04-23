from __future__ import annotations

import argparse
import copy
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from typing import Any

import pandas as pd

from src.common.config import AppConfig, load_config
from src.common.time import utc_now_iso
from src.common.utils import ensure_dir, stable_hash, to_json
from src.research.artifact_guardrails import require_mlb_tournament_path
from src.research.model_comparison import run_candidate_model_comparison
from src.services.model_compare import write_current_best_models_from_tournament
from src.training.lambda_search import c_to_lambda, default_l1_ratio_grid, default_lambda_grid

DEFAULT_FEATURE_POOLS = ("full_screened",)
DEFAULT_ROUND_TEMPLATES = (
    ("linear_core", ("glm_vanilla", "glm_ridge", "glm_lasso", "glm_elastic_net")),
    ("grouped_effects", ("glm_vanilla", "glm_ridge", "glm_lasso", "glm_elastic_net", "glmm_logit", "dglm_margin")),
    (
        "nonlinear_challengers",
        ("glm_vanilla", "glm_ridge", "glm_lasso", "glm_elastic_net", "glmm_logit", "dglm_margin", "gam_spline", "mars_hinge"),
    ),
)
DEFAULT_VALIDATION_VARIANT = "outer_40_30_30"
DEFAULT_HOLDOUT_VARIANT = "final_holdout_30pct"
DEFAULT_BOOTSTRAP_SAMPLES = 200
SERIOUS_RECOMMENDATION_STATUSES = {
    "promotion-review-ready",
    "shortlist",
    "theory-blocked",
    "validation-blocked",
    "reject",
    "not-enough-evidence",
}
VALIDATION_CONTRACT_REQUIREMENTS = (
    "fit_statistics",
    "residual_diagnostics",
    "multicollinearity_review",
    "calibration_and_lift",
    "stability_review",
    "artifact_manifest",
    "theory_traceability",
    "promotion_report",
)
HOLDOUT_SCHEMES: dict[str, dict[str, Any]] = {
    "outer_40_30_30": {
        "validation_variant": "outer_40_30_30",
        "holdout_variant": "final_holdout_30pct",
        "train_fraction": 0.40,
        "validation_fraction": 0.30,
        "description": "CAS-style train/validation/test split with the final 30% held out in time order.",
    },
    "outer_50_25_25": {
        "validation_variant": "outer_50_25_25",
        "holdout_variant": "final_holdout_25pct",
        "train_fraction": 0.50,
        "validation_fraction": 0.25,
        "description": "Wider training window with an out-of-time 25% final holdout.",
    },
    "outer_60_20_20": {
        "validation_variant": "outer_60_20_20",
        "holdout_variant": "final_holdout_20pct",
        "train_fraction": 0.60,
        "validation_fraction": 0.20,
        "description": "Deepest fit window used in this bounded slice with an out-of-time 20% final holdout.",
    },
}
THEORY_CLASSIFICATION_BY_MODEL: dict[str, dict[str, str]] = {
    "glm_vanilla": {
        "theory_classification": "core-supported",
        "theory_lane": "theory-core",
        "governing_sources": "GLM monograph 2.1, 2.8, 4.3, 6.1-6.3, 7.1-7.3; Holmes/Casotto 1.1-1.4",
        "theory_caveat": "Binary moneyline target uses binomial/logit logic; p-values belong only to unpenalized GLM review, not penalized/credibility rows.",
    },
    "glm_ridge": {
        "theory_classification": "core-supported",
        "theory_lane": "theory-core",
        "governing_sources": "GLM monograph 10.5; Holmes/Casotto 3.1.1-3.1.2, 3.3",
        "theory_caveat": "Supported as penalized GLM; less sparse than lasso, so interpretability/credibility review is more conservative.",
    },
    "glm_lasso": {
        "theory_classification": "core-supported",
        "theory_lane": "theory-core",
        "governing_sources": "GLM monograph 10.5; Holmes/Casotto 3.1.1, 3.1.3, 3.2-3.5, 6.1",
        "theory_caveat": "Review focuses on lambda, sparsity, stability, and credibility-style materiality; no p-value review.",
    },
    "glm_elastic_net": {
        "theory_classification": "core-supported",
        "theory_lane": "theory-core",
        "governing_sources": "GLM monograph 10.5; Holmes/Casotto 3.1, 3.3, C.2-C.3",
        "theory_caveat": "Core-supported penalized GLM, but lasso remains the preferred actuarial sparsity framing in Holmes/Casotto.",
    },
    "glm_lasso_market_credibility": {
        "theory_classification": "core-supported",
        "theory_lane": "theory-core",
        "governing_sources": "Holmes/Casotto 5.1-5.7, 6.1-6.4, Appendix B",
        "theory_caveat": "Complement must be explicit and reviewed; no p-values or significance-testing language.",
    },
    "glm_lasso_prior_credibility": {
        "theory_classification": "core-supported",
        "theory_lane": "theory-core",
        "governing_sources": "Holmes/Casotto 5.1-5.7, 6.1-6.4, Appendix B",
        "theory_caveat": "Prior complement must be documented as real prior evidence or labeled proxy; no p-values.",
    },
    "glmm_logit": {
        "theory_classification": "theory-compatible extension",
        "theory_lane": "extension",
        "governing_sources": "GLM monograph 10.1",
        "theory_caveat": "Supported as a GLM variation for random effects/shrinkage, not the primary theory-core champion lane.",
    },
    "dglm_margin": {
        "theory_classification": "theory-compatible extension",
        "theory_lane": "extension",
        "governing_sources": "GLM monograph 10.2",
        "theory_caveat": "Extension support is for dispersion modeling; current margin-to-win probability bridge needs cautious interpretation.",
    },
    "gam_spline": {
        "theory_classification": "theory-compatible extension",
        "theory_lane": "extension",
        "governing_sources": "GLM monograph 10.3",
        "theory_caveat": "Current implementation is spline-basis logistic regression, an approximation of the GAM lane rather than a full production GAM.",
    },
    "mars_hinge": {
        "theory_classification": "experimental",
        "theory_lane": "experimental",
        "governing_sources": "GLM monograph 5.4.4, 10.4; Holmes/Casotto 3.4 by analogy only",
        "theory_caveat": "Canonical MARS is theory-compatible, but this implementation is a hinge-basis proxy rather than a full forward/pruned MARS procedure.",
    },
    "intercept_only": {
        "theory_classification": "experimental",
        "theory_lane": "benchmark",
        "governing_sources": "GLM monograph 6.1",
        "theory_caveat": "Null/intercept reference for comparison only; not a champion candidate.",
    },
}


@dataclass(slots=True)
class TournamentResult:
    run_id: str
    artifact_root: Path
    manifest_path: Path
    ledger_path: Path
    summary_path: Path
    scorecards_path: Path
    leaderboard_csv_path: Path
    leaderboard_json_path: Path
    recommendation_json_path: Path
    recommendation_md_path: Path
    theory_compliance_summary_path: Path


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slugify(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip().lower())
    token = token.strip("_")
    return token or "item"


def _split_csv(value: str | None, *, default: tuple[str, ...]) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return list(default)
    return [token.strip() for token in raw.split(",") if token.strip()]


def _candidate_name_filter(raw_value: str | None) -> set[str] | None:
    raw = str(raw_value or "").strip()
    if not raw or raw.lower() in {"all", "*"}:
        return None
    return {token.strip() for token in raw.split(",") if token.strip()}


def _theory_metadata(model_name: str) -> dict[str, str]:
    return dict(
        THEORY_CLASSIFICATION_BY_MODEL.get(
            str(model_name),
            {
                "theory_classification": "experimental",
                "theory_lane": "experimental",
                "governing_sources": "",
                "theory_caveat": "Not directly justified by the governing monographs in this tournament contract.",
            },
        )
    )


def _holdout_scheme_names(raw_value: str | None, *, validation_variant: str, holdout_variant: str) -> list[str]:
    raw = str(raw_value or "").strip()
    if not raw:
        return [validation_variant] if validation_variant in HOLDOUT_SCHEMES else [DEFAULT_VALIDATION_VARIANT]
    if raw.lower() in {"all", "*"}:
        return list(HOLDOUT_SCHEMES)
    names = [token.strip() for token in raw.split(",") if token.strip()]
    bad = [name for name in names if name not in HOLDOUT_SCHEMES]
    if bad:
        raise ValueError(f"Unknown holdout schemes: {bad}. Valid={sorted(HOLDOUT_SCHEMES)}")
    return names


def _round_templates(candidate_filter: set[str] | None) -> list[tuple[str, tuple[str, ...]]]:
    templates: list[tuple[str, tuple[str, ...]]] = []
    seen: set[tuple[str, ...]] = set()
    for round_name, model_names in DEFAULT_ROUND_TEMPLATES:
        selected = tuple(name for name in model_names if candidate_filter is None or name in candidate_filter)
        if selected and selected not in seen:
            seen.add(selected)
            templates.append((round_name, selected))
    return templates


def _stable_run_id(
    *,
    league: str,
    feature_pools: list[str],
    candidate_models: set[str] | None,
    bootstrap_samples: int,
    holdout_schemes: list[str],
) -> str:
    payload = {
        "league": league.upper(),
        "feature_pools": feature_pools,
        "candidate_models": sorted(candidate_models) if candidate_models else [],
        "bootstrap_samples": int(bootstrap_samples),
        "holdout_schemes": list(holdout_schemes),
    }
    digest = stable_hash(payload)[:8]
    return f"{_utc_stamp()}_{digest}"


def _artifact_root(cfg: AppConfig, run_id: str) -> Path:
    root = ensure_dir(Path(cfg.paths.artifacts_dir) / "reports" / "mlb" / "tournament" / run_id)
    return require_mlb_tournament_path(root, cfg.paths.artifacts_dir, purpose="MLB model tournament artifacts")


def _round_dir(root: Path, round_id: str) -> Path:
    return ensure_dir(root / "rounds" / round_id)


def _append_ledger_entry(path: Path, entry: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _copy_artifact(source: Any, target_dir: Path) -> Path:
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing comparison artifact: {source_path}")
    ensure_dir(target_dir)
    target_path = target_dir / source_path.name
    if source_path.resolve() != target_path.resolve():
        copy2(source_path, target_path)
    return target_path


def _comparison_artifacts(result: Any, summary_payload: dict[str, Any]) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    direct = {
        "report_path": getattr(result, "report_path", None),
        "summary_path": getattr(result, "summary_path", None),
        "candidate_scorecards_path": getattr(result, "candidate_scorecards_path", None),
        "candidate_scorecards_contract_path": getattr(result, "candidate_scorecards_contract_path", None),
        "leaderboard_path": getattr(result, "leaderboard_path", None),
        "leaderboard_json_path": getattr(result, "leaderboard_json_path", None),
        "recommendation_path": getattr(result, "recommendation_path", None),
        "recommendation_surface_path": getattr(result, "recommendation_surface_path", None),
        "artifact_manifest_path": getattr(result, "artifact_manifest_path", None),
        "service_output_path": getattr(result, "service_output_path", None),
    }
    for key, value in direct.items():
        if value:
            artifacts[key] = Path(value)

    summary_artifacts = summary_payload.get("artifacts", {})
    if isinstance(summary_artifacts, dict):
        for key, value in summary_artifacts.items():
            if value and key not in artifacts:
                artifacts[key] = Path(str(value))
    return artifacts


def _parse_params_text(params_text: Any) -> dict[str, Any]:
    if not params_text:
        return {}
    parsed: dict[str, Any] = {}
    for token in str(params_text).split(";"):
        item = token.strip()
        if not item or "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            continue
        lowered = raw_value.lower()
        if lowered in {"none", "null", "nan"}:
            parsed[key] = None
            continue
        try:
            parsed[key] = int(raw_value) if re.fullmatch(r"[+-]?\d+", raw_value) else float(raw_value)
            continue
        except ValueError:
            pass
        parsed[key] = raw_value
    return parsed


def _planned_candidate_metadata(model_name: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"model_name": model_name}
    if model_name in {"glm_ridge", "glm_lasso", "glm_elastic_net"}:
        payload["lambda_grid"] = default_lambda_grid(model_name)
    if model_name == "glm_elastic_net":
        payload["l1_ratio_grid"] = default_l1_ratio_grid(model_name)
    return payload


def _build_round_plan(
    *,
    feature_pool: str,
    holdout_scheme_name: str,
    holdout_scheme: dict[str, Any],
    candidate_filter: set[str] | None,
) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for round_index, (round_name, model_names) in enumerate(_round_templates(candidate_filter), start=1):
        validation_variant = str(holdout_scheme["validation_variant"])
        holdout_variant = str(holdout_scheme["holdout_variant"])
        round_id = f"r{round_index:02d}_{_slugify(feature_pool)}_{_slugify(validation_variant)}_{round_name}"
        plans.append(
            {
                "round_id": round_id,
                "round_index": round_index,
                "round_name": round_name,
                "feature_pool": feature_pool,
                "validation_variant": validation_variant,
                "holdout_variant": holdout_variant,
                "holdout_scheme": holdout_scheme_name,
                "train_fraction": float(holdout_scheme["train_fraction"]),
                "validation_fraction": float(holdout_scheme["validation_fraction"]),
                "test_fraction": float(1.0 - float(holdout_scheme["train_fraction"]) - float(holdout_scheme["validation_fraction"])),
                "holdout_description": str(holdout_scheme.get("description") or ""),
                "candidate_models": list(model_names),
                "planned_candidates": [_planned_candidate_metadata(model_name) for model_name in model_names],
                "status": "planned",
                "started_at_utc": None,
                "completed_at_utc": None,
                "artifacts": {},
            }
        )
    return plans


def _read_test_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _candidate_record_from_summary(
    *,
    round_plan: dict[str, Any],
    summary_payload: dict[str, Any],
    candidate_record: dict[str, Any],
    test_metrics: pd.DataFrame,
    comparison_artifacts: dict[str, Path],
    round_artifacts: dict[str, str],
) -> dict[str, Any]:
    model_name = str(candidate_record.get("model_name") or "")
    score_row = test_metrics[test_metrics["model_name"] == model_name].head(1)
    params_text = str(score_row.iloc[0]["params"]) if not score_row.empty and "params" in score_row.columns else ""
    params = _parse_params_text(params_text)
    lambda_value = params.get("lambda")
    if lambda_value is None and "c" in params and params["c"] not in {None, 0}:
        try:
            lambda_value = float(c_to_lambda(float(params["c"])))
        except Exception:
            lambda_value = None
    l1_ratio = params.get("l1_ratio")
    validation_metrics = dict(candidate_record.get("validation_metrics") or {})
    stability_metrics = dict(candidate_record.get("stability_metrics") or {})
    calibration_summary = dict(candidate_record.get("calibration_summary") or {})
    complement_summary = dict(candidate_record.get("complement_summary") or {})
    feature_pool = str(round_plan.get("feature_pool") or "")
    validation_variant = str(round_plan.get("validation_variant") or "")
    holdout_variant = str(round_plan.get("holdout_variant") or "")
    candidate_id = f"{_slugify(feature_pool)}::{_slugify(validation_variant)}::{model_name}"
    theory = _theory_metadata(model_name)
    artifact_paths = {
        "artifact_root": str(summary_payload.get("metadata", {}).get("artifact_root") or ""),
        "round_root": str(round_artifacts["round_root"]),
        "report_path": round_artifacts["report_path"],
        "summary_path": round_artifacts["summary_path"],
        "candidate_scorecards_path": round_artifacts["candidate_scorecards_path"],
        "candidate_scorecards_contract_path": round_artifacts["candidate_scorecards_contract_path"],
        "leaderboard_path": round_artifacts["leaderboard_path"],
        "leaderboard_json_path": round_artifacts["leaderboard_json_path"],
        "recommendation_path": round_artifacts["recommendation_path"],
        "recommendation_surface_path": round_artifacts["recommendation_surface_path"],
        "artifact_manifest_path": round_artifacts["artifact_manifest_path"],
        "service_output_path": round_artifacts.get("service_output_path", ""),
        "mlb_service_output_path": round_artifacts.get("mlb_service_output_path", ""),
        "mlb_service_manifest_path": round_artifacts.get("mlb_service_manifest_path", ""),
    }
    return {
        "candidate_id": candidate_id,
        "family": str(candidate_record.get("lane") or complement_summary.get("family") or "unknown"),
        "model_name": model_name,
        "display_name": str(complement_summary.get("display_name") or model_name),
        **theory,
        "feature_pool": feature_pool,
        "validation_variant": validation_variant,
        "holdout_variant": holdout_variant,
        "holdout_scheme": str(round_plan.get("holdout_scheme") or ""),
        "train_fraction": float(round_plan.get("train_fraction") or 0.0),
        "validation_fraction": float(round_plan.get("validation_fraction") or 0.0),
        "test_fraction": float(round_plan.get("test_fraction") or 0.0),
        "round_id": round_plan["round_id"],
        "round_index": int(round_plan["round_index"]),
        "round_name": str(round_plan["round_name"]),
        "round_history": [round_plan["round_id"]],
        "lambda": None if lambda_value is None else float(lambda_value),
        "l1_ratio": None if l1_ratio is None else float(l1_ratio),
        "complement": {
            "display_name": str(complement_summary.get("display_name") or model_name),
            "family": str(complement_summary.get("family") or candidate_record.get("lane") or "unknown"),
            "governance_note": str(complement_summary.get("governance_note") or ""),
            "recommendation_tier": str(complement_summary.get("recommendation_tier") or ""),
            "recommended_for_next_stage": bool(complement_summary.get("recommended_for_next_stage")),
        },
        "log_loss": float(validation_metrics.get("final_holdout_log_loss"))
        if validation_metrics.get("final_holdout_log_loss") is not None
        else None,
        "brier": float(validation_metrics.get("final_holdout_brier"))
        if validation_metrics.get("final_holdout_brier") is not None
        else None,
        "calibration": calibration_summary,
        "stability": stability_metrics,
        "artifact_paths": artifact_paths,
        "params": params,
        "validation_metrics": validation_metrics,
    }


def _baseline_record_from_test_metrics(
    *,
    round_plan: dict[str, Any],
    test_metrics: pd.DataFrame,
    comparison_artifacts: dict[str, Path],
    round_artifacts: dict[str, str],
) -> dict[str, Any] | None:
    if test_metrics.empty or "model_name" not in test_metrics.columns:
        return None
    intercept_row = test_metrics[test_metrics["model_name"] == "intercept_only"].head(1)
    if intercept_row.empty:
        return None
    row = intercept_row.iloc[0]
    feature_pool = str(round_plan.get("feature_pool") or "")
    validation_variant = str(round_plan.get("validation_variant") or "")
    holdout_variant = str(round_plan.get("holdout_variant") or "")
    candidate_id = f"{_slugify(feature_pool)}::{_slugify(validation_variant)}::intercept_only"
    theory = _theory_metadata("intercept_only")
    return {
        "candidate_id": candidate_id,
        "family": "benchmark",
        "model_name": "intercept_only",
        "display_name": "Intercept Only",
        **theory,
        "feature_pool": feature_pool,
        "validation_variant": validation_variant,
        "holdout_variant": holdout_variant,
        "holdout_scheme": str(round_plan.get("holdout_scheme") or ""),
        "train_fraction": float(round_plan.get("train_fraction") or 0.0),
        "validation_fraction": float(round_plan.get("validation_fraction") or 0.0),
        "test_fraction": float(round_plan.get("test_fraction") or 0.0),
        "round_id": round_plan["round_id"],
        "round_index": int(round_plan["round_index"]),
        "round_name": str(round_plan["round_name"]),
        "round_history": [round_plan["round_id"]],
        "lambda": None,
        "l1_ratio": None,
        "complement": {
            "display_name": "Intercept Only",
            "family": "benchmark",
            "governance_note": "Benchmark-only reference row.",
            "recommendation_tier": "benchmark",
            "recommended_for_next_stage": False,
        },
        "log_loss": None if pd.isna(row.get("log_loss")) else float(row.get("log_loss")),
        "brier": None if pd.isna(row.get("brier")) else float(row.get("brier")),
        "calibration": {
            "final_holdout_ece": None if pd.isna(row.get("ece")) else float(row.get("ece")),
            "calibration_alpha": None if pd.isna(row.get("calibration_alpha")) else float(row.get("calibration_alpha")),
            "calibration_beta": None if pd.isna(row.get("calibration_beta")) else float(row.get("calibration_beta")),
        },
        "stability": {},
        "artifact_paths": {
            "artifact_root": "",
            "round_root": str(round_artifacts["round_root"]),
            "report_path": round_artifacts["report_path"],
            "summary_path": round_artifacts["summary_path"],
            "candidate_scorecards_path": round_artifacts["candidate_scorecards_path"],
            "candidate_scorecards_contract_path": round_artifacts["candidate_scorecards_contract_path"],
            "leaderboard_path": round_artifacts["leaderboard_path"],
            "leaderboard_json_path": round_artifacts["leaderboard_json_path"],
            "recommendation_path": round_artifacts["recommendation_path"],
            "recommendation_surface_path": round_artifacts["recommendation_surface_path"],
            "artifact_manifest_path": round_artifacts["artifact_manifest_path"],
            "service_output_path": round_artifacts.get("service_output_path", ""),
            "mlb_service_output_path": round_artifacts.get("mlb_service_output_path", ""),
            "mlb_service_manifest_path": round_artifacts.get("mlb_service_manifest_path", ""),
        },
        "params": _parse_params_text(row.get("params")),
        "validation_metrics": {
            "final_holdout_log_loss": None if pd.isna(row.get("log_loss")) else float(row.get("log_loss")),
            "final_holdout_brier": None if pd.isna(row.get("brier")) else float(row.get("brier")),
        },
    }


def _round_status_from_payload(summary_payload: dict[str, Any]) -> str:
    status = str(summary_payload.get("promotion_decision", {}).get("status") or "").strip()
    if status in {"research_recommended", "research_hold", "fixture_slice_screen_complete"}:
        return "completed"
    return "completed" if status else "completed"


def _flatten_candidate_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    flat = pd.json_normalize(records, sep="__")
    for column in ("log_loss", "brier", "lambda", "l1_ratio", "round_index"):
        if column in flat.columns:
            flat[column] = pd.to_numeric(flat[column], errors="coerce")
    return flat


def _sort_leaderboard(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    work = frame.copy()
    work["calibration_flag"] = 0
    if "calibration__final_holdout_ece" in work.columns:
        work["calibration_flag"] = (
            pd.to_numeric(work["calibration__final_holdout_ece"], errors="coerce").fillna(0.0).gt(0.04).astype(int)
        )
    if "calibration__calibration_alpha" in work.columns:
        alpha_gap = (pd.to_numeric(work["calibration__calibration_alpha"], errors="coerce").fillna(1.0) - 1.0).abs()
        beta_gap = pd.Series(0.0, index=work.index)
        if "calibration__calibration_beta" in work.columns:
            beta_gap = (pd.to_numeric(work["calibration__calibration_beta"], errors="coerce").fillna(1.0) - 1.0).abs()
        work["calibration_flag"] = (
            (alpha_gap > 0.15) | (beta_gap > 0.15) | (work["calibration_flag"] == 1)
        ).astype(int)

    work["stability_flag"] = 0
    if "stability__validation_to_test_log_loss_delta" in work.columns:
        work["stability_flag"] = (
            pd.to_numeric(work["stability__validation_to_test_log_loss_delta"], errors="coerce").abs().fillna(0.0).gt(0.05).astype(int)
        )
    if "stability__rank_shift_validation_to_final" in work.columns:
        work["stability_flag"] = (
            pd.to_numeric(work["stability__rank_shift_validation_to_final"], errors="coerce").abs().fillna(0.0).gt(2).astype(int)
        ) | work["stability_flag"]
        work["stability_flag"] = work["stability_flag"].astype(int)

    sort_columns = ["log_loss", "brier", "calibration_flag", "stability_flag"]
    ascending = [True, True, True, True]
    sort_columns.append("candidate_id")
    ascending.append(True)
    return work.sort_values(sort_columns, ascending=ascending, na_position="last").reset_index(drop=True)


def _validation_contract_gate(row: pd.Series) -> dict[str, Any]:
    raw_complete = row.get("validation_contract_complete")
    complete = bool(raw_complete) if raw_complete is not None and not pd.isna(raw_complete) else False
    if complete:
        return {
            "complete": True,
            "missing_items": [],
            "reason": "Separate validation-contract review recorded as complete.",
        }
    return {
        "complete": False,
        "missing_items": list(VALIDATION_CONTRACT_REQUIREMENTS),
        "reason": "Leaderboard ranking is shortlist evidence only; a separate full validation-contract review is still required.",
    }


def _recommendation_rows(leaderboard: pd.DataFrame) -> list[dict[str, Any]]:
    if leaderboard.empty:
        return []
    baseline_rows = leaderboard[leaderboard["model_name"] == "intercept_only"]
    baseline = baseline_rows.iloc[0] if not baseline_rows.empty else None
    rows: list[dict[str, Any]] = []
    serious = leaderboard[leaderboard["model_name"] != "intercept_only"].copy()
    best_serious = serious.iloc[0] if not serious.empty else None
    for _, row in serious.iterrows():
        log_loss = row.get("log_loss")
        brier = row.get("brier")
        calibration_flag = int(row.get("calibration_flag") or 0)
        stability_flag = int(row.get("stability_flag") or 0)
        theory_classification = str(row.get("theory_classification") or "")
        theory_core_gate = theory_classification == "core-supported"
        validation_contract = _validation_contract_gate(row)
        status = "not-enough-evidence"
        rationale = "Hold out for more evidence."
        if pd.notna(log_loss) and pd.notna(brier):
            if baseline is not None and pd.notna(baseline.get("log_loss")) and pd.notna(baseline.get("brier")):
                beats_baseline = float(log_loss) < float(baseline.get("log_loss")) and float(brier) <= float(
                    baseline.get("brier")
                )
                is_best_serious = best_serious is not None and str(row.get("candidate_id")) == str(
                    best_serious.get("candidate_id")
                )
                if beats_baseline and is_best_serious:
                    if not theory_core_gate:
                        status = "theory-blocked"
                        rationale = "Top holdout score came from a non-core theory lane, so it cannot auto-advance as the default actuarial winner."
                    elif not validation_contract["complete"]:
                        status = "shortlist" if calibration_flag == 0 and stability_flag == 0 else "validation-blocked"
                        rationale = (
                            "Top theory-core candidate on holdout scoring; shortlist it for a separate promotion decision after full validation-contract review."
                            if status == "shortlist"
                            else "Top theory-core candidate on holdout scoring, but calibration/stability flags and the missing validation-contract review keep it out of promotion review."
                        )
                    elif calibration_flag == 0 and stability_flag == 0:
                        status = "promotion-review-ready"
                        rationale = "Top theory-core candidate on holdout scoring and ready for a separate written promotion decision."
                    else:
                        status = "not-enough-evidence"
                        rationale = "Score is competitive, but calibration or stability flags keep it in the watchlist."
                elif float(log_loss) > float(baseline.get("log_loss")) and float(brier) > float(baseline.get("brier")):
                    status = "reject"
                    rationale = "Underperformed the intercept benchmark on both log loss and Brier."
                else:
                    status = "not-enough-evidence"
                    rationale = "Mixed holdout signal; keep the candidate under review."
            else:
                status = "not-enough-evidence"
                rationale = "Missing baseline metrics."
        rows.append(
            {
                "candidate_id": row.get("candidate_id"),
                "model_name": row.get("model_name"),
                "display_name": row.get("display_name"),
                "theory_classification": row.get("theory_classification"),
                "theory_lane": row.get("theory_lane"),
                "governing_sources": row.get("governing_sources"),
                "theory_caveat": row.get("theory_caveat"),
                "status": status,
                "rationale": rationale,
                "theory_core_gate": theory_core_gate,
                "validation_contract_complete": bool(validation_contract["complete"]),
                "validation_contract_missing_items": list(validation_contract["missing_items"]),
                "promotion_review_required": True,
                "log_loss": None if pd.isna(log_loss) else float(log_loss),
                "brier": None if pd.isna(brier) else float(brier),
                "calibration_flag": calibration_flag,
                "stability_flag": stability_flag,
            }
        )
    return rows


def _render_recommendation_markdown(
    *,
    run_id: str,
    root: Path,
    leaderboard: pd.DataFrame,
    recommendation_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# MLB Tournament Recommendation",
        "",
        f"- Run ID: `{run_id}`",
        f"- Artifact root: `{root}`",
        f"- Candidate count: {len(leaderboard)}",
        "",
        "| Candidate | Model | Theory Label | Status | Log Loss | Brier |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in recommendation_rows:
        lines.append(
            "| {candidate_id} | {model_name} | {theory_classification} | {status} | {log_loss} | {brier} |".format(
                candidate_id=row.get("candidate_id") or "",
                model_name=row.get("model_name") or "",
                theory_classification=row.get("theory_classification") or "",
                status=row.get("status") or "",
                log_loss="" if row.get("log_loss") is None else f"{float(row['log_loss']):.6f}",
                brier="" if row.get("brier") is None else f"{float(row['brier']):.6f}",
            )
        )
    if not recommendation_rows:
        lines.append("| n/a | n/a | experimental | not-enough-evidence | | |")
    lines.extend(
        [
            "",
            "Theory labels are conservative: `core-supported` is limited to direct support from the two governing monographs; extension and experimental rows are not promoted as main-lane actuarial truth.",
            "Leaderboard output is a shortlist only. Champion promotion requires a separate written decision after the full validation contract is checked.",
            "",
            "The runner keeps the model-selection lane additive and leaves champion promotion to the written evidence review.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_theory_compliance_summary(
    *,
    run_id: str,
    leaderboard: pd.DataFrame,
    holdout_schemes: list[str],
) -> str:
    lines = [
        "# MLB Tournament Theory Compliance Summary",
        "",
        f"- Run ID: `{run_id}`",
        "- Governing sources:",
        "  - `statistical_theory/09_GLM_Generalized_Linear_Models_for_Insurance_Rating.pdf`",
        "  - `statistical_theory/10_Holmes_Casotto_Penalized_Regression_and_Lasso_Credibility_2025_Revision.pdf`",
        f"- Holdout schemes run: {', '.join(holdout_schemes)}",
        "",
        "## Theory Labels",
        "",
        "| Model | Classification | Lane | Governing Source | Caveat |",
        "| --- | --- | --- | --- | --- |",
    ]
    seen: set[str] = set()
    for model_name in sorted(set(leaderboard.get("model_name", pd.Series(dtype=str)).astype(str).tolist())):
        if model_name in seen:
            continue
        seen.add(model_name)
        theory = _theory_metadata(model_name)
        lines.append(
            "| {model} | {classification} | {lane} | {sources} | {caveat} |".format(
                model=model_name,
                classification=theory["theory_classification"],
                lane=theory["theory_lane"],
                sources=theory["governing_sources"],
                caveat=theory["theory_caveat"],
            )
        )
    lines.extend(
        [
            "",
            "## Compliance Rules Applied",
            "",
            "- Penalized GLM rows are evaluated through cross-validated lambda choices, proper scoring rules, calibration, and stability evidence, not p-values.",
            "- Lasso-credibility rows require explicit complement/offset documentation and no significance-testing language.",
            "- GAM, GLMM, and DGLM rows are kept as `theory-compatible extension`, not default theory-core winners.",
            "- `mars_hinge` is labeled `experimental`: canonical MARS is traceable to the GLM monograph, but this repo implementation is a hinge-basis proxy.",
            "- Null/intercept rows are benchmark references and are not champion candidates.",
            "- Bounded-slice artifacts are not production-grade promotion evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_mlb_model_tournament(
    cfg: AppConfig,
    *,
    run_id: str | None = None,
    candidate_models: str | None = None,
    feature_pools: str | None = None,
    validation_variant: str = DEFAULT_VALIDATION_VARIANT,
    holdout_variant: str = DEFAULT_HOLDOUT_VARIANT,
    holdout_schemes: str | None = None,
    feature_map_model: str = "glm_ridge",
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> TournamentResult:
    league = str(cfg.data.league).upper()
    if league != "MLB":
        raise ValueError(f"MLB tournament runner only supports league=MLB, got {league!r}")

    candidate_filter = _candidate_name_filter(candidate_models)
    resolved_feature_pools = _split_csv(feature_pools, default=DEFAULT_FEATURE_POOLS)
    resolved_holdout_schemes = _holdout_scheme_names(
        holdout_schemes,
        validation_variant=validation_variant,
        holdout_variant=holdout_variant,
    )
    resolved_run_id = run_id or _stable_run_id(
        league=league,
        feature_pools=resolved_feature_pools,
        candidate_models=candidate_filter,
        bootstrap_samples=bootstrap_samples,
        holdout_schemes=resolved_holdout_schemes,
    )
    root = _artifact_root(cfg, resolved_run_id)
    ledger_path = root / "progress_ledger.jsonl"
    manifest_path = root / "sweep_manifest.json"
    summary_path = root / "tournament_summary.json"
    scorecards_path = root / "candidate_scorecards.json"
    leaderboard_csv_path = root / "leaderboard.csv"
    leaderboard_json_path = root / "leaderboard.json"
    recommendation_json_path = root / "recommendation.json"
    recommendation_md_path = root / "recommendation.md"
    theory_compliance_summary_path = root / "theory_compliance_summary.md"

    rounds: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "league": league,
        "run_id": resolved_run_id,
        "generated_at_utc": utc_now_iso(),
        "artifact_root": str(root),
        "config_path": str(getattr(cfg.paths, "config_path", "")) if getattr(cfg.paths, "config_path", None) else "",
        "bootstrap_samples": int(bootstrap_samples),
        "candidate_model_filter": sorted(candidate_filter) if candidate_filter else None,
        "feature_pools": resolved_feature_pools,
        "holdout_schemes": resolved_holdout_schemes,
        "holdout_scheme_specs": {name: dict(HOLDOUT_SCHEMES[name]) for name in resolved_holdout_schemes},
        "validation_variant": validation_variant,
        "holdout_variant": holdout_variant,
        "theory_contract": {
            "classification_labels": ["core-supported", "theory-compatible extension", "experimental"],
            "traceability_matrix": "docs/mlb/theory_traceability_matrix.md",
            "governing_sources": [
                "statistical_theory/09_GLM_Generalized_Linear_Models_for_Insurance_Rating.pdf",
                "statistical_theory/10_Holmes_Casotto_Penalized_Regression_and_Lasso_Credibility_2025_Revision.pdf",
            ],
        },
        "rounds": rounds,
        "status": "running",
    }
    _write_json(manifest_path, manifest)

    latest_records: dict[tuple[str, str, str], dict[str, Any]] = {}
    round_history: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    completed_rounds: list[dict[str, Any]] = []

    try:
        for feature_pool in resolved_feature_pools:
            for holdout_scheme_name in resolved_holdout_schemes:
                holdout_scheme = dict(HOLDOUT_SCHEMES[holdout_scheme_name])
                round_plans = _build_round_plan(
                    feature_pool=feature_pool,
                    holdout_scheme_name=holdout_scheme_name,
                    holdout_scheme=holdout_scheme,
                    candidate_filter=candidate_filter,
                )
                for round_plan in round_plans:
                    round_plan = copy.deepcopy(round_plan)
                    round_dir = _round_dir(root, round_plan["round_id"])
                    task_id = f"{resolved_run_id}:{round_plan['round_id']}"
                    command = (
                        "python3 scripts/run_mlb_model_tournament.py "
                        f"--config {getattr(cfg.paths, 'config_path', '') or 'configs/mlb.yaml'} "
                        f"--run-id {resolved_run_id} --holdout-schemes {','.join(resolved_holdout_schemes)}"
                    )
                    round_plan["status"] = "running"
                    round_plan["started_at_utc"] = utc_now_iso()
                    round_plan["artifacts"]["round_root"] = str(round_dir)
                    completed_rounds.append(round_plan)
                    _append_ledger_entry(
                        ledger_path,
                        {
                            "task_id": task_id,
                            "status": "started",
                            "command": command,
                            "callable": "src.research.mlb_model_tournament:run_mlb_model_tournament",
                            "started_at_utc": round_plan["started_at_utc"],
                            "completed_at_utc": None,
                            "artifacts": {"round_root": str(round_dir)},
                            "error": None,
                            "feature_pool": feature_pool,
                            "validation_variant": round_plan["validation_variant"],
                            "holdout_variant": round_plan["holdout_variant"],
                            "holdout_scheme": round_plan["holdout_scheme"],
                        },
                    )
                    compare_slug = f"{resolved_run_id}_{round_plan['round_id']}"
                    result = run_candidate_model_comparison(
                        cfg,
                        report_slug=compare_slug,
                        bootstrap_samples=int(bootstrap_samples),
                        candidate_models=list(round_plan["candidate_models"]),
                        feature_pool=str(feature_pool),
                        feature_map_model=str(feature_map_model),
                        train_fraction=float(round_plan["train_fraction"]),
                        validation_fraction=float(round_plan["validation_fraction"]),
                        split_label=str(round_plan["validation_variant"]),
                        execution_metadata={
                            "execution_data_scope": "bounded_mlb_feature_slice",
                            "source_label": "local_processed_mlb_features",
                            "production_grade": False,
                            "holdout_scheme": str(round_plan["holdout_scheme"]),
                            "theory_traceability_matrix": "docs/mlb/theory_traceability_matrix.md",
                        },
                    )
                    summary_payload = json.loads(Path(result.summary_path).read_text())
                    comparison_artifacts = _comparison_artifacts(result, summary_payload)
                    round_artifacts: dict[str, str] = {"round_root": str(round_dir)}
                    for key, value in comparison_artifacts.items():
                        round_artifacts[key] = str(_copy_artifact(value, round_dir))
                    round_plan["completed_at_utc"] = utc_now_iso()
                    round_plan["status"] = _round_status_from_payload(summary_payload)
                    round_plan["artifacts"].update(round_artifacts)
                    round_plan["promotion_decision"] = summary_payload.get("promotion_decision", {})
                    round_plan["split"] = summary_payload.get("metadata", {}).get("split", {})
                    round_plan["candidate_count"] = len(summary_payload.get("candidate_scorecards", []))
                    round_plan["recommended_model"] = str(summary_payload.get("promotion_decision", {}).get("recommended_model") or "")
                    test_metrics = _read_test_metrics(Path(comparison_artifacts.get("test_metrics_path", result.test_metrics_path)))
                    baseline_record = _baseline_record_from_test_metrics(
                        round_plan=round_plan,
                        test_metrics=test_metrics,
                        comparison_artifacts=comparison_artifacts,
                        round_artifacts=round_artifacts,
                    )
                    if baseline_record is not None:
                        baseline_key = (baseline_record["feature_pool"], baseline_record["validation_variant"], baseline_record["model_name"])
                        latest_records[baseline_key] = baseline_record
                        round_history[baseline_key].append(round_plan["round_id"])
                    for candidate_record in summary_payload.get("candidate_scorecards", []):
                        record = _candidate_record_from_summary(
                            round_plan=round_plan,
                            summary_payload=summary_payload,
                            candidate_record=candidate_record,
                            test_metrics=test_metrics,
                            comparison_artifacts=comparison_artifacts,
                            round_artifacts=round_artifacts,
                        )
                        key = (record["feature_pool"], record["validation_variant"], record["model_name"])
                        latest_records[key] = record
                        round_history[key].append(round_plan["round_id"])
                    _append_ledger_entry(
                        ledger_path,
                        {
                            "task_id": task_id,
                            "status": "completed",
                            "command": command,
                            "callable": "src.research.mlb_model_tournament:run_mlb_model_tournament",
                            "started_at_utc": round_plan["started_at_utc"],
                            "completed_at_utc": round_plan["completed_at_utc"],
                            "artifacts": round_artifacts,
                            "error": None,
                            "feature_pool": feature_pool,
                            "validation_variant": round_plan["validation_variant"],
                            "holdout_variant": round_plan["holdout_variant"],
                            "holdout_scheme": round_plan["holdout_scheme"],
                        },
                    )

        records = [record for record in latest_records.values() if record["model_name"] != "intercept_only"]
        leaderboard_records = list(latest_records.values())
        for record in records:
            key = (record["feature_pool"], record["validation_variant"], record["model_name"])
            record["round_history"] = list(dict.fromkeys(round_history.get(key, record.get("round_history", []))))
            if record["round_history"]:
                record["round_id"] = record["round_history"][-1]
                record["round_index"] = int(next(
                    (item["round_index"] for item in completed_rounds if item["round_id"] == record["round_id"]),
                    record["round_index"],
                ))
        for record in leaderboard_records:
            key = (record["feature_pool"], record["validation_variant"], record["model_name"])
            record["round_history"] = list(dict.fromkeys(round_history.get(key, record.get("round_history", []))))
            if record["round_history"]:
                record["round_id"] = record["round_history"][-1]
                record["round_index"] = int(next(
                    (item["round_index"] for item in completed_rounds if item["round_id"] == record["round_id"]),
                    record["round_index"],
                ))
        leaderboard = _sort_leaderboard(_flatten_candidate_records(leaderboard_records))
        leaderboard["rank"] = range(1, len(leaderboard) + 1)
        recommendation_rows = _recommendation_rows(leaderboard)
        current_best_models_path = write_current_best_models_from_tournament(
            cfg,
            run_id=resolved_run_id,
            artifact_root=root,
            summary_path=summary_path,
            leaderboard_rows=leaderboard.to_dict(orient="records"),
            recommendation_rows=recommendation_rows,
            recommendation_path=recommendation_json_path,
            production_grade=False,
            execution_scope="bounded_mlb_feature_slice",
        )
        summary_payload = {
            "league": league,
            "run_id": resolved_run_id,
            "generated_at_utc": utc_now_iso(),
            "artifact_root": str(root),
            "round_count": len(completed_rounds),
            "candidate_count": len(records),
            "top_candidate_id": None if leaderboard.empty else str(leaderboard.iloc[0]["candidate_id"]),
            "top_model_name": None if leaderboard.empty else str(leaderboard.iloc[0]["model_name"]),
            "leaderboard_path": str(leaderboard_csv_path),
            "leaderboard_json_path": str(leaderboard_json_path),
            "recommendation_path": str(recommendation_json_path),
            "theory_compliance_summary_path": str(theory_compliance_summary_path),
            "manifest_path": str(manifest_path),
            "current_best_models_path": str(current_best_models_path),
        }
        recommendation_payload = {
            "league": league,
            "run_id": resolved_run_id,
            "generated_at_utc": utc_now_iso(),
            "status": "complete",
            "summary": {
                "top_candidate_id": summary_payload["top_candidate_id"],
                "top_model_name": summary_payload["top_model_name"],
                "candidate_count": len(records),
                "round_count": len(completed_rounds),
            },
            "recommendations": recommendation_rows,
        }
        theory_compliance_summary_path.write_text(
            _render_theory_compliance_summary(
                run_id=resolved_run_id,
                leaderboard=leaderboard,
                holdout_schemes=resolved_holdout_schemes,
            )
        )
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = utc_now_iso()
        manifest["rounds"] = completed_rounds
        manifest["candidate_scorecards_path"] = str(scorecards_path)
        manifest["leaderboard_path"] = str(leaderboard_csv_path)
        manifest["leaderboard_json_path"] = str(leaderboard_json_path)
        manifest["recommendation_path"] = str(recommendation_json_path)
        manifest["theory_compliance_summary_path"] = str(theory_compliance_summary_path)
        manifest["summary_path"] = str(summary_path)
        manifest["current_best_models_path"] = str(current_best_models_path)
        _write_json(scorecards_path, records)
        leaderboard.to_csv(leaderboard_csv_path, index=False)
        leaderboard_json_path.write_text(to_json(leaderboard.to_dict(orient="records")) + "\n")
        _write_json(recommendation_json_path, recommendation_payload)
        recommendation_md_path.write_text(
            _render_recommendation_markdown(run_id=resolved_run_id, root=root, leaderboard=leaderboard, recommendation_rows=recommendation_rows)
        )
        _write_json(summary_path, summary_payload)
        _write_json(manifest_path, manifest)
        return TournamentResult(
            run_id=resolved_run_id,
            artifact_root=root,
            manifest_path=manifest_path,
            ledger_path=ledger_path,
            summary_path=summary_path,
            scorecards_path=scorecards_path,
            leaderboard_csv_path=leaderboard_csv_path,
            leaderboard_json_path=leaderboard_json_path,
            recommendation_json_path=recommendation_json_path,
            recommendation_md_path=recommendation_md_path,
            theory_compliance_summary_path=theory_compliance_summary_path,
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["completed_at_utc"] = utc_now_iso()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(manifest_path, manifest)
        if completed_rounds:
            manifest["rounds"] = completed_rounds
        failure_entry = {
            "task_id": f"{resolved_run_id}:tournament",
            "status": "failed",
            "command": f"python3 scripts/run_mlb_model_tournament.py --config {getattr(cfg.paths, 'config_path', '') or 'configs/mlb.yaml'} --run-id {resolved_run_id}",
            "callable": "src.research.mlb_model_tournament:run_mlb_model_tournament",
            "started_at_utc": manifest.get("generated_at_utc"),
            "completed_at_utc": utc_now_iso(),
            "artifacts": {"artifact_root": str(root), "manifest_path": str(manifest_path)},
            "error": f"{type(exc).__name__}: {exc}",
        }
        _append_ledger_entry(ledger_path, failure_entry)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MLB model tournament harness.")
    parser.add_argument("--config", default="configs/mlb.yaml", help="Path to the MLB config file.")
    parser.add_argument("--run-id", default=None, help="Optional run identifier override.")
    parser.add_argument("--candidate-models", default=None, help="Optional comma-separated candidate-model filter.")
    parser.add_argument("--feature-pools", default="full_screened", help="Comma-separated feature pools to sweep.")
    parser.add_argument("--validation-variant", default=DEFAULT_VALIDATION_VARIANT, help="Manifest label for the validation split variant.")
    parser.add_argument("--holdout-variant", default=DEFAULT_HOLDOUT_VARIANT, help="Manifest label for the holdout split variant.")
    parser.add_argument(
        "--holdout-schemes",
        default=None,
        help="Comma-separated holdout schemes to run, or 'all'. Valid: outer_40_30_30, outer_50_25_25, outer_60_20_20.",
    )
    parser.add_argument("--feature-map-model", default="glm_ridge", help="Feature-map model for production_model_map pools.")
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES, help="Bootstrap sample count per round.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if hasattr(cfg.paths, "config_path"):
        cfg.paths.config_path = args.config
    result = run_mlb_model_tournament(
        cfg,
        run_id=args.run_id,
        candidate_models=args.candidate_models,
        feature_pools=args.feature_pools,
        validation_variant=args.validation_variant,
        holdout_variant=args.holdout_variant,
        holdout_schemes=args.holdout_schemes,
        feature_map_model=args.feature_map_model,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    print(f"MLB_TOURNAMENT::{result.run_id}::{result.artifact_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
