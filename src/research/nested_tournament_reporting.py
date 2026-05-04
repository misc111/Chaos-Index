"""Artifact reporting helpers for nested MLB tournaments.

Reporting code is deliberately separated from fitting and selection so generated
CSV, JSON, Markdown, and bootstrap outputs can evolve without changing candidate
or governance logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.utils import ensure_dir
from src.evaluation.metrics import per_game_scores
from src.governance.evidence import (
    GOVERNANCE_CONTRACT_VERSION,
    THEORY_COMPATIBLE_ENGINEERING_SUPPORT,
    target_scope_fields,
)
from src.research.nested_tournament_contracts import NestedTournamentResult, _safe_json
from src.research.nested_tournament_evidence import nested_tournament_evidence_fields
from src.research.nested_tournament_features import _feature_coverage_summary
from src.research.nested_tournament_governance import _blocked_reason_summary, _gate_reason_labels, _reason_list


@dataclass(frozen=True, slots=True)
class NestedTournamentArtifactPaths:
    """Canonical output paths for a nested tournament run."""

    target_coverage_path: Path
    variant_leaderboard_path: Path
    family_champions_path: Path
    inter_family_path: Path
    predictions_path: Path
    cv_path: Path
    bootstrap_path: Path
    feature_coverage_path: Path
    feature_coverage_json_path: Path
    summary_path: Path
    report_path: Path
    current_best_path: Path


def nested_tournament_artifact_paths(*, artifact_root: Path, artifacts_dir: str | Path) -> NestedTournamentArtifactPaths:
    """Resolve the shared artifact path contract for sequential and parallel runs."""

    return NestedTournamentArtifactPaths(
        target_coverage_path=artifact_root / "target_coverage.csv",
        variant_leaderboard_path=artifact_root / "variant_leaderboard.csv",
        family_champions_path=artifact_root / "family_champions.csv",
        inter_family_path=artifact_root / "inter_family_leaderboard.csv",
        predictions_path=artifact_root / "final_holdout_predictions.csv",
        cv_path=artifact_root / "cv_folds.csv",
        bootstrap_path=artifact_root / "bootstrap.csv",
        feature_coverage_path=artifact_root / "feature_coverage_by_split.csv",
        feature_coverage_json_path=artifact_root / "feature_coverage_by_split.json",
        summary_path=artifact_root / "nested_tournament_summary.json",
        report_path=artifact_root / "nested_tournament_summary.md",
        current_best_path=Path(artifacts_dir) / "reports" / "mlb" / "current_best_models.json",
    )


def write_nested_tournament_artifacts(
    *,
    paths: NestedTournamentArtifactPaths,
    artifact_root: Path,
    league: str,
    resolved_run_id: str,
    source_kind: str,
    target_coverage: pd.DataFrame,
    validation_leaderboard: pd.DataFrame,
    family_champions: pd.DataFrame,
    inter_family: pd.DataFrame,
    predictions: pd.DataFrame,
    cv_frames: list[pd.DataFrame],
    bootstrap: pd.DataFrame,
    feature_coverage_rows: list[dict[str, Any]],
    feature_pool_note: str,
    report_title: str,
    report_intro: str,
    ranking_rule: dict[str, Any] | None = None,
    summary_extra: dict[str, Any] | None = None,
) -> NestedTournamentResult:
    """Write all shared tournament artifacts and return the public result object."""

    validation_autopsy_paths = _write_validation_autopsy(
        artifact_root=artifact_root,
        validation_leaderboard=validation_leaderboard,
        family_champions=family_champions,
        target_coverage=target_coverage,
    )

    target_coverage.to_csv(paths.target_coverage_path, index=False)
    validation_leaderboard.to_csv(paths.variant_leaderboard_path, index=False)
    family_champions.to_csv(paths.family_champions_path, index=False)
    inter_family.to_csv(paths.inter_family_path, index=False)
    predictions.to_csv(paths.predictions_path, index=False)
    (pd.concat(cv_frames, ignore_index=True) if cv_frames else pd.DataFrame()).to_csv(paths.cv_path, index=False)
    bootstrap.to_csv(paths.bootstrap_path, index=False)
    feature_coverage = pd.DataFrame(feature_coverage_rows)
    feature_coverage.to_csv(paths.feature_coverage_path, index=False)
    _safe_json(
        paths.feature_coverage_json_path,
        {
            "rows": feature_coverage.to_dict(orient="records"),
            "summary": _feature_coverage_summary(feature_coverage_rows),
        },
    )

    champion_rows = [
        bucket.sort_values(["target_rank"]).iloc[0].to_dict()
        for _, bucket in inter_family.groupby("target_name", sort=True)
    ] if not inter_family.empty else []
    blocked_champions = (
        family_champions[family_champions["champion_status"] != "shortlist"].to_dict(orient="records")
        if not family_champions.empty and "champion_status" in family_champions.columns
        else []
    )
    current_best: dict[str, Any] = {
        "governance_contract_version": GOVERNANCE_CONTRACT_VERSION,
        "evidence_scope": "nested_mlb_model_tournament",
        "theory_governance": THEORY_COMPATIBLE_ENGINEERING_SUPPORT,
        "target_scope": target_scope_fields(league=league),
        "league": str(league).upper(),
        "as_of_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_kind": source_kind,
        "report_slug": resolved_run_id,
        **nested_tournament_evidence_fields(
            run_id=resolved_run_id,
            source_kind=source_kind,
            artifact_root=artifact_root,
            summary_path=paths.summary_path,
        ),
        "target_champions": champion_rows,
        "blocked_family_champions": blocked_champions,
        "top_models": champion_rows,
        "artifact_paths": {
            "summary_path": str(paths.summary_path),
            "variant_leaderboard_path": str(paths.variant_leaderboard_path),
            "family_champions_path": str(paths.family_champions_path),
            "inter_family_leaderboard_path": str(paths.inter_family_path),
            "target_coverage_path": str(paths.target_coverage_path),
            "feature_coverage_by_split_path": str(paths.feature_coverage_path),
            "feature_coverage_by_split_json": str(paths.feature_coverage_json_path),
            **validation_autopsy_paths,
        },
    }
    if ranking_rule is not None:
        current_best["ranking_rule"] = dict(ranking_rule)
    ensure_dir(paths.current_best_path.parent)
    _safe_json(paths.current_best_path, current_best)

    summary_payload = {
        "governance_contract_version": GOVERNANCE_CONTRACT_VERSION,
        "evidence_scope": "nested_mlb_model_tournament",
        "theory_governance": THEORY_COMPATIBLE_ENGINEERING_SUPPORT,
        "target_scope": target_scope_fields(league=league),
        "league": str(league).upper(),
        "run_id": resolved_run_id,
        "generated_at_utc": current_best["as_of_utc"],
        "feature_pool_note": feature_pool_note,
        "targets": target_coverage.to_dict(orient="records"),
        "family_champions": family_champions.to_dict(orient="records"),
        "inter_family_leaderboard": inter_family.to_dict(orient="records"),
        "artifacts": current_best["artifact_paths"]
        | {
            "report_path": str(paths.report_path),
            "predictions_path": str(paths.predictions_path),
            "cv_path": str(paths.cv_path),
            "bootstrap_path": str(paths.bootstrap_path),
            "current_best_models_path": str(paths.current_best_path),
        },
    }
    if summary_extra:
        summary_payload.update(summary_extra)
    _safe_json(paths.summary_path, summary_payload)

    report_lines = [
        f"# {report_title}",
        "",
        report_intro,
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
    paths.report_path.write_text("\n".join(report_lines) + "\n")

    return NestedTournamentResult(
        run_id=resolved_run_id,
        artifact_root=artifact_root,
        summary_path=paths.summary_path,
        target_coverage_path=paths.target_coverage_path,
        variant_leaderboard_path=paths.variant_leaderboard_path,
        family_champions_path=paths.family_champions_path,
        inter_family_leaderboard_path=paths.inter_family_path,
        current_best_models_path=paths.current_best_path,
    )


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
                    "gate_reason_labels": ";".join(_gate_reason_labels(reasons)),
                    "blocked_reason_summary": _blocked_reason_summary(reasons),
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
