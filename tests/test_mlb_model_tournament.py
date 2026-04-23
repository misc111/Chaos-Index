from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.common.config import load_config
from src.research.mlb_model_tournament import _sort_leaderboard, run_mlb_model_tournament


def _write_round_artifacts(base_dir: Path, report_slug: str, rows: list[dict[str, object]]) -> SimpleNamespace:
    base_dir.mkdir(parents=True, exist_ok=True)
    report_path = base_dir / f"{report_slug}_summary.md"
    summary_path = base_dir / f"{report_slug}_comparison_summary.json"
    scorecards_csv = base_dir / f"{report_slug}_candidate_scorecards.csv"
    scorecards_json = base_dir / f"{report_slug}_candidate_scorecards.json"
    leaderboard_csv = base_dir / f"{report_slug}_candidate_leaderboard.csv"
    leaderboard_json = base_dir / f"{report_slug}_candidate_leaderboard.json"
    recommendation_json = base_dir / f"{report_slug}_recommendation.json"
    recommendation_surface = base_dir / f"{report_slug}_recommendation_surface.json"
    artifact_manifest = base_dir / f"{report_slug}_artifact_manifest.json"
    validation_metrics = base_dir / f"{report_slug}_validation_metrics.csv"
    test_metrics = base_dir / f"{report_slug}_test_metrics.csv"
    bootstrap = base_dir / f"{report_slug}_bootstrap.csv"
    fit_stats = base_dir / f"{report_slug}_fit_stats.csv"
    cv = base_dir / f"{report_slug}_cv_summary.csv"
    service_output = base_dir / f"{report_slug}_service_output.json"
    service_manifest = base_dir / f"{report_slug}_service_manifest.json"

    report_path.write_text("# round report\n")
    scorecards_csv.write_text("model_name\n")
    scorecards_json.write_text("[]\n")
    leaderboard_csv.write_text("model_name\n")
    leaderboard_json.write_text("[]\n")
    recommendation_json.write_text(json.dumps({"recommended_model": "glm_elastic_net"}) + "\n")
    recommendation_surface.write_text(json.dumps({"decision": {"recommended_model": "glm_elastic_net"}}) + "\n")
    artifact_manifest.write_text(json.dumps({"report_slug": report_slug}) + "\n")
    validation_metrics.write_text("model_name,log_loss\n")
    bootstrap.write_text("reference_model,comparison_model,delta_log_loss_mean\n")
    fit_stats.write_text("model_name,n_features\n")
    cv.write_text("model_name,mean_log_loss\n")
    service_output.write_text(json.dumps({"report_slug": report_slug}) + "\n")
    service_manifest.write_text(json.dumps({"report_slug": report_slug}) + "\n")

    metric_rows = []
    for row in rows:
        metric_rows.append(
            {
                "model_name": row["model_name"],
                "display_name": row.get("display_name", row["model_name"]),
                "params": row.get("params", ""),
                "log_loss": row["log_loss"],
                "brier": row["brier"],
                "accuracy": row.get("accuracy", 0.6),
                "auc": row.get("auc", 0.7),
                "ece": row.get("ece", 0.02),
                "calibration_alpha": row.get("calibration_alpha", 1.0),
                "calibration_beta": row.get("calibration_beta", 1.0),
                "normalized_gini": row.get("normalized_gini", 0.25),
                "fit_status": row.get("fit_status", "ok"),
            }
        )
    test_metric_rows = [
        {
            "model_name": "intercept_only",
            "display_name": "Intercept Only",
            "params": "p_home_win=0.500",
            "log_loss": 0.700,
            "brier": 0.250,
            "accuracy": 0.50,
            "auc": 0.50,
            "ece": 0.01,
            "calibration_alpha": 1.0,
            "calibration_beta": 1.0,
            "normalized_gini": 0.0,
            "fit_status": "ok",
        },
        *metric_rows,
    ]
    pd.DataFrame(test_metric_rows).to_csv(test_metrics, index=False)
    pd.DataFrame(
        [
            {"model_name": "intercept_only", "display_name": "Intercept Only", "params": "p_home_win=0.500", "log_loss": 0.700, "brier": 0.250, "accuracy": 0.50, "auc": 0.50, "ece": 0.01, "calibration_alpha": 1.0, "calibration_beta": 1.0, "normalized_gini": 0.0, "fit_status": "ok"},
            {"model_name": rows[0]["model_name"], "display_name": rows[0].get("display_name", rows[0]["model_name"]), "params": rows[0].get("params", ""), "log_loss": rows[0]["log_loss"], "brier": rows[0]["brier"], "accuracy": 0.60, "auc": 0.71, "ece": 0.02, "calibration_alpha": 1.0, "calibration_beta": 1.0, "normalized_gini": 0.26, "fit_status": "ok"},
        ]
    ).to_csv(validation_metrics, index=False)

    candidate_scorecards = []
    for row in metric_rows:
        candidate_scorecards.append(
            {
                "model_name": row["model_name"],
                "lane": "core" if row["model_name"] != "intercept_only" else "benchmark",
                "target_name": "moneyline_home_win",
                "distribution": "binomial",
                "link_function": "logit",
                "validation_metrics": {
                    "final_holdout_log_loss": row["log_loss"],
                    "final_holdout_brier": row["brier"],
                },
                "stability_metrics": {
                    "validation_to_test_log_loss_delta": row.get("stability_delta", 0.0),
                    "rank_shift_validation_to_final": row.get("rank_shift", 0),
                },
                "calibration_summary": {
                    "final_holdout_ece": row.get("ece", 0.02),
                    "calibration_alpha": row.get("calibration_alpha", 1.0),
                    "calibration_beta": row.get("calibration_beta", 1.0),
                },
                "complement_summary": {
                    "display_name": row.get("display_name", row["model_name"]),
                    "family": "glm",
                    "governance_note": "",
                    "recommendation_tier": "research_screen_leader" if row["model_name"] == "glm_elastic_net" else "challenge_watchlist",
                    "recommended_for_next_stage": row["model_name"] == "glm_elastic_net",
                },
                "rejection_reasons": [],
            }
        )

    summary_payload = {
        "league": "MLB",
        "report_slug": report_slug,
        "candidate_scorecards": candidate_scorecards,
        "promotion_decision": {
            "recommended_model": "glm_elastic_net",
            "baseline_model": "intercept_only",
            "status": "research_recommended",
            "rationale": "unit test",
            "evidence": {},
            "rejected_models": {},
        },
        "artifacts": {
            "report_path": str(report_path),
            "summary_path": str(summary_path),
            "candidate_scorecards_path": str(scorecards_csv),
            "candidate_scorecards_contract_path": str(scorecards_json),
            "leaderboard_path": str(leaderboard_csv),
            "leaderboard_json_path": str(leaderboard_json),
            "recommendation_path": str(recommendation_json),
            "recommendation_surface_path": str(recommendation_surface),
            "artifact_manifest_path": str(artifact_manifest),
            "validation_metrics_path": str(validation_metrics),
            "test_metrics_path": str(test_metrics),
            "bootstrap_path": str(bootstrap),
            "fit_stats_path": str(fit_stats),
            "cv_path": str(cv),
            "service_output_path": str(service_output),
            "mlb_service_output_path": str(service_output),
            "mlb_service_manifest_path": str(service_manifest),
        },
        "metadata": {
            "artifact_root": str(base_dir),
            "recommended_display_name": "Elastic Net GLM",
            "recommendation_surface": {"decision": {"recommended_model": "glm_elastic_net"}},
        },
    }
    summary_path.write_text(json.dumps(summary_payload) + "\n")

    return SimpleNamespace(
        league="MLB",
        report_slug=report_slug,
        report_path=report_path,
        summary_path=summary_path,
        candidate_scorecards_path=scorecards_csv,
        candidate_scorecards_contract_path=scorecards_json,
        recommendation_path=recommendation_json,
        validation_metrics_path=validation_metrics,
        test_metrics_path=test_metrics,
        bootstrap_path=bootstrap,
        fit_stats_path=fit_stats,
        cv_path=cv,
        recommendation_model="glm_elastic_net",
        recommendation_display_name="Elastic Net GLM",
        validation_metrics=pd.DataFrame(),
        test_metrics=pd.DataFrame(),
        bootstrap_summary=pd.DataFrame(),
        leaderboard_path=leaderboard_csv,
        leaderboard_json_path=leaderboard_json,
        recommendation_surface_path=recommendation_surface,
        artifact_manifest_path=artifact_manifest,
        service_output_path=service_output,
    )


def test_mlb_model_tournament_writes_manifest_ledger_and_summary(tmp_path, monkeypatch) -> None:
    cfg = load_config("configs/mlb.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.processed_dir = str(tmp_path / "processed")
    cfg.paths.db_path = str(tmp_path / "processed" / "mlb_forecast.db")

    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_compare(cfg, **kwargs):
        candidate_models = tuple(kwargs["candidate_models"])
        calls.append((kwargs["report_slug"], candidate_models))
        round_dir = tmp_path / "source" / kwargs["report_slug"]
        model_rows = [
            {
                "model_name": model_name,
                "display_name": model_name.replace("_", " ").title(),
                "params": "c=0.5" if model_name == "glm_ridge" else ("c=0.25; l1_ratio=0.5" if model_name == "glm_elastic_net" else ""),
                "log_loss": {"glm_vanilla": 0.610, "glm_ridge": 0.570, "glm_lasso": 0.580, "glm_elastic_net": 0.560, "glmm_logit": 0.595, "dglm_margin": 0.592, "gam_spline": 0.586}.get(model_name, 0.600),
                "brier": {"glm_vanilla": 0.220, "glm_ridge": 0.200, "glm_lasso": 0.205, "glm_elastic_net": 0.190, "glmm_logit": 0.208, "dglm_margin": 0.207, "gam_spline": 0.202}.get(model_name, 0.210),
            }
            for model_name in candidate_models
        ]
        return _write_round_artifacts(round_dir, kwargs["report_slug"], model_rows)

    monkeypatch.setattr("src.research.mlb_model_tournament.run_candidate_model_comparison", fake_compare)

    result = run_mlb_model_tournament(
        cfg,
        run_id="20260422T120000Z_unit",
        bootstrap_samples=10,
    )

    assert result.artifact_root == tmp_path / "artifacts" / "reports" / "mlb" / "tournament" / "20260422T120000Z_unit"
    assert result.manifest_path.exists()
    assert result.ledger_path.exists()
    assert result.scorecards_path.exists()
    assert result.leaderboard_csv_path.exists()
    assert result.leaderboard_json_path.exists()
    assert result.recommendation_json_path.exists()
    assert result.recommendation_md_path.exists()
    assert result.theory_compliance_summary_path.exists()
    current_best_models_path = tmp_path / "artifacts" / "reports" / "mlb" / "current_best_models.json"
    assert current_best_models_path.exists()

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["status"] == "complete"
    assert manifest["rounds"][0]["status"] == "completed"
    assert manifest["rounds"][-1]["candidate_models"][-1] == "gam_spline"
    assert manifest["candidate_scorecards_path"] == str(result.scorecards_path)
    assert manifest["current_best_models_path"] == str(current_best_models_path)

    ledger_lines = [line for line in result.ledger_path.read_text().splitlines() if line.strip()]
    assert len(ledger_lines) == 6
    first_entry = json.loads(ledger_lines[0])
    assert first_entry["status"] == "started"
    assert first_entry["task_id"].startswith("20260422T120000Z_unit:r01_full_screened_outer_40_30_30_linear_core")

    scorecards = json.loads(result.scorecards_path.read_text())
    assert any(row["candidate_id"] == "full_screened::outer_40_30_30::glm_elastic_net" for row in scorecards)
    elastic = next(row for row in scorecards if row["model_name"] == "glm_elastic_net")
    assert elastic["lambda"] == 4.0
    assert elastic["l1_ratio"] == 0.5
    assert elastic["theory_classification"] == "core-supported"
    assert elastic["artifact_paths"]["round_root"].endswith("r03_full_screened_outer_40_30_30_nonlinear_challengers")

    leaderboard = pd.read_csv(result.leaderboard_csv_path)
    assert leaderboard.iloc[0]["model_name"] == "glm_elastic_net"
    assert list(leaderboard["rank"]) == list(range(1, len(leaderboard) + 1))

    recommendation = json.loads(result.recommendation_json_path.read_text())
    assert recommendation["status"] == "complete"
    statuses = {row["status"] for row in recommendation["recommendations"]}
    assert statuses <= {
        "promotion-review-ready",
        "shortlist",
        "theory-blocked",
        "validation-blocked",
        "reject",
        "not-enough-evidence",
    }
    assert any(row["status"] == "shortlist" for row in recommendation["recommendations"])
    assert {row["theory_classification"] for row in recommendation["recommendations"]} <= {
        "core-supported",
        "theory-compatible extension",
        "experimental",
    }
    top_recommendation = recommendation["recommendations"][0]
    assert top_recommendation["theory_core_gate"] is True
    assert top_recommendation["validation_contract_complete"] is False
    assert top_recommendation["promotion_review_required"] is True
    current_best = json.loads(current_best_models_path.read_text())
    assert current_best["source_kind"] == "mlb_model_tournament"
    assert current_best["recommended_model"] == "glm_elastic_net"
    assert current_best["top_models"][0]["model_name"] == "glm_elastic_net"
    assert current_best["artifact_paths"]["summary_path"] == str(result.summary_path)

    assert len(calls) == 3
    assert calls[0][1] == ("glm_vanilla", "glm_ridge", "glm_lasso", "glm_elastic_net")


def test_mlb_tournament_sort_ignores_roi_secondary_signal() -> None:
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "model_name": "glm_ridge",
                "log_loss": 0.60,
                "brier": 0.20,
                "calibration__final_holdout_ece": 0.02,
                "stability__validation_to_test_log_loss_delta": 0.01,
                "stability__rank_shift_validation_to_final": 0,
                "roi": 0.25,
            },
            {
                "candidate_id": "b",
                "model_name": "glm_elastic_net",
                "log_loss": 0.60,
                "brier": 0.20,
                "calibration__final_holdout_ece": 0.02,
                "stability__validation_to_test_log_loss_delta": 0.01,
                "stability__rank_shift_validation_to_final": 0,
                "roi": 0.90,
            },
        ]
    )

    sorted_frame = _sort_leaderboard(frame)

    assert list(sorted_frame["candidate_id"]) == ["a", "b"]
