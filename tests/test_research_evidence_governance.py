from __future__ import annotations

import json

import pandas as pd

from src.research.nested_tournament_evaluation import _failed_row
from src.research.nested_tournament_governance import _select_family_champions
from src.common.config import load_config
from src.services.model_compare import _build_current_best_models_payload_from_comparison, write_current_best_models_from_promotion_review
from src.services.research_backtest import _promotion_summary


class _Target:
    target_name = "moneyline_home_win"
    target_col = "home_win"
    market = "moneyline"


class _Spec:
    target = _Target()
    candidate_key = "moneyline_home_win::glm_ridge::unit"
    model_name = "glm_ridge"
    display_name = "GLM Ridge"
    variant_key = "unit"
    variant_display_name = "Unit"
    feature_source = "unit"
    features = ("signal",)


def test_nested_tournament_rows_and_champions_carry_model_evidence_governance():
    """Tournament rows should not require readers to infer model lane metadata."""

    failed = _failed_row(_Spec(), split="validation", error="unit failure")

    assert failed["model_lane"] == "core"
    assert failed["theory_classification"] == "core-supported"
    assert failed["evidence_lane"] == "theory_core_diagnostics"
    assert failed["theory_governance"] == "direct theory implementation"
    assert failed["target_scope"]["market"] == "moneyline"

    champions = _select_family_champions(
        pd.DataFrame(
            [
                {
                    **failed,
                    "fit_status": "ok",
                    "fit_error": "",
                    "has_class_contrast": True,
                    "ece": 0.01,
                    "auc": 0.60,
                    "calibration_alpha": 0.01,
                    "calibration_beta": 1.0,
                    "top_vs_bottom_actual_lift_ratio": 1.2,
                    "validation_to_cv_log_loss_delta": 0.01,
                    "log_loss": 0.60,
                    "brier": 0.20,
                    "active_parameter_count": 1,
                    "n_features": 1,
                }
            ]
        )
    )

    assert champions.iloc[0]["model_lane"] == "core"
    assert champions.iloc[0]["theory_governance"] == "direct theory implementation"


def test_current_best_comparison_top_models_include_governance_metadata(tmp_path):
    """The latest comparison surface is an evidence packet, not just a ranking."""

    summary_payload = {
        "league": "MLB",
        "report_slug": "unit_compare",
        "target_name": "moneyline_home_win",
        "candidate_scorecards": [
            {
                "model_name": "glm_ridge",
                "target_name": "moneyline_home_win",
                "validation_metrics": {
                    "final_holdout_log_loss": 0.58,
                    "final_holdout_brier": 0.20,
                    "final_holdout_auc": 0.67,
                },
                "stability_metrics": {"final_holdout_rank": 1},
                "complement_summary": {"display_name": "GLM Ridge", "family": "linear"},
            }
        ],
        "promotion_decision": {
            "recommended_model": "glm_ridge",
            "baseline_model": "intercept_only",
            "status": "research_recommended",
            "rationale": "unit",
            "evidence": {"promotion_ready": False},
        },
        "metadata": {},
        "artifacts": {},
    }

    payload = _build_current_best_models_payload_from_comparison(
        summary_payload=summary_payload,
        canonical_output_dir=tmp_path,
    )

    assert payload["governance_contract_version"] == 1
    assert payload["evidence_scope"] == "candidate_model_comparison"
    assert payload["theory_governance"] == "theory-compatible engineering support"
    assert payload["target_scope"]["market"] == "moneyline"
    assert payload["top_models"][0]["model_lane"] == "core"
    assert payload["top_models"][0]["theory_classification"] == "core-supported"
    assert payload["top_models"][0]["governing_sources"]


def test_smoke_comparison_current_best_is_research_only_and_promotion_blocked(tmp_path):
    """Smoke or fixture-labeled latest recommendations must not look promotion eligible."""

    summary_payload = {
        "league": "MLB",
        "report_slug": "codex_current_best_models_smoke",
        "target_name": "moneyline_home_win",
        "candidate_scorecards": [
            {
                "model_name": "glm_ridge",
                "target_name": "moneyline_home_win",
                "validation_metrics": {
                    "final_holdout_log_loss": 0.58,
                    "final_holdout_brier": 0.20,
                    "final_holdout_auc": 0.67,
                },
                "stability_metrics": {"final_holdout_rank": 1},
                "complement_summary": {"display_name": "GLM Ridge", "family": "linear"},
            }
        ],
        "promotion_decision": {
            "recommended_model": "glm_ridge",
            "baseline_model": "intercept_only",
            "status": "research_recommended",
            "rationale": "unit",
            "evidence": {"promotion_ready": False},
        },
        "metadata": {"execution_metadata": {}},
        "artifacts": {
            "recommendation_path": "artifacts/reports/mlb/codex_current_best_models_smoke_mlb_recommendation.json",
            "summary_path": "artifacts/reports/mlb/codex_current_best_models_smoke_mlb_comparison_summary.json",
        },
    }

    payload = _build_current_best_models_payload_from_comparison(
        summary_payload=summary_payload,
        canonical_output_dir=tmp_path,
    )

    assert payload["latest_artifact_role"] == "latest_research_recommendation"
    assert payload["promotion_eligible"] is False
    assert payload["latest_promotion_eligible_evidence"] is None
    assert payload["evidence_status"]["evidence_stage"] == "fixture_demo_smoke"
    assert payload["evidence_status"]["promotion_eligible"] is False
    assert payload["evidence_status"]["production_grade"] is False
    assert "fixture_demo_smoke" in payload["evidence_status"]["blocked_reasons"]
    assert "latest research recommendation" in payload["latest_research_recommendation"]["pointer_semantics"]


def test_research_backtest_promotion_summary_declares_betting_overlay_governance():
    """Betting ROI promotion gates must not masquerade as theory-core validation."""

    scorecard = pd.DataFrame(
        [
            {
                "model_name": "baseline",
                "strategy": "flat",
                "mean_ending_bankroll": 5000.0,
                "mean_net_profit": 0.0,
                "median_roi": 0.0,
                "profit_winning_folds": 0,
                "profitable_folds": 0,
                "mean_max_drawdown": 100.0,
                "mean_log_loss": 0.69,
                "mean_brier": 0.25,
                "mean_ece": 0.02,
                "all_integrity_checks": True,
            },
            {
                "model_name": "glm_ridge",
                "strategy": "flat",
                "mean_ending_bankroll": 5200.0,
                "mean_net_profit": 200.0,
                "median_roi": 0.03,
                "profit_winning_folds": 2,
                "profitable_folds": 2,
                "mean_max_drawdown": 120.0,
                "mean_log_loss": 0.67,
                "mean_brier": 0.23,
                "mean_ece": 0.02,
                "all_integrity_checks": True,
            },
        ]
    )

    promotion = _promotion_summary(scorecard, best_model="glm_ridge", baseline_model="baseline")

    assert promotion["governance_contract_version"] == 1
    assert promotion["governance_lane"] == "betting_overlay"
    assert promotion["theory_governance"] == "betting overlay"
    assert promotion["ranking_rule"]["primary"] == "mean_ending_bankroll"


def test_current_best_promotion_review_keeps_betting_overlay_and_model_theory_labels(tmp_path):
    cfg = load_config("configs/mlb.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    payload = {
        "league": "MLB",
        "status": "rejected",
        "promoted": False,
        "active_model_name": "glm_ridge",
        "candidate_model_name": "gam_spline",
        "incumbent_model_name": "glm_ridge",
        "reason_summary": "unit promotion review stayed put",
        "strategy": "flat",
        "gates": {"research_backtest_eligible": True, "candidate_beats_incumbent": False},
        "production_grade": True,
        "ledger_audit": {
            "execution_scope": "full_immutable_pregame_mlb_research_backtest",
            "production_grade": True,
            "full_immutable_pregame_ledger": True,
            "pregame_timing_proven": True,
            "immutable_write_proven": True,
        },
        "policy": {"min_bet_count": 10},
        "candidate_scorecards": [
            {
                "model_name": "gam_spline",
                "lane": "extension",
                "target_name": "moneyline_home_win",
                "validation_metrics": {
                    "strategy": "flat",
                    "mean_ending_bankroll": 5100.0,
                    "mean_net_profit": 100.0,
                    "mean_log_loss": 0.68,
                    "mean_brier": 0.24,
                    "bet_count": 12,
                },
                "stability_metrics": {
                    "scorecard_rank": 1,
                    "profitable_folds": 3,
                    "profit_winning_folds": 3,
                    "all_integrity_checks": True,
                },
                "calibration_summary": {"mean_ece": 0.02},
                "complement_summary": {
                    "display_name": "GAM Spline",
                    "family": "nonlinear",
                    "governance_note": "Theory-compatible GLM extension.",
                    "promotion_role": "candidate_under_review",
                },
            }
        ],
        "promotion_decision": {"status": "rejected", "rationale": "unit"},
        "artifacts": {"promotion_path": str(tmp_path / "promotion_summary.json")},
    }

    path = write_current_best_models_from_promotion_review(cfg, run_id="unit-run", promotion_payload=payload)
    current_best = json.loads(path.read_text())

    assert current_best["evidence_scope"] == "research_desk_promotion_review"
    assert current_best["theory_governance"] == "betting overlay"
    assert current_best["candidate_model_name"] == "gam_spline"
    assert current_best["top_models"][0]["model_name"] == "gam_spline"
    assert current_best["top_models"][0]["model_lane"] == "extension"
    assert current_best["top_models"][0]["theory_classification"] == "theory-compatible extension"
    latest_path = tmp_path / "artifacts" / "reports" / "mlb" / "promotion_review_latest.json"
    eligible_latest_path = tmp_path / "artifacts" / "reports" / "mlb" / "promotion_eligible_evidence_latest.json"
    assert latest_path.exists()
    assert eligible_latest_path.exists()
    eligible_latest = json.loads(eligible_latest_path.read_text())
    assert current_best["latest_artifact_role"] == "latest_promotion_eligible_evidence"
    assert current_best["promotion_eligible"] is True
    assert current_best["production_ready"] is False
    assert current_best["evidence_status"]["evidence_stage"] == "promotion_eligible"
    assert current_best["evidence_status"]["production_ready"] is False
    assert "candidate_beats_incumbent" in current_best["evidence_status"]["readiness_blocked_reasons"]
    assert eligible_latest["latest_artifact_role"] == "latest_promotion_eligible_evidence"
    assert eligible_latest["promotion_eligible"] is True
    assert eligible_latest["promoted"] is False


def test_promotion_review_without_ledger_audit_cannot_update_eligible_latest(tmp_path):
    cfg = load_config("configs/mlb.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    output_dir = tmp_path / "artifacts" / "reports" / "mlb"
    output_dir.mkdir(parents=True)
    stale_eligible = output_dir / "promotion_eligible_evidence_latest.json"
    stale_eligible.write_text(json.dumps({"stale": True}) + "\n")

    payload = {
        "league": "MLB",
        "status": "rejected",
        "promoted": False,
        "active_model_name": "glm_ridge",
        "candidate_model_name": "gam_spline",
        "incumbent_model_name": "glm_ridge",
        "reason_summary": "unit blocked promotion review",
        "strategy": "flat",
        "gates": {"research_backtest_eligible": False, "validation_contract_complete": False},
        "policy": {"min_bet_count": 10},
        "candidate_scorecards": [],
        "promotion_decision": {"status": "rejected", "rationale": "unit"},
        "artifacts": {"promotion_path": str(tmp_path / "promotion_summary.json")},
    }

    path = write_current_best_models_from_promotion_review(cfg, run_id="unit-run-no-audit", promotion_payload=payload)
    current_best = json.loads(path.read_text())

    assert current_best["promotion_eligible"] is False
    assert current_best["latest_artifact_role"] == "blocked_promotion_evidence"
    assert current_best["latest_promotion_eligible_evidence"] is None
    assert current_best["evidence_status"]["evidence_stage"] == "research_only"
    assert "missing_ledger_audit" in current_best["evidence_status"]["blocked_reasons"]
    assert "not_full_immutable_pregame_mlb_ledger" in current_best["evidence_status"]["blocked_reasons"]
    assert not stale_eligible.exists()
