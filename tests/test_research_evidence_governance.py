from __future__ import annotations

import pandas as pd

from src.research.nested_tournament_evaluation import _failed_row
from src.research.nested_tournament_governance import _select_family_champions
from src.services.model_compare import _build_current_best_models_payload_from_comparison
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
