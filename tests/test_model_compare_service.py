from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.commands.modeling import best_models
from src.common.config import load_config
from src.research.candidate_models import PenalizedLogitCandidate, VanillaGLMBinomialCandidate
from src.research.model_comparison import CandidateSpec
from src.services.model_compare import compare_candidate_models, current_best_models_path, load_current_best_models
from src.storage.db import Database


def _synthetic_mlb_candidate_frame(n: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(20260420)
    dates = pd.date_range("2025-03-25", periods=n, freq="D")
    signal = rng.normal(0.0, 1.0, n)
    bullpen = rng.normal(0.0, 1.0, n)
    travel = rng.normal(0.0, 1.0, n)
    rest = rng.normal(0.0, 1.0, n)
    park = rng.normal(0.0, 1.0, n)
    market_offset_logit = 0.65 * signal + 0.15 * park + rng.normal(0.0, 0.3, n)
    logits = 0.80 * signal + 0.45 * bullpen - 0.20 * travel + 0.15 * rest + 0.30 * park
    prob = 1.0 / (1.0 + np.exp(-logits))
    home_win = rng.binomial(1, prob)

    return pd.DataFrame(
        {
            "game_id": np.arange(1, n + 1),
            "season": [2025] * n,
            "game_date_utc": dates.date.astype(str),
            "start_time_utc": dates.astype(str),
            "home_team": ["CHC"] * n,
            "away_team": ["LAD"] * n,
            "venue": ["Wrigley Field"] * n,
            "as_of_utc": ["2025-03-24T18:00:00+00:00"] * n,
            "status_final": [1] * n,
            "home_win": home_win,
            "diff_starting_pitcher_quality": signal,
            "diff_bullpen_quality": bullpen,
            "travel_diff": travel,
            "rest_diff": rest,
            "park_run_effect": park,
            "market_offset_logit": market_offset_logit,
        }
    )


def test_compare_candidate_models_persists_experiment_run_and_service_artifact(tmp_path, monkeypatch) -> None:
    df = _synthetic_mlb_candidate_frame()
    cfg = load_config("configs/mlb.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.processed_dir = str(tmp_path / "processed")
    cfg.paths.db_path = str(tmp_path / "processed" / "mlb_forecast.db")
    cfg.modeling.cv_splits = 2
    cfg.research.inner_folds = 2

    processed_dir = Path(cfg.paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / "features.csv", index=False)

    def small_specs(feature_sets, *, selected_models=None, glm_feature_overrides=None):
        specs = [
            CandidateSpec(
                model_name="glm_ridge",
                display_name="GLM Ridge",
                param_grid=[{"c": 0.5}],
                builder=lambda fs, params: PenalizedLogitCandidate(
                    model_name="glm_ridge",
                    display_name="GLM Ridge",
                    features=fs.screened_features,
                    penalty="l2",
                    c=float(params["c"]),
                    solver="lbfgs",
                ),
            ),
            CandidateSpec(
                model_name="glm_vanilla",
                display_name="Vanilla GLM",
                param_grid=[{}],
                builder=lambda fs, params: VanillaGLMBinomialCandidate(features=fs.screened_features),
            ),
        ]
        if not selected_models:
            return specs
        return [spec for spec in specs if spec.model_name in selected_models]

    monkeypatch.setattr("src.research.model_comparison._candidate_specs", small_specs)

    result = compare_candidate_models(
        cfg,
        report_slug="mlb_service_compare",
        bootstrap_samples=40,
        candidate_models=["glm_ridge", "glm_vanilla"],
    )

    assert result.report_path.exists()
    assert result.summary_path.exists()
    assert result.report_path.parent == (tmp_path / "artifacts" / "reports" / "mlb")
    legacy_history_report = tmp_path / "artifacts" / "reports" / "history" / result.report_path.name
    assert not legacy_history_report.exists()
    assert result.candidate_scorecards_path.exists()
    assert result.recommendation_path.exists()

    summary_payload = json.loads(result.summary_path.read_text())
    artifacts = summary_payload["artifacts"]
    metadata = summary_payload["metadata"]
    assert artifacts["candidate_scorecards_path"].endswith("_candidate_scorecards.csv")
    assert artifacts["recommendation_path"].endswith("_recommendation.json")
    assert artifacts["service_output_path"].endswith("candidate_comparison_service.json")
    assert artifacts["mlb_service_output_path"].endswith("candidate_model_comparison_latest.json")
    assert artifacts["mlb_service_manifest_path"].endswith("candidate_model_comparison_latest_manifest.json")
    assert metadata["experiment_run_id"].startswith("mlb::candidate_compare::")

    latest_payload = json.loads(Path(artifacts["mlb_service_output_path"]).read_text())
    latest_manifest = json.loads(Path(artifacts["mlb_service_manifest_path"]).read_text())
    current_best = json.loads(Path(artifacts["current_best_models_path"]).read_text())
    mlb_report_dir = tmp_path / "artifacts" / "reports" / "mlb"
    assert latest_payload["report_lane"] == "mlb"
    assert latest_manifest["report_lane"] == "mlb"
    assert Path(latest_payload["canonical_artifact_root"]) == mlb_report_dir
    assert Path(latest_manifest["canonical_artifact_root"]) == mlb_report_dir
    assert latest_payload["material_artifacts"]["report_path"] == str(result.report_path)
    assert latest_payload["material_artifacts"]["candidate_scorecards_path"] == str(result.candidate_scorecards_path)
    assert latest_payload["material_artifacts"]["recommendation_path"] == str(result.recommendation_path)
    assert latest_payload["material_artifacts"]["summary_path"] == str(result.summary_path)
    assert latest_manifest["material_artifacts"] == latest_payload["material_artifacts"]
    assert latest_manifest["service_output_path"] == artifacts["mlb_service_output_path"]
    assert latest_manifest["current_best_models_path"] == artifacts["current_best_models_path"]
    assert latest_payload["current_best_models_path"] == artifacts["current_best_models_path"]
    assert artifacts["mlb_latest_material_artifacts"] == latest_manifest["material_artifacts"]
    assert current_best["recommended_model"] == summary_payload["promotion_decision"]["recommended_model"]
    assert current_best["best_candidate_model"] == "glm_ridge"
    assert current_best["source_kind"] == "candidate_model_comparison"
    assert current_best["ranking_rule"]["primary"] == "final_holdout_log_loss"
    assert current_best["top_models"][0]["model_name"] == current_best["best_candidate_model"]
    for material_path in latest_manifest["material_artifacts"].values():
        path = Path(material_path)
        assert path.exists()
        assert path.parent == mlb_report_dir
        assert "reports/history" not in path.as_posix()

    db = Database(cfg.paths.db_path)
    runs = db.query("SELECT * FROM experiment_runs")
    assert len(runs) == 1
    row = runs[0]
    assert row["run_id"] == metadata["experiment_run_id"]
    assert row["league"] == "MLB"
    assert row["candidate_model_name"] == summary_payload["promotion_decision"]["recommended_model"]
    assert row["scorecard_path"].endswith("_candidate_scorecards.csv")
    assert row["promotion_path"].endswith("_recommendation.json")
    assert Path(artifacts["service_output_path"]).exists()


def test_best_models_command_falls_back_to_latest_experiment_run(tmp_path, capsys) -> None:
    cfg = load_config("configs/mlb.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.processed_dir = str(tmp_path / "processed")
    cfg.paths.db_path = str(tmp_path / "processed" / "mlb_forecast.db")

    db = Database(cfg.paths.db_path)
    db.init_schema()
    summary_payload = {
        "league": "MLB",
        "report_slug": "unit_compare",
        "target_name": "moneyline_home_win",
        "candidate_scorecards": [
            {
                "model_name": "glm_elastic_net",
                "target_name": "moneyline_home_win",
                "validation_metrics": {
                    "validation_log_loss": 0.61,
                    "validation_brier": 0.21,
                    "validation_auc": 0.66,
                    "final_holdout_log_loss": 0.59,
                    "final_holdout_brier": 0.20,
                    "final_holdout_auc": 0.67,
                    "final_holdout_ece": 0.03,
                },
                "stability_metrics": {"final_holdout_rank": 1},
                "calibration_summary": {},
                "complement_summary": {
                    "display_name": "Elastic Net GLM",
                    "family": "elastic_net",
                    "recommendation_tier": "research_screen_leader",
                    "recommended_for_next_stage": True,
                },
            },
            {
                "model_name": "dglm_margin",
                "target_name": "moneyline_home_win",
                "validation_metrics": {
                    "validation_log_loss": 0.62,
                    "validation_brier": 0.22,
                    "validation_auc": 0.64,
                    "final_holdout_log_loss": 0.60,
                    "final_holdout_brier": 0.21,
                    "final_holdout_auc": 0.65,
                    "final_holdout_ece": 0.04,
                },
                "stability_metrics": {"final_holdout_rank": 2},
                "calibration_summary": {},
                "complement_summary": {
                    "display_name": "DGLM Margin",
                    "family": "nonlinear",
                    "recommendation_tier": "challenge_watchlist",
                    "recommended_for_next_stage": False,
                    "margin_diagnostics": {
                        "promotion_gate": "separate_margin_and_bridge_validation_required",
                        "final_holdout": {
                            "status": "ok",
                            "bridge": "logit_calibrated",
                            "margin_rmse": 3.2,
                        },
                    },
                },
            },
        ],
        "promotion_decision": {
            "recommended_model": "glm_elastic_net",
            "baseline_model": "intercept_only",
            "status": "research_recommended",
            "rationale": "unit rationale",
            "evidence": {
                "promotion_ready": False,
                "intercept_only_benchmark": {
                    "model_name": "intercept_only",
                    "display_name": "Intercept Only",
                    "log_loss": 0.69,
                    "brier": 0.25,
                    "auc": 0.5,
                    "ece": 0.02,
                },
            },
            "rejected_models": {},
        },
        "metadata": {"recommended_display_name": "Elastic Net GLM"},
        "artifacts": {},
    }
    db.execute(
        """
        INSERT INTO experiment_runs(
          run_id, league, profile_key, brief_id, brief_key, incumbent_model_name, candidate_model_name,
          report_slug, report_path, scorecard_path, fold_metrics_path, promotion_path, status,
          auto_promote, started_at_utc, completed_at_utc, summary_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "mlb::candidate_compare::unit_compare",
            "MLB",
            "default",
            None,
            "candidate_model_comparison",
            "intercept_only",
            "glm_elastic_net",
            "unit_compare",
            "",
            "",
            "",
            "",
            "candidate_screen_complete",
            0,
            "2026-04-23T00:00:00Z",
            "2026-04-23T00:05:00Z",
            json.dumps(summary_payload, sort_keys=True),
        ),
    )

    assert not current_best_models_path(cfg).exists()
    payload = load_current_best_models(cfg)
    assert payload["recommended_model"] == "glm_elastic_net"
    assert payload["source_kind"] == "candidate_model_comparison"
    assert payload["top_models"][1]["model_name"] == "dglm_margin"
    assert payload["top_models"][1]["margin_diagnostics"]["final_holdout"]["margin_rmse"] == 3.2
    assert payload["top_models"][1]["final_holdout_log_loss"] == 0.60

    best_models(cfg, Namespace())
    captured = capsys.readouterr()
    rendered = json.loads(captured.out)
    assert rendered["recommended_model"] == "glm_elastic_net"
    assert rendered["top_models"][0]["display_name"] == "Elastic Net GLM"
