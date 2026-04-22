from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.research.candidate_models import PenalizedLogitCandidate, VanillaGLMBinomialCandidate
from src.research.model_comparison import CandidateSpec
from src.services.model_compare import compare_candidate_models
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
    assert artifacts["mlb_latest_material_artifacts"] == latest_manifest["material_artifacts"]
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
