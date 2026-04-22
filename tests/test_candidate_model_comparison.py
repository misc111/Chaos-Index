import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.research.candidate_models import (
    DGLMMarginCandidate,
    GAMSplineCandidate,
    GLMMLogitCandidate,
    MARSHingeCandidate,
    PenalizedLogitCandidate,
    VanillaGLMBinomialCandidate,
)
from src.research.model_comparison import (
    CandidateSpec,
    ComparisonRunResult,
    _feature_screening,
    run_candidate_model_comparison,
)
from src.services.model_compare import compare_candidate_models


def _synthetic_candidate_frame(n: int = 210) -> pd.DataFrame:
    rng = np.random.default_rng(12)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    teams = np.array(["ANA", "BOS", "CHI", "DAL", "EDM", "FLA"])
    home_team = teams[np.arange(n) % len(teams)]
    away_team = teams[(np.arange(n) + 2) % len(teams)]

    linear = rng.normal(0.0, 1.0, n)
    smooth = 0.7 * linear**2 - 0.2 * linear + rng.normal(0.0, 0.15, n)
    hinge = np.maximum(linear - 0.2, 0.0) - np.maximum(-0.4 - linear, 0.0)
    rest_diff = rng.normal(0.0, 1.0, n)
    goalie_quality_diff = rng.normal(0.0, 1.0, n)
    elo_home_prob = 1.0 / (1.0 + np.exp(-(0.8 * linear + 0.2 * rest_diff)))
    dyn_home_prob = 1.0 / (1.0 + np.exp(-(0.6 * linear + 0.4 * hinge)))
    noise = rng.normal(0.0, 1.0, n)

    latent_margin = 1.1 * linear + 0.6 * smooth + 0.7 * hinge + 0.3 * rest_diff + 0.25 * goalie_quality_diff + noise
    home_win = (latent_margin > 0.0).astype(int)
    home_score = np.where(home_win == 1, 4 + np.clip(np.round(latent_margin), 0, 3), 2 + np.clip(np.round(latent_margin), -2, 1))
    away_score = np.where(home_win == 1, 2 + np.clip(np.round(-latent_margin), -1, 1), 4 + np.clip(np.round(-latent_margin), 0, 3))

    return pd.DataFrame(
        {
            "game_id": np.arange(1, n + 1),
            "season": [20252026] * n,
            "game_date_utc": dates.date.astype(str),
            "start_time_utc": dates.astype(str),
            "venue": ["Arena"] * n,
            "home_team": home_team,
            "away_team": away_team,
            "status_final": [1] * n,
            "home_win": home_win,
            "as_of_utc": ["2025-01-01T00:00:00+00:00"] * n,
            "home_score": home_score.astype(int),
            "away_score": away_score.astype(int),
            "diff_signal_linear": linear,
            "diff_signal_smooth": smooth,
            "diff_signal_hinge": hinge,
            "rest_diff": rest_diff,
            "goalie_quality_diff": goalie_quality_diff,
            "elo_home_prob": elo_home_prob,
            "dyn_home_prob": dyn_home_prob,
            "noise_feature": noise,
            "duplicate_signal_linear": linear,
            "constant_feature": np.ones(n),
        }
    )


def test_feature_screening_drops_constant_and_exact_duplicate():
    df = _synthetic_candidate_frame(120)
    kept, report = _feature_screening(
        df,
        [
            "diff_signal_linear",
            "diff_signal_smooth",
            "duplicate_signal_linear",
            "constant_feature",
            "rest_diff",
        ],
    )

    assert "diff_signal_linear" in kept
    assert "rest_diff" in kept
    assert "duplicate_signal_linear" not in kept
    assert "constant_feature" not in kept

    reason_lookup = report.set_index("feature")["reason"].to_dict()
    assert reason_lookup["duplicate_signal_linear"] == "exact_duplicate"
    assert reason_lookup["constant_feature"] == "constant_or_singleton_on_fit_window"


def test_glmm_candidate_predicts_probabilities_for_seen_and_unseen_teams():
    df = _synthetic_candidate_frame(180)
    train = df.iloc[:140].copy()
    test = df.iloc[140:].copy()
    test.loc[test.index[:3], "home_team"] = "SEA"
    test.loc[test.index[:3], "away_team"] = "UTA"

    model = GLMMLogitCandidate(
        fixed_features=["diff_signal_linear", "diff_signal_smooth", "rest_diff", "elo_home_prob"],
    )
    model.fit(train)
    probs = model.predict_proba(test)

    assert len(probs) == len(test)
    assert np.isfinite(probs).all()
    assert ((probs > 0.0) & (probs < 1.0)).all()


def test_candidate_model_comparison_writes_report_bundle(tmp_path, monkeypatch):
    df = _synthetic_candidate_frame(210)
    cfg = load_config("configs/default.yaml")
    cfg.data.league = "MLB"
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.processed_dir = str(tmp_path / "processed")
    cfg.paths.db_path = str(tmp_path / "processed" / "mlb_forecast.db")
    cfg.modeling.cv_splits = 2

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / "features.csv", index=False)

    def small_specs(feature_sets, *, selected_models=None):
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
            CandidateSpec(
                model_name="glmm_logit",
                display_name="GLMM Logit",
                param_grid=[{"feature_cap": min(4, len(feature_sets.glmm_features))}],
                builder=lambda fs, params: GLMMLogitCandidate(
                    fixed_features=fs.glmm_features[: int(params["feature_cap"])],
                ),
            ),
            CandidateSpec(
                model_name="dglm_margin",
                display_name="DGLM Margin",
                param_grid=[{"feature_cap": min(4, len(feature_sets.dglm_features)), "iterations": 1}],
                builder=lambda fs, params: DGLMMarginCandidate(
                    features=fs.dglm_features[: int(params["feature_cap"])],
                    iterations=int(params["iterations"]),
                ),
            ),
            CandidateSpec(
                model_name="gam_spline",
                display_name="GAM Spline",
                param_grid=[{"feature_cap": min(3, len(feature_sets.gam_features)), "n_knots": 4, "c": 0.5}],
                builder=lambda fs, params: GAMSplineCandidate(
                    linear_features=fs.core_features[:3],
                    spline_features=fs.gam_features[: int(params["feature_cap"])],
                    n_knots=int(params["n_knots"]),
                    c=float(params["c"]),
                ),
            ),
            CandidateSpec(
                model_name="mars_hinge",
                display_name="MARS Hinge",
                param_grid=[{"feature_cap": min(2, len(feature_sets.mars_features)), "knots_per_feature": 3, "interaction_degree": 1, "c": 0.25}],
                builder=lambda fs, params: MARSHingeCandidate(
                    linear_features=fs.core_features[:3],
                    hinge_features=fs.mars_features[: int(params["feature_cap"])],
                    knots_per_feature=int(params["knots_per_feature"]),
                    interaction_degree=int(params["interaction_degree"]),
                    c=float(params["c"]),
                ),
            ),
        ]
        if not selected_models:
            return specs
        return [spec for spec in specs if spec.model_name in selected_models]

    monkeypatch.setattr("src.research.model_comparison._candidate_specs", small_specs)

    result = run_candidate_model_comparison(
        cfg,
        report_slug="unit_candidate_compare",
        bootstrap_samples=80,
    )

    assert result.report_path.exists()
    assert result.summary_path.exists()
    assert result.report_path.parent == (tmp_path / "artifacts" / "reports" / "mlb")
    history_report = tmp_path / "artifacts" / "reports" / "history" / result.report_path.name
    assert not history_report.exists()
    assert result.validation_metrics_path.exists()
    assert result.test_metrics_path.exists()
    assert result.bootstrap_path.exists()
    assert result.candidate_scorecards_path.exists()
    assert result.candidate_scorecards_contract_path.exists()
    assert result.recommendation_path.exists()
    assert result.leaderboard_path is not None and result.leaderboard_path.exists()
    assert result.leaderboard_json_path is not None and result.leaderboard_json_path.exists()
    assert result.recommendation_surface_path is not None and result.recommendation_surface_path.exists()
    assert result.artifact_manifest_path is not None and result.artifact_manifest_path.exists()
    assert result.recommendation_model in result.test_metrics["model_name"].tolist()
    report_text = result.report_path.read_text()
    assert "This comparison is a research screen only" in report_text
    summary_payload = json.loads(result.summary_path.read_text())
    assert "candidate_scorecards" in summary_payload
    assert "promotion_decision" in summary_payload
    assert summary_payload["promotion_decision"]["status"] in {"research_recommended", "research_hold"}
    assert summary_payload["promotion_decision"]["rationale"]
    assert summary_payload["metadata"]["leaderboard_preview"]
    assert summary_payload["metadata"]["recommendation_surface"]["decision"]["recommended_model"]
    first_scorecard = summary_payload["candidate_scorecards"][0]
    assert "log_loss_improvement_vs_intercept" in first_scorecard["validation_metrics"]
    assert "recommendation_tier" in first_scorecard["complement_summary"]


def test_compare_candidate_models_generates_mlb_service_output(tmp_path, monkeypatch):
    cfg = load_config("configs/default.yaml")
    cfg.data.league = "MLB"
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.processed_dir = str(tmp_path / "processed")
    cfg.paths.db_path = str(tmp_path / "processed" / "mlb_forecast.db")
    Path(cfg.paths.processed_dir).mkdir(parents=True, exist_ok=True)

    history_dir = tmp_path / "artifacts" / "reports" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    report_path = history_dir / "unit_mlb_service_report.md"
    summary_path = history_dir / "unit_mlb_service_summary.json"
    candidate_scorecards_path = history_dir / "unit_mlb_service_candidate_scorecards.csv"
    candidate_scorecards_contract_path = history_dir / "unit_mlb_service_candidate_scorecards.json"
    recommendation_path = history_dir / "unit_mlb_service_recommendation.json"
    leaderboard_json_path = history_dir / "unit_mlb_service_leaderboard.json"
    recommendation_surface_path = history_dir / "unit_mlb_service_surface.json"
    validation_metrics_path = history_dir / "unit_mlb_service_validation.csv"
    test_metrics_path = history_dir / "unit_mlb_service_test.csv"
    bootstrap_path = history_dir / "unit_mlb_service_bootstrap.csv"
    fit_stats_path = history_dir / "unit_mlb_service_fit_stats.csv"
    cv_path = history_dir / "unit_mlb_service_cv.csv"

    report_path.write_text("# unit report\n")
    candidate_scorecards_path.write_text("model_name\nglm_ridge\n")
    candidate_scorecards_contract_path.write_text(json.dumps([{"model_name": "glm_ridge"}]) + "\n")
    recommendation_path.write_text(
        json.dumps(
            {
                "recommended_model": "glm_ridge",
                "baseline_model": "intercept_only",
                "status": "research_recommended",
                "rationale": "unit",
                "evidence": {},
                "rejected_models": {},
            }
        )
        + "\n"
    )
    leaderboard_json_path.write_text(json.dumps([{"model_name": "glm_ridge", "final_holdout_rank": 1}]) + "\n")
    recommendation_surface_path.write_text(
        json.dumps({"decision": {"recommended_model": "glm_ridge"}, "top_candidates": []}) + "\n"
    )
    validation_metrics_path.write_text("model_name,log_loss\nglm_ridge,0.6\n")
    test_metrics_path.write_text("model_name,log_loss\nglm_ridge,0.5\n")
    bootstrap_path.write_text("reference_model,comparison_model,delta_log_loss_mean\nglm_ridge,glm_vanilla,0.01\n")
    fit_stats_path.write_text("model_name,n_features\nglm_ridge,2\n")
    cv_path.write_text("model_name,mean_log_loss\nglm_ridge,0.6\n")

    summary_payload = {
        "league": "MLB",
        "report_slug": "unit_mlb_service",
        "candidate_scorecards": [
            {
                "model_name": "glm_ridge",
                "validation_metrics": {"final_holdout_log_loss": 0.5},
                "complement_summary": {"display_name": "GLM Ridge", "recommendation_tier": "research_screen_leader"},
            }
        ],
        "promotion_decision": {
            "recommended_model": "glm_ridge",
            "baseline_model": "intercept_only",
            "status": "research_recommended",
            "rationale": "unit rationale",
            "evidence": {},
            "rejected_models": {},
        },
        "artifacts": {
            "candidate_scorecards_path": str(candidate_scorecards_path),
            "candidate_scorecards_contract_path": str(candidate_scorecards_contract_path),
            "recommendation_path": str(recommendation_path),
            "recommendation_surface_path": str(recommendation_surface_path),
            "leaderboard_json_path": str(leaderboard_json_path),
            "cv_path": str(cv_path),
        },
        "metadata": {
            "recommended_display_name": "GLM Ridge",
            "recommendation_surface": {"decision": {"recommended_model": "glm_ridge"}},
        },
    }
    summary_path.write_text(json.dumps(summary_payload) + "\n")

    fake_result = ComparisonRunResult(
        league="MLB",
        report_slug="unit_mlb_service",
        report_path=report_path,
        summary_path=summary_path,
        candidate_scorecards_path=candidate_scorecards_path,
        candidate_scorecards_contract_path=candidate_scorecards_contract_path,
        recommendation_path=recommendation_path,
        validation_metrics_path=validation_metrics_path,
        test_metrics_path=test_metrics_path,
        bootstrap_path=bootstrap_path,
        fit_stats_path=fit_stats_path,
        cv_path=cv_path,
        recommendation_model="glm_ridge",
        recommendation_display_name="GLM Ridge",
        validation_metrics=pd.DataFrame(),
        test_metrics=pd.DataFrame(),
        bootstrap_summary=pd.DataFrame(),
        leaderboard_json_path=leaderboard_json_path,
        recommendation_surface_path=recommendation_surface_path,
    )

    monkeypatch.setattr(
        "src.services.model_compare.run_candidate_model_comparison",
        lambda *args, **kwargs: fake_result,
    )
    result = compare_candidate_models(cfg, report_slug="unit_mlb_service", bootstrap_samples=20)

    assert result.service_output_path is not None
    assert result.service_output_path.exists()
    service_payload = json.loads(result.service_output_path.read_text())
    assert service_payload["league"] == "MLB"
    assert service_payload["report_lane"] == "mlb"
    assert service_payload["recommended_model"] == "glm_ridge"
    assert service_payload["promotion_decision"]["status"] == "research_recommended"
    assert service_payload["manifest_path"].endswith("candidate_model_comparison_latest_manifest.json")
    manifest_payload = json.loads(Path(service_payload["manifest_path"]).read_text())
    assert manifest_payload["report_lane"] == "mlb"
    mlb_dir = tmp_path / "artifacts" / "reports" / "mlb"
    assert Path(service_payload["canonical_artifact_root"]) == mlb_dir
    assert Path(manifest_payload["canonical_artifact_root"]) == mlb_dir
    expected_material_names = {
        "report_path": report_path.name,
        "summary_path": summary_path.name,
        "candidate_scorecards_path": candidate_scorecards_path.name,
        "candidate_scorecards_contract_path": candidate_scorecards_contract_path.name,
        "recommendation_path": recommendation_path.name,
        "recommendation_surface_path": recommendation_surface_path.name,
        "leaderboard_json_path": leaderboard_json_path.name,
    }
    for key, file_name in expected_material_names.items():
        canonical_path = Path(manifest_payload["material_artifacts"][key])
        assert canonical_path == mlb_dir / file_name
        assert canonical_path.exists()
        assert "reports/history" not in canonical_path.as_posix()
    assert service_payload["material_artifacts"] == manifest_payload["material_artifacts"]
    assert Path(service_payload["report_path"]) == mlb_dir / report_path.name
    assert Path(service_payload["summary_path"]) == mlb_dir / summary_path.name

    persisted_summary = json.loads(result.summary_path.read_text())
    assert persisted_summary["artifacts"]["service_output_path"]
    assert persisted_summary["artifacts"]["mlb_service_output_path"]
    assert persisted_summary["artifacts"]["mlb_service_manifest_path"]
    assert persisted_summary["artifacts"]["mlb_latest_material_artifacts"] == manifest_payload["material_artifacts"]
    assert report_path.exists()
    assert "reports/history" not in json.dumps(persisted_summary["artifacts"]["mlb_latest_material_artifacts"])


def test_candidate_model_comparison_can_use_production_feature_map_for_penalized_glms(tmp_path, monkeypatch):
    df = _synthetic_candidate_frame(210)
    cfg = load_config("configs/default.yaml")
    cfg.data.league = "NBA"
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.processed_dir = str(tmp_path / "processed")
    cfg.paths.db_path = str(tmp_path / "processed" / "nba_forecast.db")
    cfg.modeling.cv_splits = 2

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / "features.csv", index=False)

    def small_penalized_specs(feature_sets, *, selected_models=None):
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
                model_name="glm_elastic_net",
                display_name="Elastic Net GLM",
                param_grid=[{"c": 0.25, "l1_ratio": 0.5}],
                builder=lambda fs, params: PenalizedLogitCandidate(
                    model_name="glm_elastic_net",
                    display_name="Elastic Net GLM",
                    features=fs.screened_features,
                    penalty="elasticnet",
                    c=float(params["c"]),
                    l1_ratio=float(params["l1_ratio"]),
                    solver="saga",
                ),
            ),
            CandidateSpec(
                model_name="glm_lasso",
                display_name="Lasso GLM",
                param_grid=[{"c": 0.25}],
                builder=lambda fs, params: PenalizedLogitCandidate(
                    model_name="glm_lasso",
                    display_name="Lasso GLM",
                    features=fs.screened_features,
                    penalty="l1",
                    c=float(params["c"]),
                    solver="saga",
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

    monkeypatch.setattr("src.research.model_comparison._candidate_specs", small_penalized_specs)
    monkeypatch.setattr(
        "src.research.model_comparison.load_model_feature_map",
        lambda league: {"glm_ridge": ["diff_signal_linear", "rest_diff"]},
    )

    result = run_candidate_model_comparison(
        cfg,
        report_slug="unit_candidate_compare_production_map",
        bootstrap_samples=40,
        candidate_models=["glm_ridge", "glm_lasso", "glm_elastic_net", "glm_vanilla"],
        feature_pool="production_model_map",
        feature_map_model="glm_ridge",
    )

    metric_models = set(result.test_metrics["model_name"].tolist())
    assert metric_models == {
        "intercept_only",
        "glm_ridge",
        "glm_lasso",
        "glm_elastic_net",
        "glm_vanilla",
    }

    fit_stats = pd.read_csv(result.report_path.with_name("unit_candidate_compare_production_map_nba_fit_stats.csv"))
    penalized_rows = fit_stats[fit_stats["model_name"].isin(["glm_ridge", "glm_lasso", "glm_elastic_net"])]
    assert set(penalized_rows["n_features"].tolist()) == {2}

    report_text = result.report_path.read_text()
    assert "Restricted to the production model feature map for `glm_ridge`" in report_text


def test_candidate_model_comparison_can_use_research_broad_dataset(tmp_path, monkeypatch):
    df = _synthetic_candidate_frame(210)
    cfg = load_config("configs/default.yaml")
    cfg.data.league = "NBA"
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.processed_dir = str(tmp_path / "processed" / "nba")
    cfg.paths.db_path = str(tmp_path / "processed" / "nba_forecast.db")
    cfg.research.inner_folds = 2

    research_processed_dir = tmp_path / "processed" / "research" / "nba"
    research_processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(research_processed_dir / "features.csv", index=False)

    def small_specs(feature_sets, *, selected_models=None):
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

    result = run_candidate_model_comparison(
        cfg,
        report_slug="unit_candidate_compare_research",
        bootstrap_samples=40,
        feature_pool="research_broad",
    )

    assert result.report_path.exists()
    assert result.recommendation_model in result.test_metrics["model_name"].tolist()
    assert "research dataset's broad numeric feature pool" in result.report_path.read_text()


def test_candidate_model_comparison_supports_structured_glm_research_spec(tmp_path):
    df = _synthetic_candidate_frame(210)
    cfg = load_config("configs/default.yaml")
    cfg.data.league = "NBA"
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.processed_dir = str(tmp_path / "processed")
    cfg.paths.db_path = str(tmp_path / "processed" / "nba_forecast.db")
    cfg.modeling.cv_splits = 2

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / "features.csv", index=False)

    spec_path = tmp_path / "nba_structured_glm.yaml"
    spec_path.write_text(
        """
version: 1
league: NBA
experiment_name: unit_structured_glm
default_slate: synthetic
default_width_variant: medium
slates:
  synthetic:
    feature_order:
      - diff_signal_linear
      - rest_diff
      - elo_home_prob
      - dyn_home_prob
      - diff_signal_hinge
    width_variants:
      medium:
        feature_count: 3
"""
    )

    result = run_candidate_model_comparison(
        cfg,
        report_slug="unit_candidate_compare_structured_glm",
        bootstrap_samples=40,
        candidate_models=["glm_ridge", "glm_lasso", "glm_elastic_net", "glm_vanilla"],
        feature_pool="full_screened",
        structured_glm_spec_path=str(spec_path),
    )

    metric_models = set(result.test_metrics["model_name"].tolist())
    assert metric_models == {
        "intercept_only",
        "glm_ridge",
        "glm_lasso",
        "glm_elastic_net",
        "glm_vanilla",
    }

    fit_stats = pd.read_csv(result.report_path.with_name("unit_candidate_compare_structured_glm_nba_fit_stats.csv"))
    glm_rows = fit_stats[fit_stats["model_name"].isin(["glm_ridge", "glm_lasso", "glm_elastic_net", "glm_vanilla"])]
    assert set(glm_rows["n_features"].tolist()) == {3}

    report_text = result.report_path.read_text()
    assert "structured GLM experiment `unit_structured_glm`" in report_text
