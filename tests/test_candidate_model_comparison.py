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
from src.research.model_comparison import CandidateSpec, _feature_screening, run_candidate_model_comparison


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
    cfg.data.league = "NHL"
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.processed_dir = str(tmp_path / "processed")
    cfg.paths.db_path = str(tmp_path / "processed" / "nhl_forecast.db")
    cfg.modeling.cv_splits = 2

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / "features.csv", index=False)

    def small_specs(feature_sets):
        return [
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

    monkeypatch.setattr("src.research.model_comparison._candidate_specs", small_specs)

    result = run_candidate_model_comparison(
        cfg,
        report_slug="unit_candidate_compare",
        bootstrap_samples=80,
    )

    assert result.report_path.exists()
    assert result.validation_metrics_path.exists()
    assert result.test_metrics_path.exists()
    assert result.bootstrap_path.exists()
    assert result.recommendation_model in result.test_metrics["model_name"].tolist()
    assert "GLM Ridge" in result.report_path.read_text()
