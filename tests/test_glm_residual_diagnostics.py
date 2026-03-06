import numpy as np
import pandas as pd

from src.evaluation.diagnostics_glm import save_glm_diagnostics, working_residual_logit, working_weight_logit
from src.models.glm_logit import GLMLogitModel


def test_logit_working_residual_and_weight_follow_paper_formula():
    y = np.array([1.0, 0.0, 1.0])
    p = np.array([0.8, 0.25, 0.6])

    residual = working_residual_logit(y, p)
    weight = working_weight_logit(p)

    np.testing.assert_allclose(residual, np.array([1.25, -1.3333333333333333, 1.6666666666666667]))
    np.testing.assert_allclose(weight, np.array([0.16, 0.1875, 0.24]))


def test_save_glm_diagnostics_creates_all_feature_residual_outputs(tmp_path):
    rng = np.random.default_rng(42)
    n = 420
    signal = rng.normal(0.0, 1.0, n)
    counter = rng.normal(0.0, 1.0, n)
    logits = 1.1 * signal - 0.8 * counter
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "home_win": y,
            "signal": signal,
            "counter": counter,
        }
    )
    glm = GLMLogitModel(c=1.0)
    glm.fit(df, feature_columns=["signal", "counter"])

    report = save_glm_diagnostics(
        df,
        glm=glm,
        target_col="home_win",
        out_dir=str(tmp_path / "plots"),
        prefix="unit_glm",
    )

    assert report["summary"]["status"] == "ok"
    assert report["summary"]["n_observations"] == n
    assert report["summary"]["n_features"] == 2
    assert (tmp_path / report["summary"]["linear_predictor_plot_file"]).exists()
    assert (tmp_path / report["summary"]["deviance_plot_file"]).exists()

    feature_summary = report["feature_summary"]
    assert feature_summary["feature"].tolist() == ["signal", "counter"]
    assert feature_summary["bin_count"].gt(0).all()
    for rel_path in feature_summary["working_residual_plot_file"].tolist():
        assert (tmp_path / rel_path).exists()
    for rel_path in feature_summary["partial_residual_plot_file"].tolist():
        assert (tmp_path / rel_path).exists()

    linear_bins = report["linear_predictor_bins"]
    assert not linear_bins.empty
    assert linear_bins["working_residual_mean"].notna().all()

    working_bins = report["feature_working_bins"]
    partial_bins = report["partial_residual_bins"]
    assert set(working_bins["feature"]) == {"signal", "counter"}
    assert set(partial_bins["feature"]) == {"signal", "counter"}
    assert partial_bins["component_mean"].notna().all()
