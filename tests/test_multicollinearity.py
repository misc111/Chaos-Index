import numpy as np
import pandas as pd

from src.evaluation.validation_stability import assess_multicollinearity, vif_table


def test_multicollinearity_assessment_flags_critical_clusters():
    rng = np.random.default_rng(42)
    n = 400
    x1 = rng.normal(0, 1, n)
    x2 = x1.copy()
    x3 = 0.98 * x1 + rng.normal(0, 0.01, n)
    x4 = rng.normal(0, 1, n)
    home_win = (x1 + 0.2 * x4 + rng.normal(0, 0.4, n) > 0).astype(int)

    df = pd.DataFrame(
        {
            "home_win": home_win,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "const": np.ones(n),
        }
    )

    report = assess_multicollinearity(df, features=["x1", "x2", "x3", "x4", "const"])

    summary = report["summary"]
    structural = report["structural"].set_index("feature")
    vif = report["vif"].set_index("feature")
    pairwise = report["pairwise"]

    assert summary["status"] == "critical"
    assert summary["exact_duplicate_features"] == 1
    assert summary["critical_condition_dimensions"] >= 1
    assert summary["full_rank"] is False

    assert structural.loc["x2", "duplicate_of"] == "x1"
    assert "exact_duplicate" in structural.loc["x2", "flags"]
    assert "constant" in structural.loc["const", "flags"]
    assert bool(structural.loc["const", "included_in_matrix"]) is False

    duplicate_pair = pairwise[(pairwise["feature_a"] == "x1") & (pairwise["feature_b"] == "x2")]
    assert not duplicate_pair.empty
    assert duplicate_pair.iloc[0]["severity"] == "severe"
    assert duplicate_pair.iloc[0]["abs_corr"] >= 0.999

    assert np.isinf(vif.loc["x1", "vif"]) or float(vif.loc["x1", "vif"]) >= 10.0
    assert np.isinf(vif.loc["x2", "vif"]) or float(vif.loc["x2", "vif"]) >= 10.0


def test_multicollinearity_assessment_handles_missing_near_constant_and_vif_compatibility():
    rng = np.random.default_rng(7)
    n = 200
    base = rng.normal(0, 1, n)
    sparse = base + rng.normal(0, 0.1, n)
    sparse[:40] = np.nan
    near_constant = np.zeros(n)
    near_constant[-1] = 1.0
    independent = rng.normal(0, 1, n)

    df = pd.DataFrame(
        {
            "base": base,
            "sparse": sparse,
            "near_constant": near_constant,
            "independent": independent,
        }
    )

    report = assess_multicollinearity(df, features=["base", "sparse", "near_constant", "independent"])
    structural = report["structural"].set_index("feature")
    vif = vif_table(df, features=["base", "sparse", "near_constant", "independent"])

    assert "near_constant" in structural.loc["near_constant", "flags"]
    assert bool(structural.loc["sparse", "included_in_matrix"]) is True
    assert report["summary"]["n_complete_cases"] == n - 40

    assert {"feature", "vif", "condition_number"}.issubset(vif.columns)
    assert "flags" in vif.columns
    assert len(vif) == 4
