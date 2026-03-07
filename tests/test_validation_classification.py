import math

from src.evaluation.validation_classification import validate_logistic_probability_model
from src.evaluation.validation_pipeline import build_validation_tasks


def test_validate_logistic_probability_model_emits_expected_reports(tmp_path):
    p = [
        0.95,
        0.90,
        0.82,
        0.78,
        0.71,
        0.66,
        0.61,
        0.58,
        0.54,
        0.51,
        0.49,
        0.46,
        0.42,
        0.37,
        0.31,
        0.26,
        0.19,
        0.12,
    ]
    y = [
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
    ]

    report = validate_logistic_probability_model(
        y,
        p,
        bins=5,
        current_tossup_half_width=0.05,
        plot_dir=tmp_path,
        plot_prefix="unit",
    )

    quantile_summary = report["quantile_summary"]
    actual_vs_predicted_summary = report["actual_vs_predicted_summary"]
    lift_summary = report["lift_summary"]
    lorenz_summary = report["lorenz_summary"]
    roc_summary = report["roc_summary"]
    tossup_summary = report["tossup_summary"]

    assert quantile_summary["n_obs"] == len(y)
    assert quantile_summary["bins_realized"] == 5
    assert quantile_summary["last_bin_actual_rate"] > quantile_summary["first_bin_actual_rate"]
    assert quantile_summary["monotonicity_violations"] <= 2

    assert actual_vs_predicted_summary["bins_realized"] == 5
    assert actual_vs_predicted_summary["mean_abs_identity_gap"] >= 0

    assert lift_summary["top_quantile_actual_lift"] > lift_summary["bottom_quantile_actual_lift"]
    assert lift_summary["top_vs_bottom_actual_lift_diff"] > 0

    assert lorenz_summary["top_decile_event_capture"] > 0
    assert lorenz_summary["normalized_gini"] > 0
    assert math.isclose(lorenz_summary["normalized_gini"], 2 * roc_summary["auroc"] - 1, rel_tol=1e-9)

    assert roc_summary["auroc"] > 0.8
    assert roc_summary["best_youden_threshold"] > 0.0
    assert len(report["operating_points"]) == 19
    assert not report["roc_curve"].empty

    assert math.isclose(tossup_summary["tossup_lower_threshold"], 0.45, rel_tol=1e-9)
    assert math.isclose(tossup_summary["tossup_upper_threshold"], 0.55, rel_tol=1e-9)
    assert math.isclose(tossup_summary["tossup_actual_rate"], 0.5, rel_tol=1e-9)
    assert tossup_summary["decisive_share"] < 1.0

    assert (tmp_path / "unit_quantile_plot.png").exists()
    assert (tmp_path / "unit_actual_vs_predicted_plot.png").exists()
    assert (tmp_path / "unit_lift_plot.png").exists()
    assert (tmp_path / "unit_lorenz_curve.png").exists()
    assert (tmp_path / "unit_roc_curve.png").exists()


def test_validation_task_registry_includes_classification_curves():
    task_names = [task.name for task in build_validation_tasks()]

    assert "classification_curves" in task_names
