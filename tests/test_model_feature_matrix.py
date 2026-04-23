from __future__ import annotations

from src.reporting.model_feature_matrix import (
    CoefficientRow,
    ModelFeatureReport,
    _coefficient_detail_markdown,
    _parse_param_string,
    _selection_matrix_table,
)


def test_parse_param_string_handles_numeric_and_null_values() -> None:
    parsed = _parse_param_string("c=0.5; feature_cap=3; n_knots=5; l1_ratio=None; enabled=true")

    assert parsed == {
        "c": 0.5,
        "feature_cap": 3,
        "n_knots": 5,
        "l1_ratio": None,
        "enabled": True,
    }


def test_selection_matrix_and_detail_markdown_render_roles_and_betas() -> None:
    reports = [
        ModelFeatureReport(
            scope="production",
            model_name="glm_lasso",
            display_name="Lasso GLM",
            lane="core",
            selected_features={"rest_diff": ["driver"], "travel_diff": ["driver"]},
            coefficient_rows=[
                CoefficientRow(
                    term="rest_diff",
                    feature="rest_diff",
                    role="driver",
                    beta=0.125,
                    beta_scale="original_feature",
                ),
                CoefficientRow(
                    term="travel_diff",
                    feature="travel_diff",
                    role="driver",
                    beta=-0.5,
                    beta_scale="original_feature",
                ),
            ],
        ),
        ModelFeatureReport(
            scope="candidate_comparison",
            model_name="gam_spline",
            display_name="GAM Spline",
            lane="extension",
            selected_features={"dyn_var_diff": ["spline"]},
            coefficient_rows=[
                CoefficientRow(
                    term="dyn_var_diff::spline_01",
                    feature="dyn_var_diff",
                    role="spline_basis",
                    beta=0.3333,
                    beta_scale="basis_term",
                )
            ],
        ),
    ]

    matrix = _selection_matrix_table(reports)
    assert "rest_diff" in matrix
    assert "0.1250" in matrix
    assert "spline" in matrix

    detail = _coefficient_detail_markdown(reports)
    assert "dyn_var_diff::spline_01" in detail
    assert "basis_term" in detail
