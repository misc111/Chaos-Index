import pytest

from src.research.structured_glm_specs import load_structured_glm_selection


def test_load_structured_glm_selection_resolves_default_slate_and_variant(tmp_path):
    spec_path = tmp_path / "nba_structured_glm.yaml"
    spec_path.write_text(
        """
version: 1
league: NBA
experiment_name: unit_glm_spec
default_slate: baseline
default_width_variant: medium
slates:
  baseline:
    feature_order:
      - diff_form_point_margin
      - rest_diff
      - elo_home_prob
      - dyn_home_prob
    width_variants:
      medium:
        feature_count: 3
"""
    )

    selection = load_structured_glm_selection(
        league="NBA",
        available_features=["diff_form_point_margin", "rest_diff", "elo_home_prob", "noise_only"],
        spec_path=str(spec_path),
    )

    assert selection is not None
    assert selection.experiment_name == "unit_glm_spec"
    assert selection.slate_name == "baseline"
    assert selection.width_variant == "medium"
    assert selection.requested_feature_count == 3
    assert selection.available_feature_count == 3
    assert selection.features == ("diff_form_point_margin", "rest_diff", "elo_home_prob")
    assert set(selection.feature_overrides().keys()) == {"glm_ridge", "glm_elastic_net", "glm_lasso", "glm_vanilla"}


def test_load_structured_glm_selection_rejects_non_nba_league(tmp_path):
    spec_path = tmp_path / "nhl_structured_glm.yaml"
    spec_path.write_text(
        """
version: 1
league: NHL
experiment_name: invalid_for_lane
slates:
  baseline:
    feature_order:
      - diff_form_goal_diff
"""
    )

    with pytest.raises(ValueError, match="supported for NBA only"):
        load_structured_glm_selection(
            league="NHL",
            available_features=["diff_form_goal_diff"],
            spec_path=str(spec_path),
        )
