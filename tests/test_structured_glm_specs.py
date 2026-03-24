import pytest

from src.research.structured_glm_specs import load_structured_glm_selection, resolve_structured_glm_experiment


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


def test_resolve_structured_glm_experiment_without_spec_is_passthrough():
    calls: list[dict[str, object]] = []

    def candidate_spec_builder(feature_sets, **kwargs):
        calls.append({"feature_sets": feature_sets, **kwargs})
        return ["ok"]

    resolution = resolve_structured_glm_experiment(
        league="NBA",
        available_features=["rest_diff"],
        spec_path=None,
    )
    assert resolution.selection is None
    assert resolution.extend_feature_pool_note("base note", connector="; plus ") == "base note"

    specs = resolution.build_candidate_specs(
        feature_sets={"dummy": True},
        selected_models={"glm_ridge"},
        candidate_spec_builder=candidate_spec_builder,
    )
    assert specs == ["ok"]
    assert calls == [{"feature_sets": {"dummy": True}, "selected_models": {"glm_ridge"}}]


def test_resolve_structured_glm_experiment_builds_glm_overrides_and_extends_note(tmp_path):
    spec_path = tmp_path / "nba_structured_glm.yaml"
    spec_path.write_text(
        """
version: 1
league: NBA
experiment_name: unit_structured_glm
default_slate: baseline
default_width_variant: medium
slates:
  baseline:
    feature_order:
      - diff_form_point_margin
      - rest_diff
      - elo_home_prob
    width_variants:
      medium:
        feature_count: 2
"""
    )

    calls: list[dict[str, object]] = []

    def candidate_spec_builder(feature_sets, **kwargs):
        calls.append({"feature_sets": feature_sets, **kwargs})
        return ["glm_ridge", "glm_vanilla"]

    resolution = resolve_structured_glm_experiment(
        league="NBA",
        available_features=["diff_form_point_margin", "rest_diff", "noise_only"],
        spec_path=str(spec_path),
    )

    assert resolution.selection is not None
    assert resolution.selection.features == ("diff_form_point_margin", "rest_diff")
    assert "structured GLM experiment `unit_structured_glm`" in resolution.extend_feature_pool_note(
        "base note",
        connector=" Plus ",
        suffix=".",
    )

    specs = resolution.build_candidate_specs(
        feature_sets={"dummy": True},
        selected_models={"glm_ridge", "glm_vanilla"},
        candidate_spec_builder=candidate_spec_builder,
    )

    assert specs == ["glm_ridge", "glm_vanilla"]
    assert calls and "glm_feature_overrides" in calls[0]
    overrides = calls[0]["glm_feature_overrides"]
    assert isinstance(overrides, dict)
    assert overrides["glm_ridge"] == ["diff_form_point_margin", "rest_diff"]
