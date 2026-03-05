from pathlib import Path

import pytest
import yaml

from src.training.feature_policy import apply_feature_policy, resolve_registry_path


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def test_production_requires_explicit_approval_for_bootstrap_and_drift(tmp_path: Path) -> None:
    registry_tmpl = str(tmp_path / "registry_{league}.yaml")

    with pytest.raises(RuntimeError):
        apply_feature_policy(
            ["f1", "f2"],
            league="NHL",
            mode="production",
            registry_path_template=registry_tmpl,
            approve_changes=False,
        )

    first = apply_feature_policy(
        ["f1", "f2"],
        league="NHL",
        mode="production",
        registry_path_template=registry_tmpl,
        approve_changes=True,
    )
    assert first.registry_created is True
    path = resolve_registry_path(registry_tmpl, league="NHL")
    assert path.exists()
    payload = _load_yaml(path)
    assert payload["active_features"] == ["f1", "f2"]

    with pytest.raises(RuntimeError):
        apply_feature_policy(
            ["f1", "f2", "f3"],
            league="NHL",
            mode="production",
            registry_path_template=registry_tmpl,
            approve_changes=False,
        )

    second = apply_feature_policy(
        ["f1", "f2", "f3"],
        league="NHL",
        mode="production",
        registry_path_template=registry_tmpl,
        approve_changes=True,
    )
    assert second.registry_updated is True
    payload = _load_yaml(path)
    assert payload["active_features"] == ["f1", "f2", "f3"]


def test_research_tracks_candidates_and_explicit_promotion(tmp_path: Path) -> None:
    registry_tmpl = str(tmp_path / "registry_{league}.yaml")

    # Bootstrap baseline in research mode.
    apply_feature_policy(
        ["a", "b"],
        league="NBA",
        mode="research",
        registry_path_template=registry_tmpl,
        approve_changes=False,
    )
    path = resolve_registry_path(registry_tmpl, league="NBA")
    payload = _load_yaml(path)
    assert payload["active_features"] == ["a", "b"]
    assert payload["candidate_features"] == []

    # New feature is tracked as candidate.
    out = apply_feature_policy(
        ["a", "b", "c"],
        league="NBA",
        mode="research",
        registry_path_template=registry_tmpl,
        approve_changes=False,
    )
    assert out.candidates_added == ["c"]
    payload = _load_yaml(path)
    assert payload["active_features"] == ["a", "b"]
    assert payload["candidate_features"] == ["c"]

    # Explicit approval promotes new contract into active set.
    apply_feature_policy(
        ["a", "b", "c"],
        league="NBA",
        mode="research",
        registry_path_template=registry_tmpl,
        approve_changes=True,
    )
    payload = _load_yaml(path)
    assert payload["active_features"] == ["a", "b", "c"]
    assert payload["candidate_features"] == []

