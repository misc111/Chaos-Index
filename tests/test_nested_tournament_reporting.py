from __future__ import annotations

from pathlib import Path

from src.research.nested_tournament_reporting import nested_tournament_artifact_paths


def test_nested_tournament_artifact_paths_are_canonical(tmp_path):
    """Artifact finalization uses one shared path contract for both runners."""

    paths = nested_tournament_artifact_paths(
        artifact_root=tmp_path / "artifacts" / "reports" / "mlb" / "tournament" / "nested" / "run_1",
        artifacts_dir=tmp_path / "artifacts",
    )

    assert paths.summary_path.name == "nested_tournament_summary.json"
    assert paths.report_path.name == "nested_tournament_summary.md"
    assert paths.current_best_path == Path(tmp_path / "artifacts" / "reports" / "mlb" / "current_best_models.json")
    assert paths.feature_coverage_json_path.name == "feature_coverage_by_split.json"
