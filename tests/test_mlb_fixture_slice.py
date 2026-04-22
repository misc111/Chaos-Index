from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.generate_mlb_artifact_slice import (
    COMPARISON_ARTIFACT_PATH_KEYS,
    _fixture_execution_metadata,
    _validate_comparison_outputs,
    _validate_validation_outputs,
    build_fixture_frame,
)


def _cfg(tmp_path: Path):
    return SimpleNamespace(paths=SimpleNamespace(artifacts_dir=str(tmp_path / "artifacts")))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _comparison_with_summary(tmp_path: Path, *, history_key: str | None = None):
    cfg = _cfg(tmp_path)
    frame = build_fixture_frame(n_games=80, seed=7)
    metadata = _fixture_execution_metadata(frame=frame, cfg=cfg)
    reports_dir = Path(cfg.paths.artifacts_dir) / "reports" / "mlb"
    history_dir = Path(cfg.paths.artifacts_dir) / "reports" / "history"
    reports_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}
    for key in COMPARISON_ARTIFACT_PATH_KEYS:
        suffix = ".md" if key == "report_path" else ".json"
        path = reports_dir / f"fixture_{key}{suffix}"
        path.write_text("{}\n")
        artifacts[key] = str(path)

    if history_key:
        history_path = history_dir / f"fixture_{history_key}.json"
        history_path.write_text("{}\n")
        artifacts[history_key] = str(history_path)

    summary_path = Path(artifacts["summary_path"])
    summary_payload = {
        "metadata": {"execution_metadata": metadata},
        "artifacts": artifacts,
    }
    _write_json(summary_path, summary_payload)

    comparison = SimpleNamespace(
        report_path=Path(artifacts["report_path"]),
        summary_path=summary_path,
        candidate_scorecards_path=Path(artifacts["candidate_scorecards_path"]),
        candidate_scorecards_contract_path=Path(artifacts["candidate_scorecards_contract_path"]),
        recommendation_path=Path(artifacts["recommendation_path"]),
        validation_metrics_path=Path(artifacts["validation_metrics_path"]),
        test_metrics_path=Path(artifacts["test_metrics_path"]),
        bootstrap_path=Path(artifacts["bootstrap_path"]),
        fit_stats_path=Path(artifacts["fit_stats_path"]),
        cv_path=Path(artifacts["cv_path"]),
        leaderboard_path=Path(artifacts["leaderboard_path"]),
        leaderboard_json_path=Path(artifacts["leaderboard_json_path"]),
        recommendation_surface_path=Path(artifacts["recommendation_surface_path"]),
        artifact_manifest_path=Path(artifacts["artifact_manifest_path"]),
    )
    return cfg, metadata, comparison


def test_fixture_slice_metadata_is_explicitly_fixture_demo(tmp_path):
    cfg = _cfg(tmp_path)
    frame = build_fixture_frame(n_games=24, seed=11)

    metadata = _fixture_execution_metadata(frame=frame, cfg=cfg)

    assert metadata["execution_data_scope"] == "mlb_fixture_slice"
    assert metadata["execution_label_class"] == "fixture_or_demo"
    assert metadata["data_origin"] == "synthetic_fixture"
    assert metadata["production_grade"] is False
    assert metadata["required_comparison_output_dir"].endswith("artifacts/reports/mlb")
    assert metadata["required_validation_output_dir"].endswith("artifacts/validation/mlb")


def test_fixture_slice_comparison_outputs_must_resolve_under_reports_mlb(tmp_path):
    cfg, metadata, comparison = _comparison_with_summary(tmp_path)

    material_paths = _validate_comparison_outputs(cfg, comparison, metadata=metadata)

    reports_dir = Path(cfg.paths.artifacts_dir) / "reports" / "mlb"
    assert material_paths
    assert all(Path(path).resolve().is_relative_to(reports_dir.resolve()) for path in material_paths.values())


def test_fixture_slice_comparison_outputs_reject_history_paths(tmp_path):
    cfg, metadata, comparison = _comparison_with_summary(tmp_path, history_key="report_path")

    with pytest.raises(RuntimeError, match="artifacts/reports/history"):
        _validate_comparison_outputs(cfg, comparison, metadata=metadata)


def test_fixture_slice_validation_outputs_must_resolve_under_validation_mlb(tmp_path):
    cfg = _cfg(tmp_path)
    frame = build_fixture_frame(n_games=80, seed=13)
    metadata = _fixture_execution_metadata(frame=frame, cfg=cfg)
    validation_dir = Path(cfg.paths.artifacts_dir) / "validation" / "mlb"
    validation_dir.mkdir(parents=True, exist_ok=True)
    _write_json(validation_dir / "validation_manifest.json", {"league": "MLB", "sections": []})
    _write_json(validation_dir / "validation_outputs_contract.json", {"validation_outputs": []})
    _write_json(
        validation_dir / "validation_run_metadata.json",
        {
            "league": "MLB",
            "latest_validation_dir": "validation/mlb",
            "execution_metadata": metadata,
            "execution_data_scope": "mlb_fixture_slice",
            "execution_source_label": metadata["execution_source_label"],
            "execution_label_class": "fixture_or_demo",
            "production_grade": False,
        },
    )

    run_metadata = _validate_validation_outputs(cfg, metadata=metadata)

    assert run_metadata["latest_validation_dir"] == "validation/mlb"
    assert run_metadata["execution_label_class"] == "fixture_or_demo"
    assert run_metadata["production_grade"] is False
