from __future__ import annotations

import json

import pandas as pd
import pytest

from src.common.config import load_config
from src.evaluation.validation_artifacts import ValidationOutputs, validation_path
from src.evaluation.validation_splits import concat_validation_frames, resolve_validation_split


def test_validation_outputs_write_manifest_and_defensively_copy_payloads(tmp_path):
    """Characterize the validation artifact registry before extracting it.

    The validation pipeline relies on registered sections being immutable from
    caller-side dataframe mutations, and on manifests remaining deterministic so
    staging and dashboard contracts can read the latest run without guessing at
    file names.
    """

    rows = pd.DataFrame({"metric": ["brier"], "value": [0.25]})
    outputs = ValidationOutputs()
    outputs.add_csv(
        section="scorecard",
        file_name=validation_path("diagnostics", "scorecard.csv"),
        rows=rows,
        tail_rows=1,
    )
    outputs.add_json(
        section="summary",
        file_name=validation_path("diagnostics", "summary.json"),
        payload={"status": "ok"},
    )

    rows.loc[0, "value"] = 999.0
    outputs.write(tmp_path, league="MLB", manifest_metadata={"run_label": "fixture"})

    manifest = json.loads((tmp_path / "validation_manifest.json").read_text())
    scorecard = pd.read_csv(tmp_path / "diagnostics" / "scorecard.csv")
    summary = json.loads((tmp_path / "diagnostics" / "summary.json").read_text())

    assert manifest["league"] == "MLB"
    assert manifest["metadata"] == {"run_label": "fixture"}
    assert manifest["sections"] == [
        {
            "section": "scorecard",
            "file_name": "diagnostics/scorecard.csv",
            "kind": "csv",
            "tail_rows": 1,
        },
        {
            "section": "summary",
            "file_name": "diagnostics/summary.json",
            "kind": "json",
            "tail_rows": None,
        },
    ]
    assert float(scorecard.loc[0, "value"]) == 0.25
    assert summary == {"status": "ok"}


def test_validation_outputs_reject_duplicate_sections_and_paths():
    """Keep duplicate validation artifacts fail-closed after the module split."""

    outputs = ValidationOutputs()
    outputs.add_json(section="summary", file_name="summary.json", payload={"status": "ok"})

    with pytest.raises(ValueError, match="Duplicate validation section"):
        outputs.add_json(section="summary", file_name="summary_v2.json", payload={})

    with pytest.raises(ValueError, match="Duplicate validation artifact path"):
        outputs.add_json(section="other", file_name="summary.json", payload={})


def test_resolve_validation_split_falls_back_when_three_way_split_is_too_thin():
    """Document the train/validation/test fallback used by validation reports."""

    cfg = load_config("configs/default.yaml")
    cfg.validation_split.mode = "train_validation_test"
    cfg.validation_split.method = "time"
    train_df = pd.DataFrame(
        {
            "start_time_utc": pd.date_range("2025-03-01", periods=45, freq="D").astype(str),
            "home_win": [idx % 2 for idx in range(45)],
        }
    )

    split_plan, train, validation, holdout = resolve_validation_split(train_df, cfg)

    assert split_plan.requested_mode == "train_validation_test"
    assert split_plan.resolved_mode == "train_test"
    assert split_plan.note == "requested_train_validation_test_was_too_thin_so_train_test_was_used"
    assert len(train) == 31
    assert validation.empty
    assert len(holdout) == 14
    assert train["start_time_utc"].is_monotonic_increasing
    assert holdout["start_time_utc"].is_monotonic_increasing


def test_concat_validation_frames_sorts_and_skips_empty_frames():
    """The validation fit frame should remain time-ordered after extraction."""

    early = pd.DataFrame({"start_time_utc": ["2025-03-01"], "home_win": [0]})
    late = pd.DataFrame({"start_time_utc": ["2025-03-03"], "home_win": [1]})
    middle = pd.DataFrame({"start_time_utc": ["2025-03-02"], "home_win": [1]})

    out = concat_validation_frames(late, pd.DataFrame(), early, middle)

    assert out["start_time_utc"].tolist() == ["2025-03-01", "2025-03-02", "2025-03-03"]
