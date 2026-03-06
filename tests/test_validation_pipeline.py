import json

import pandas as pd

from src.common.config import load_config
from src.evaluation.validation_pipeline import ValidationOutputs, ValidationTask, build_validation_tasks, run_validation_pipeline


def test_validation_pipeline_supports_custom_extension_tasks(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.data.league = "NHL"

    train_df = pd.DataFrame(
        {
            "game_date_utc": pd.date_range("2025-01-01", periods=6, freq="D").astype(str),
            "home_win": [0, 1, 0, 1, 0, 1],
            "signal": [0.1, 0.8, 0.2, 0.7, 0.3, 0.9],
        }
    )
    result = {
        "models": {},
        "train_df": train_df,
        "feature_columns": ["signal"],
    }

    def custom_task(_ctx):
        out = ValidationOutputs()
        out.add_csv(
            section="nonlinearity_probe_table",
            file_name="validation_nonlinearity_probe.csv",
            rows=pd.DataFrame([{"feature": "signal", "score": 0.42}]),
        )
        out.add_json(
            section="nonlinearity_probe_summary",
            file_name="validation_nonlinearity_probe_summary.json",
            payload={"status": "ok", "headline": "extension task executed"},
        )
        return out

    outputs = run_validation_pipeline(
        result,
        cfg,
        tasks=[ValidationTask(name="nonlinearity_probe", runner=custom_task)],
    )

    assert [spec.section for spec in outputs.sections] == [
        "nonlinearity_probe_table",
        "nonlinearity_probe_summary",
    ]

    manifest = json.loads((tmp_path / "artifacts" / "validation" / "nhl" / "validation_manifest.json").read_text())
    assert [section["section"] for section in manifest["sections"]] == [
        "nonlinearity_probe_table",
        "nonlinearity_probe_summary",
    ]
    assert (tmp_path / "artifacts" / "validation" / "nhl" / "validation_nonlinearity_probe.csv").exists()
    assert (tmp_path / "artifacts" / "validation" / "nhl" / "validation_nonlinearity_probe_summary.json").exists()


def test_validation_task_registry_allows_extension_without_touching_defaults():
    extra = ValidationTask(name="nonlinearity_probe", runner=lambda _ctx: ValidationOutputs())
    task_names = [task.name for task in build_validation_tasks(extra_tasks=[extra])]

    assert "collinearity" in task_names
    assert "significance" in task_names
    assert "fragility" in task_names
    assert task_names[-1] == "nonlinearity_probe"
