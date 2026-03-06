import json

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.models.glm_logit import GLMLogitModel
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


def test_validation_pipeline_writes_glm_residual_artifacts(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.data.league = "NHL"

    rng = np.random.default_rng(7)
    n = 320
    signal = rng.normal(0.0, 1.0, n)
    counter = rng.normal(0.0, 1.0, n)
    logits = 1.0 * signal - 0.75 * counter
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob)

    train_df = pd.DataFrame(
        {
            "home_win": y,
            "signal": signal,
            "counter": counter,
        }
    )
    glm = GLMLogitModel(c=1.0)
    glm.fit(train_df, feature_columns=["signal", "counter"])

    result = {
        "models": {"glm_logit": glm},
        "train_df": train_df,
        "feature_columns": ["signal", "counter"],
    }
    tasks = [task for task in build_validation_tasks() if task.name == "glm_diagnostics"]

    outputs = run_validation_pipeline(result, cfg, tasks=tasks)

    assert [spec.section for spec in outputs.sections] == [
        "glm_residual_summary",
        "glm_residual_feature_summary",
        "glm_working_residual_bins_linear_predictor",
        "glm_working_residual_bins_features",
        "glm_partial_residual_bins",
    ]

    root = tmp_path / "artifacts" / "validation" / "nhl"
    manifest = json.loads((root / "validation_manifest.json").read_text())
    assert [section["section"] for section in manifest["sections"]] == [
        "glm_residual_summary",
        "glm_residual_feature_summary",
        "glm_working_residual_bins_linear_predictor",
        "glm_working_residual_bins_features",
        "glm_partial_residual_bins",
    ]
    assert (root / "validation_glm_residual_summary.json").exists()
    assert (root / "validation_glm_residual_feature_summary.csv").exists()
    assert (root / "validation_glm_working_residual_bins_linear_predictor.csv").exists()
    assert (root / "validation_glm_working_residual_bins_features.csv").exists()
    assert (root / "validation_glm_partial_residual_bins.csv").exists()

    feature_summary = pd.read_csv(root / "validation_glm_residual_feature_summary.csv")
    for rel_path in feature_summary["working_residual_plot_file"].tolist():
        assert (root / rel_path).exists()
    for rel_path in feature_summary["partial_residual_plot_file"].tolist():
        assert (root / rel_path).exists()
