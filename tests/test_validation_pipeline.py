import json

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.models.glm_ridge import GLMRidgeModel
from src.evaluation.validation_pipeline import (
    _validation_split,
    ValidationContext,
    ValidationOutputs,
    ValidationTask,
    build_validation_tasks,
    run_validation_pipeline,
)


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


def test_validation_split_defaults_to_time_based_70_30_train_test(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.data.league = "NBA"

    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    train_df = pd.DataFrame(
        {
            "game_id": np.arange(n),
            "start_time_utc": dates.astype(str),
            "game_date_utc": dates.date.astype(str),
            "home_win": ([0, 1] * 50),
            "signal": np.linspace(-1.0, 1.0, n),
        }
    )

    plan, tr, va, te = _validation_split(train_df, cfg)

    assert plan.requested_mode == "train_test"
    assert plan.resolved_mode == "train_test"
    assert plan.requested_method == "time"
    assert plan.resolved_method == "time"
    assert plan.train_fraction == 0.7
    assert plan.validation_fraction == 0.0
    assert plan.holdout_fraction == 0.3
    assert plan.random_seed is None
    assert len(tr) == 70
    assert len(va) == 0
    assert len(te) == 30
    assert tr["game_id"].tolist() == list(range(70))
    assert te["game_id"].tolist() == list(range(70, 100))


def test_validation_split_supports_reproducible_random_40_30_30():
    cfg = load_config("configs/default.yaml")
    cfg.data.league = "NHL"
    cfg.validation_split.mode = "train_validation_test"
    cfg.validation_split.method = "random"
    cfg.validation_split.random_seed = 17

    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    train_df = pd.DataFrame(
        {
            "game_id": np.arange(n),
            "start_time_utc": dates.astype(str),
            "game_date_utc": dates.date.astype(str),
            "home_win": ([0, 1] * 50),
            "signal": np.linspace(-1.0, 1.0, n),
        }
    )

    plan_a, tr_a, va_a, te_a = _validation_split(train_df, cfg)
    plan_b, tr_b, va_b, te_b = _validation_split(train_df, cfg)

    assert plan_a.requested_mode == "train_validation_test"
    assert plan_a.resolved_mode == "train_validation_test"
    assert plan_a.requested_method == "random"
    assert plan_a.resolved_method == "random"
    assert plan_a.train_fraction == 0.4
    assert plan_a.validation_fraction == 0.3
    assert plan_a.holdout_fraction == 0.3
    assert plan_a.random_seed == 17
    assert len(tr_a) == 40
    assert len(va_a) == 30
    assert len(te_a) == 30
    assert set(tr_a["game_id"]).isdisjoint(set(va_a["game_id"]))
    assert set(tr_a["game_id"]).isdisjoint(set(te_a["game_id"]))
    assert set(va_a["game_id"]).isdisjoint(set(te_a["game_id"]))
    assert set(pd.concat([tr_a["game_id"], va_a["game_id"], te_a["game_id"]])) == set(range(n))
    assert tr_a["game_id"].tolist() == tr_b["game_id"].tolist()
    assert va_a["game_id"].tolist() == va_b["game_id"].tolist()
    assert te_a["game_id"].tolist() == te_b["game_id"].tolist()
    assert tr_a["game_id"].tolist() != list(range(40))


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
    glm = GLMRidgeModel(c=1.0)
    glm.fit(train_df, feature_columns=["signal", "counter"])

    result = {
        "models": {"glm_ridge": glm},
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
        "glm_working_residual_bins_weight",
        "glm_partial_residual_bins",
    ]

    root = tmp_path / "artifacts" / "validation" / "nhl"
    manifest = json.loads((root / "validation_manifest.json").read_text())
    assert [section["section"] for section in manifest["sections"]] == [
        "glm_residual_summary",
        "glm_residual_feature_summary",
        "glm_working_residual_bins_linear_predictor",
        "glm_working_residual_bins_features",
        "glm_working_residual_bins_weight",
        "glm_partial_residual_bins",
    ]
    glm_root = root / "glm" / "residuals"
    assert (glm_root / "validation_glm_residual_summary.json").exists()
    assert (glm_root / "validation_glm_residual_feature_summary.csv").exists()
    assert (glm_root / "validation_glm_working_residual_bins_linear_predictor.csv").exists()
    assert (glm_root / "validation_glm_working_residual_bins_features.csv").exists()
    assert (glm_root / "validation_glm_working_residual_bins_weight.csv").exists()
    assert (glm_root / "validation_glm_partial_residual_bins.csv").exists()

    feature_summary = pd.read_csv(glm_root / "validation_glm_residual_feature_summary.csv")
    for rel_path in feature_summary["working_residual_plot_file"].tolist():
        assert (root / rel_path).exists()
    for rel_path in feature_summary["partial_residual_plot_file"].tolist():
        assert (root / rel_path).exists()


def test_validation_pipeline_runs_significance_stability_and_influence_for_nba(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.data.league = "NBA"

    rng = np.random.default_rng(19)
    n = 260
    availability_absence_diff = rng.normal(0.0, 1.0, n)
    shot_margin = rng.normal(0.0, 1.0, n)
    discipline_free_throw_pressure_diff = rng.normal(0.0, 1.0, n)
    travel_diff = rng.normal(0.0, 1.0, n)
    arena_altitude_diff = rng.normal(0.0, 1.0, n)
    logits = (
        1.0 * availability_absence_diff
        + 0.8 * shot_margin
        - 0.5 * discipline_free_throw_pressure_diff
        - 0.4 * travel_diff
        + 0.3 * arena_altitude_diff
    )
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob)

    feature_cols = [
        "availability_absence_diff",
        "shot_margin",
        "discipline_free_throw_pressure_diff",
        "travel_diff",
        "arena_altitude_diff",
    ]
    train_df = pd.DataFrame(
        {
            "start_time_utc": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "game_date_utc": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "home_win": y,
            "availability_absence_diff": availability_absence_diff,
            "shot_margin": shot_margin,
            "discipline_free_throw_pressure_diff": discipline_free_throw_pressure_diff,
            "travel_diff": travel_diff,
            "arena_altitude_diff": arena_altitude_diff,
        }
    )
    glm = GLMRidgeModel(c=1.0)
    glm.fit(train_df, feature_columns=feature_cols)

    result = {
        "models": {"glm_ridge": glm},
        "train_df": train_df,
        "feature_columns": feature_cols,
        "run_payload": {
            "selected_models": ["glm_ridge"],
            "model_feature_columns": {"glm_ridge": feature_cols},
        },
    }
    tasks = [task for task in build_validation_tasks() if task.name in {"split_summary", "significance", "stability", "influence"}]

    outputs = run_validation_pipeline(result, cfg, tasks=tasks)
    sections = [spec.section for spec in outputs.sections]

    assert "split_summary" in sections
    assert "significance" in sections
    assert "information_criteria_summary" in sections
    assert "information_criteria_candidates" in sections
    assert "cv_summary" in sections
    assert "bootstrap_summary" in sections
    assert "influence_summary" in sections

    root = tmp_path / "artifacts" / "validation" / "nba"
    significance_root = root / "diagnostics" / "significance"
    stability_root = root / "diagnostics" / "stability"
    influence_root = root / "diagnostics" / "influence"
    split_root = root / "split"
    assert (split_root / "validation_split_summary.json").exists()
    split_summary = json.loads((split_root / "validation_split_summary.json").read_text())
    assert split_summary["split_mode"] == "train_test"
    assert split_summary["split_method"] == "time"
    assert split_summary["requested_split_mode"] == "train_test"
    assert split_summary["requested_split_method"] == "time"
    assert split_summary["train_fraction"] == 0.7
    assert split_summary["validation_fraction"] == 0.0
    assert split_summary["holdout_fraction"] == 0.3
    assert (significance_root / "validation_significance.csv").exists()
    assert (significance_root / "validation_information_criteria_summary.json").exists()
    assert (significance_root / "validation_information_criteria_candidates.csv").exists()
    assert (stability_root / "validation_cv_summary.json").exists()
    assert (stability_root / "validation_bootstrap_summary.json").exists()
    assert (influence_root / "validation_influence_summary.json").exists()


def test_validation_pipeline_writes_classification_curves_to_structured_plot_dir(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.data.league = "NBA"

    rng = np.random.default_rng(23)
    n = 220
    signal = rng.normal(0.0, 1.0, n)
    counter = rng.normal(0.0, 1.0, n)
    logits = 1.1 * signal - 0.7 * counter
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob)

    train_df = pd.DataFrame(
        {
            "start_time_utc": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "game_date_utc": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "home_win": y,
            "signal": signal,
            "counter": counter,
        }
    )
    glm = GLMRidgeModel(c=1.0)
    glm.fit(train_df, feature_columns=["signal", "counter"])

    result = {
        "models": {"glm_ridge": glm},
        "train_df": train_df,
        "feature_columns": ["signal", "counter"],
        "run_payload": {
            "selected_models": ["glm_ridge"],
            "model_feature_columns": {"glm_ridge": ["signal", "counter"]},
        },
    }
    tasks = [task for task in build_validation_tasks() if task.name == "classification_curves"]

    run_validation_pipeline(result, cfg, tasks=tasks)

    performance_root = tmp_path / "artifacts" / "plots" / "nba" / "glm" / "performance"
    assert (performance_root / "quantile.png").exists()
    assert (performance_root / "actual_vs_predicted.png").exists()
    assert (performance_root / "lift.png").exists()
    assert (performance_root / "lorenz.png").exists()
    assert (performance_root / "roc.png").exists()


def test_validation_pipeline_archives_clean_validation_run_snapshot(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.data.league = "NBA"

    rng = np.random.default_rng(29)
    n = 240
    signal = rng.normal(0.0, 1.0, n)
    counter = rng.normal(0.0, 1.0, n)
    logits = 1.0 * signal - 0.8 * counter
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob)

    train_df = pd.DataFrame(
        {
            "start_time_utc": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "game_date_utc": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "home_win": y,
            "signal": signal,
            "counter": counter,
        }
    )
    glm = GLMRidgeModel(c=1.0)
    glm.fit(train_df, feature_columns=["signal", "counter"])

    validation_root = tmp_path / "artifacts" / "validation" / "nba"
    performance_root = tmp_path / "artifacts" / "plots" / "nba" / "glm" / "performance"
    validation_root.mkdir(parents=True, exist_ok=True)
    performance_root.mkdir(parents=True, exist_ok=True)
    (validation_root / "stale.txt").write_text("stale")
    (performance_root / "stale.png").write_text("stale")

    result = {
        "models": {"glm_ridge": glm},
        "train_df": train_df,
        "feature_columns": ["signal", "counter"],
        "run_payload": {
            "model_run_id": "run_test_archive",
            "feature_set_version": "fset_test_archive",
            "selected_models": ["glm_ridge"],
            "model_feature_columns": {"glm_ridge": ["signal", "counter"]},
        },
    }
    tasks = [task for task in build_validation_tasks() if task.name in {"glm_diagnostics", "classification_curves"}]

    run_validation_pipeline(result, cfg, tasks=tasks)

    assert not (validation_root / "stale.txt").exists()
    assert not (performance_root / "stale.png").exists()
    assert (validation_root / "validation_run_metadata.json").exists()
    assert (performance_root / "roc.png").exists()

    archive_roots = list((tmp_path / "artifacts" / "validation-runs" / "nba").glob("*/*"))
    assert len(archive_roots) == 1
    archive_root = archive_roots[0]

    assert (archive_root / "validation_manifest.json").exists()
    assert (archive_root / "validation_run_metadata.json").exists()
    assert (archive_root / "glm" / "residuals" / "plots" / "glm_validation_deviance_residuals.png").exists()
    assert (archive_root / "performance" / "roc.png").exists()

    metadata = json.loads((archive_root / "validation_run_metadata.json").read_text())
    assert metadata["model_run_id"] == "run_test_archive"
    assert metadata["feature_set_version"] == "fset_test_archive"
    assert metadata["artifact_counts"]["performance_files"] == 5
    assert "validation_manifest.json" in metadata["artifact_groups"][0]["files"]


def test_validation_context_refits_holdout_models_instead_of_using_production_model(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.data.league = "NHL"
    cfg.modeling.cv_splits = 4
    cfg.validation_split.mode = "train_validation_test"

    rng = np.random.default_rng(17)
    n = 260
    start = pd.date_range("2025-01-01", periods=n, freq="D")
    signal = rng.normal(0.0, 1.0, n)
    counter = rng.normal(0.0, 1.0, n)
    logits = 1.3 * signal - 0.8 * counter
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob)

    train_df = pd.DataFrame(
        {
            "start_time_utc": start.astype(str),
            "game_date_utc": start.date.astype(str),
            "home_win": y,
            "signal": signal,
            "counter": counter,
        }
    )

    class ConstantProductionModel:
        feature_columns = ["signal", "counter"]

        def predict_proba(self, df):
            return np.full(len(df), 0.99)

    production_glm = ConstantProductionModel()
    result = {
        "models": {"glm_ridge": production_glm},
        "train_df": train_df,
        "feature_columns": ["signal", "counter"],
        "run_payload": {
            "selected_models": ["glm_ridge"],
            "glm_feature_columns": ["signal", "counter"],
            "model_feature_columns": {"glm_ridge": ["signal", "counter"]},
        },
    }

    ctx = ValidationContext.from_result(result, cfg)

    assert not ctx.va.empty
    assert ctx.glm is not production_glm
    assert max(pd.to_datetime(ctx.tr["start_time_utc"])) < min(pd.to_datetime(ctx.va["start_time_utc"]))
    assert not np.allclose(ctx.glm.predict_proba(ctx.va), 0.99)
