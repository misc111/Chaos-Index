import json

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.models.glm_ridge import GLMRidgeModel
from src.storage.db import Database
from src.training.lasso_credibility import resolve_lasso_credibility_model_name
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
        "glm_raw_residual_rows",
        "glm_residual_vs_fitted_bins",
        "glm_grouped_residual_summary",
        "glm_grouped_residual_metadata",
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
        "glm_raw_residual_rows",
        "glm_residual_vs_fitted_bins",
        "glm_grouped_residual_summary",
        "glm_grouped_residual_metadata",
    ]
    glm_root = root / "glm" / "residuals"
    assert (glm_root / "validation_glm_residual_summary.json").exists()
    assert (glm_root / "validation_glm_residual_feature_summary.csv").exists()
    assert (glm_root / "validation_glm_working_residual_bins_linear_predictor.csv").exists()
    assert (glm_root / "validation_glm_working_residual_bins_features.csv").exists()
    assert (glm_root / "validation_glm_working_residual_bins_weight.csv").exists()
    assert (glm_root / "validation_glm_partial_residual_bins.csv").exists()
    assert (glm_root / "validation_glm_raw_residual_rows.csv").exists()
    assert (glm_root / "validation_glm_residual_vs_fitted_bins.csv").exists()
    assert (glm_root / "validation_glm_grouped_residual_summary.csv").exists()
    assert (glm_root / "validation_glm_grouped_residual_metadata.json").exists()

    feature_summary = pd.read_csv(glm_root / "validation_glm_residual_feature_summary.csv")
    for rel_path in feature_summary["working_residual_plot_file"].tolist():
        assert (root / rel_path).exists()
    for rel_path in feature_summary["partial_residual_plot_file"].tolist():
        assert (root / rel_path).exists()


def test_validation_pipeline_writes_market_truth_artifacts_for_mlb(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.db_path = str(tmp_path / "mlb_market_truth.db")
    cfg.data.league = "MLB"

    n = 80
    dates = pd.date_range("2026-04-01", periods=n, freq="D")
    signal = np.linspace(-1.5, 1.5, n)
    prob = 1.0 / (1.0 + np.exp(-signal))
    y = (prob >= 0.5).astype(int)
    train_df = pd.DataFrame(
        {
            "game_id": np.arange(10_000, 10_000 + n),
            "season": [2026] * n,
            "start_time_utc": (dates + pd.Timedelta(hours=23)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "game_date_utc": dates.date.astype(str),
            "home_team": ["CHC"] * n,
            "away_team": ["LAD"] * n,
            "home_win": y,
            "signal": signal,
        }
    )

    db = Database(cfg.paths.db_path)
    db.init_schema()
    game_rows = [
        (
            int(row.game_id),
            2026,
            row.game_date_utc,
            row.start_time_utc,
            "Final",
            row.home_team,
            row.away_team,
            int(row.home_win),
            1,
            row.start_time_utc,
        )
        for row in train_df.itertuples(index=False)
    ]
    db.executemany(
        """
        INSERT INTO games(
          game_id, season, game_date_utc, start_time_utc, game_state,
          home_team, away_team, home_win, status_final, as_of_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        game_rows,
    )

    snapshot_rows = []
    line_rows = []
    for row in train_df.itertuples(index=False):
        start_ts = pd.Timestamp(row.start_time_utc)
        current_as_of = (start_ts - pd.Timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        close_as_of = (start_ts - pd.Timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for label, as_of, home_price, away_price in (
            ("current", current_as_of, -110, 100),
            ("close", close_as_of, -135 if row.home_win else 120, 115 if row.home_win else -140),
        ):
            snapshot_id = f"{row.game_id}_{label}"
            snapshot_rows.append((snapshot_id, "unit_test", "MLB", as_of, "h2h", 1, 2, "{}"))
            line_rows.extend(
                [
                    (
                        snapshot_id,
                        "MLB",
                        int(row.game_id),
                        "baseball_mlb",
                        str(row.game_id),
                        row.start_time_utc,
                        row.home_team,
                        row.away_team,
                        "book-a",
                        as_of,
                        "h2h",
                        row.home_team,
                        "home",
                        row.home_team,
                        float(home_price),
                        as_of,
                    ),
                    (
                        snapshot_id,
                        "MLB",
                        int(row.game_id),
                        "baseball_mlb",
                        str(row.game_id),
                        row.start_time_utc,
                        row.home_team,
                        row.away_team,
                        "book-a",
                        as_of,
                        "h2h",
                        row.away_team,
                        "away",
                        row.away_team,
                        float(away_price),
                        as_of,
                    ),
                ]
            )
    db.executemany(
        """
        INSERT INTO odds_snapshots(
          odds_snapshot_id, source, league, as_of_utc, markets,
          event_count, row_count, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        snapshot_rows,
    )
    db.executemany(
        """
        INSERT INTO odds_market_lines(
          odds_snapshot_id, league, game_id, sport_key, odds_event_id,
          commence_time_utc, home_team, away_team, bookmaker_key,
          bookmaker_last_update_utc, market_key, outcome_name, outcome_side,
          outcome_team, outcome_price, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        line_rows,
    )

    result = {
        "models": {},
        "train_df": train_df,
        "feature_columns": ["signal"],
        "run_payload": {"selected_models": ["glm_ridge"], "model_run_id": "unit_market_truth_validation"},
    }
    tasks = [task for task in build_validation_tasks() if task.name == "market_truth"]

    outputs = run_validation_pipeline(result, cfg, tasks=tasks)

    assert [spec.section for spec in outputs.sections] == [
        "market_truth_summary",
        "market_truth_by_model",
        "market_truth_calibration",
    ]
    root = tmp_path / "artifacts" / "validation" / "mlb" / "market_truth"
    summary = json.loads((root / "validation_market_truth_summary.json").read_text())
    assert summary["status"] == "ok"
    assert summary["market_truth_classification"] == "betting_overlay"
    assert summary["models"][0]["model_name"] == "glm_ridge"
    assert "log_loss_gain_vs_closing_market" in summary["models"][0]

    contract = json.loads((tmp_path / "artifacts" / "validation" / "mlb" / "validation_outputs_contract.json").read_text())
    record = contract["validation_outputs"][0]
    assert record["task_name"] == "market_truth"
    assert record["family"] == "market_truth"
    assert record["applicability"] == "applicable"


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

    validation_root = tmp_path / "artifacts" / "validation" / "nba"
    classification_root = validation_root / "diagnostics" / "classification"
    performance_root = tmp_path / "artifacts" / "plots" / "nba" / "glm" / "performance"
    assert (classification_root / "validation_logit_classification_summary.json").exists()
    assert (performance_root / "quantile.png").exists()
    assert (performance_root / "actual_vs_predicted.png").exists()
    assert (performance_root / "lift.png").exists()
    assert (performance_root / "lorenz.png").exists()
    assert (performance_root / "roc.png").exists()

    contract_payload = json.loads((validation_root / "validation_outputs_contract.json").read_text())
    records = {record["task_name"]: record for record in contract_payload["validation_outputs"]}
    summary = records["classification_curves"]["summary"]
    assert records["classification_curves"]["applicability"] == "applicable"
    assert summary["metric_applicability"]["lift_curve"] == "applicable"
    assert summary["model_family_applicability"]["status"] == "applicable"
    assert summary["model_family_applicability"]["model_family"] == "linear"
    assert summary["model_family_applicability"]["model_lane"] == "core"
    assert summary["lift_summary"]["top_vs_bottom_actual_lift_diff"] > 0


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
    assert metadata["league"] == "NBA"
    assert metadata["model_run_id"] == "run_test_archive"
    assert metadata["selected_models"] == ["glm_ridge"]
    assert metadata["feature_set_version"] == "fset_test_archive"
    assert metadata["artifact_counts"]["performance_files"] == 5
    assert metadata["artifact_counts"]["total_files"] >= metadata["artifact_counts"]["performance_files"]
    assert "validation_manifest.json" in metadata["artifact_groups"][0]["files"]
    for group in metadata["artifact_groups"]:
        rel_dir = str(group["relative_dir"])
        for file_name in group["files"]:
            rebuilt = archive_root / file_name if rel_dir == "." else archive_root / rel_dir / file_name
            assert rebuilt.exists()


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


def test_validation_context_routes_glm_vanilla_as_real_primary_model(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.data.league = "MLB"

    rng = np.random.default_rng(111)
    n = 180
    signal = rng.normal(0.0, 1.0, n)
    counter = rng.normal(0.0, 1.0, n)
    logits = 0.9 * signal - 0.4 * counter
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

    ctx = ValidationContext.from_result(
        {
            "models": {},
            "train_df": train_df,
            "feature_columns": ["signal", "counter"],
            "run_payload": {
                "selected_models": ["glm_vanilla"],
                "glm_primary_model": "glm_vanilla",
                "model_feature_columns": {"glm_vanilla": ["signal", "counter"]},
            },
        },
        cfg,
    )

    assert ctx.glm is not None
    assert getattr(ctx.glm, "model_name", "") == "glm_vanilla"
    assert ctx.glm_validation_metadata.model_key == "glm_vanilla"
    assert ctx.primary_model_resolution["resolved_primary_model"] == "glm_vanilla"


def test_validation_context_routes_lasso_credibility_as_real_primary_model(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.data.league = "MLB"

    rng = np.random.default_rng(114)
    n = 220
    signal = rng.normal(0.0, 1.0, n)
    counter = rng.normal(0.0, 1.0, n)
    market_offset_logit = 0.6 * signal + rng.normal(0.0, 0.25, n)
    logits = market_offset_logit + 0.7 * signal - 0.5 * counter
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob)
    model_name = resolve_lasso_credibility_model_name("market")
    train_df = pd.DataFrame(
        {
            "start_time_utc": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "game_date_utc": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "home_win": y,
            "signal": signal,
            "counter": counter,
            "market_offset_logit": market_offset_logit,
        }
    )

    ctx = ValidationContext.from_result(
        {
            "models": {},
            "train_df": train_df,
            "feature_columns": ["signal", "counter", "market_offset_logit"],
            "run_payload": {
                "selected_models": [model_name],
                "glm_primary_model": model_name,
                "model_feature_columns": {model_name: ["signal", "counter"]},
                "credibility_tuning_by_model": {model_name: {"best_lambda": 1.0, "best_c": 1.0}},
            },
        },
        cfg,
    )

    assert ctx.glm is not None
    assert getattr(ctx.glm, "model_name", "") == model_name
    assert ctx.glm_validation_metadata.model_key == model_name
    assert ctx.glm_validation_metadata.credibility is not None
    assert ctx.primary_model_resolution["resolved_primary_model"] == model_name


def test_validation_task_records_report_pending_for_unsupported_extension_coverage(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.data.league = "MLB"

    rng = np.random.default_rng(515)
    n = 120
    train_df = pd.DataFrame(
        {
            "start_time_utc": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "game_date_utc": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "home_win": rng.integers(0, 2, n),
            "signal": rng.normal(0.0, 1.0, n),
            "counter": rng.normal(0.0, 1.0, n),
            "home_team": [f"H{i % 5}" for i in range(n)],
            "away_team": [f"A{i % 5}" for i in range(n)],
        }
    )

    tasks = [task for task in build_validation_tasks() if task.name in {"stability", "influence"}]
    run_validation_pipeline(
        {
            "models": {},
            "train_df": train_df,
            "feature_columns": ["signal", "counter"],
            "run_payload": {
                "selected_models": ["glmm_logit"],
                "glm_primary_model": "glmm_logit",
                "model_feature_columns": {"glmm_logit": ["signal", "counter"]},
            },
        },
        cfg,
        tasks=tasks,
    )

    contract = json.loads((tmp_path / "artifacts" / "validation" / "mlb" / "validation_outputs_contract.json").read_text())
    records = {row["task_name"]: row for row in contract["validation_outputs"]}
    assert records["stability"]["applicability"] == "not_applicable"
    assert records["influence"]["applicability"] == "not_applicable"
    assert records["stability"]["summary"]["applicability_split"]["contract_level"] == "not_applicable"
    assert records["stability"]["summary"]["model_family_applicability"]["status"] in {"pending", "not_applicable"}
