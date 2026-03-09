"""Training service and persistence closeout for daily model runs.

The training package handles model math. This service owns repository-level
concerns: loading approved features, persisting forecasts, logging run
metadata, and triggering validation outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import AppConfig
from src.common.logging import get_logger
from src.common.time import utc_now_iso
from src.common.utils import to_json
from src.evaluation.validation_pipeline import run_validation_pipeline
from src.services.ingest import latest_snapshot_id
from src.storage.db import Database
from src.storage.tracker import RunTracker
from src.training.feature_policy import apply_feature_policy
from src.training.model_feature_research import load_model_feature_map, research_model_feature_map
from src.training.train import normalize_selected_models, select_feature_columns, train_and_predict

logger = get_logger(__name__)


def parse_models_arg(models_arg: str | None) -> list[str] | None:
    if models_arg is None:
        return None
    tokens = [t.strip() for t in str(models_arg).split(",") if t.strip()]
    return normalize_selected_models(tokens)


def apply_model_feature_policy(
    cfg: AppConfig,
    features_df: pd.DataFrame,
    *,
    approve_feature_changes: bool,
    run_context: str,
) -> list[str]:
    raw_feature_cols = select_feature_columns(features_df)
    policy = apply_feature_policy(
        raw_feature_cols,
        league=cfg.data.league,
        mode=cfg.feature_policy.mode,
        registry_path_template=cfg.feature_policy.registry_path,
        approve_changes=approve_feature_changes,
    )
    logger.info(
        "Feature policy | context=%s mode=%s registry=%s selected=%d added=%d removed=%d candidates_added=%d updated=%s",
        run_context,
        policy.mode,
        policy.registry_path,
        len(policy.approved_feature_columns),
        len(policy.added_features),
        len(policy.removed_features),
        len(policy.candidates_added),
        policy.registry_updated,
    )
    return policy.approved_feature_columns


def load_features_dataframe(processed_dir: str) -> pd.DataFrame:
    feat_path = Path(processed_dir) / "features.parquet"
    if not feat_path.exists():
        feat_path = Path(processed_dir) / "features.csv"
    if not feat_path.exists():
        raise FileNotFoundError("features.parquet not found. Run features first.")

    if feat_path.suffix == ".parquet":
        return pd.read_parquet(feat_path)
    return pd.read_csv(feat_path)


def research_features(cfg: AppConfig, models_arg: str | None = None, approve_feature_changes: bool = False) -> None:
    features_df = load_features_dataframe(cfg.paths.processed_dir)
    feature_cols = select_feature_columns(features_df)
    selected_models = parse_models_arg(models_arg)
    result = research_model_feature_map(
        features_df,
        league=cfg.data.league,
        artifacts_dir=cfg.paths.artifacts_dir,
        feature_columns=feature_cols,
        selected_models=selected_models,
        approve_changes=approve_feature_changes,
    )
    logger.info(
        "Feature research complete | league=%s registry=%s updated=%s report=%s",
        result.league,
        result.registry_path,
        result.registry_updated,
        result.report_path,
    )


def persist_predictions(
    db: Database,
    forecasts: pd.DataFrame,
    per_model_probs: pd.DataFrame,
    model_run_id: str,
    feature_set_version: str,
) -> None:
    snapshot_id = latest_snapshot_id(db)
    pred_rows = []
    forecast_rows = []
    per_model_map = per_model_probs.set_index("game_id").to_dict(orient="index")

    for r in forecasts.itertuples(index=False):
        game_id = int(r.game_id)
        model_probs = per_model_map.get(game_id, {})
        as_of = str(r.as_of_utc)

        for model_name, p in model_probs.items():
            if model_name == "game_id":
                continue
            prob = float(p)
            winner = r.home_team if prob >= 0.5 else r.away_team
            pred_rows.append(
                (
                    game_id,
                    as_of,
                    model_name,
                    f"{model_run_id}__{model_name}",
                    feature_set_version,
                    snapshot_id,
                    r.game_date_utc,
                    r.home_team,
                    r.away_team,
                    prob,
                    winner,
                    None,
                    None,
                    r.uncertainty_flags_json,
                    to_json({"source": "train_upcoming"}),
                )
            )

        ensemble_prob = float(r.ensemble_prob_home_win)
        ensemble_winner = r.home_team if ensemble_prob >= 0.5 else r.away_team
        pred_rows.append(
            (
                game_id,
                as_of,
                "ensemble",
                f"{model_run_id}__ensemble",
                feature_set_version,
                snapshot_id,
                r.game_date_utc,
                r.home_team,
                r.away_team,
                ensemble_prob,
                ensemble_winner,
                r.bayes_ci_low,
                r.bayes_ci_high,
                r.uncertainty_flags_json,
                to_json({"source": "train_upcoming"}),
            )
        )

        forecast_rows.append(
            (
                game_id,
                as_of,
                r.game_date_utc,
                r.home_team,
                r.away_team,
                ensemble_prob,
                ensemble_winner,
                r.per_model_probs_json,
                r.spread_min,
                r.spread_median,
                r.spread_max,
                r.spread_mean,
                r.spread_sd,
                r.spread_iqr,
                r.bayes_ci_low,
                r.bayes_ci_high,
                r.uncertainty_flags_json,
                snapshot_id,
                feature_set_version,
                model_run_id,
            )
        )

    db.executemany(
        """
        INSERT OR REPLACE INTO predictions(
          game_id, as_of_utc, model_name, model_run_id, feature_set_version, snapshot_id,
          game_date_utc, home_team, away_team, prob_home_win, pred_winner, prob_low, prob_high,
          uncertainty_flags_json, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        pred_rows,
    )
    db.executemany(
        """
        INSERT OR REPLACE INTO upcoming_game_forecasts(
          game_id, as_of_utc, game_date_utc, home_team, away_team,
          ensemble_prob_home_win, predicted_winner, per_model_probs_json,
          spread_min, spread_median, spread_max, spread_mean, spread_sd, spread_iqr,
          bayes_ci_low, bayes_ci_high, uncertainty_flags_json, snapshot_id,
          feature_set_version, model_run_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        forecast_rows,
    )


def run_validation_outputs(result: dict[str, Any], cfg: AppConfig) -> None:
    run_validation_pipeline(result, cfg)


def train_models(cfg: AppConfig, models_arg: str | None = None, approve_feature_changes: bool = False) -> None:
    def emit_train_progress(event: dict[str, Any]) -> None:
        print(f"TRAIN_PROGRESS::{json.dumps(event, sort_keys=True)}", flush=True)

    db = Database(cfg.paths.db_path)
    db.init_schema()
    features_df = load_features_dataframe(cfg.paths.processed_dir)
    approved_feature_columns = apply_model_feature_policy(
        cfg,
        features_df,
        approve_feature_changes=approve_feature_changes,
        run_context="train",
    )
    model_feature_columns = load_model_feature_map(cfg.data.league)
    feature_set_rows = db.query("SELECT feature_set_version FROM feature_sets ORDER BY created_at_utc DESC LIMIT 1")
    feature_set_version = feature_set_rows[0]["feature_set_version"] if feature_set_rows else "unknown_feature_set"
    selected_models = parse_models_arg(models_arg)

    tracker = RunTracker(cfg.paths.artifacts_dir)
    run_id = tracker.start_run(
        "train",
        {
            "feature_set_version": feature_set_version,
            "selected_models": selected_models if selected_models is not None else ["all"],
        },
    )
    emit_train_progress(
        {
            "kind": "pipeline",
            "stage": "train_command",
            "status": "started",
            "message": "Starting cmd_train",
            "feature_set_version": feature_set_version,
            "selected_models": selected_models if selected_models is not None else ["all"],
        }
    )
    result = train_and_predict(
        features_df=features_df,
        feature_set_version=feature_set_version,
        artifacts_dir=cfg.paths.artifacts_dir,
        bayes_cfg=cfg.bayes.model_dump(),
        selected_models=selected_models,
        progress_callback=emit_train_progress,
        selected_feature_columns=approved_feature_columns,
        selected_model_feature_columns=model_feature_columns,
        league=cfg.data.league,
    )
    tracker.log_metrics(
        run_id,
        {
            "n_upcoming": int(len(result["forecasts"])),
            "stack_ready": int(result["stack_ready"]),
            "n_selected_models": int(len(result["run_payload"].get("selected_models", []))),
        },
    )
    tracker.log_artifact(run_id, "train_metrics", result["train_metrics"])

    persist_predictions(
        db,
        forecasts=result["forecasts"],
        per_model_probs=result["upcoming_model_probs"],
        model_run_id=result["model_run_id"],
        feature_set_version=feature_set_version,
    )

    run_rows = []
    for model_name in [c for c in result["upcoming_model_probs"].columns if c != "game_id"] + ["ensemble"]:
        run_rows.append(
            (
                f"{result['model_run_id']}__{model_name}",
                model_name,
                "daily_train",
                utc_now_iso(),
                latest_snapshot_id(db),
                feature_set_version,
                to_json({"weights": result["weights"]}),
                to_json(result["train_metrics"].get(model_name, {})),
                result["model_dir"],
                result["model_run_id"],
            )
        )
    db.executemany(
        """
        INSERT OR REPLACE INTO model_runs(
          model_run_id, model_name, run_type, created_at_utc, snapshot_id,
          feature_set_version, params_json, metrics_json, artifact_path, model_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        run_rows,
    )

    run_validation_outputs(result, cfg)
    tracker.end_run(run_id)
    emit_train_progress(
        {
            "kind": "pipeline",
            "stage": "train_command",
            "status": "completed",
            "message": "Completed cmd_train",
            "model_run_id": result["model_run_id"],
        }
    )
    logger.info(
        "Train complete | model_run_id=%s upcoming=%d selected_models=%s",
        result["model_run_id"],
        len(result["forecasts"]),
        result["run_payload"].get("selected_models"),
    )
