"""Backtest service and persistence closeout.

This keeps walk-forward evaluation orchestration out of the CLI while leaving
the model-level evaluation math in the training package.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.common.config import AppConfig
from src.common.logging import get_logger
from src.common.time import utc_now_iso
from src.common.utils import ensure_dir, to_json
from src.services.ingest import latest_snapshot_id
from src.services.train import apply_model_feature_policy, load_features_dataframe, parse_models_arg
from src.storage.db import Database
from src.training.backtest import run_walk_forward_backtest
from src.training.model_feature_research import load_model_feature_map
from src.training.prequential import score_predictions
from src.evaluation.validation_backtest_integrity import run_backtest_integrity_checks

logger = get_logger(__name__)


def run_backtest(cfg: AppConfig, models_arg: str | None = None, approve_feature_changes: bool = False) -> None:
    db = Database(cfg.paths.db_path)
    db.init_schema()
    features_df = load_features_dataframe(cfg.paths.processed_dir)
    approved_feature_columns = apply_model_feature_policy(
        cfg,
        features_df,
        approve_feature_changes=approve_feature_changes,
        run_context="backtest",
    )
    model_feature_columns = load_model_feature_map(cfg.data.league)
    selected_models = parse_models_arg(models_arg)
    bt = run_walk_forward_backtest(
        features_df,
        artifacts_dir=cfg.paths.artifacts_dir,
        bayes_cfg=cfg.bayes.model_dump(),
        n_splits=cfg.modeling.cv_splits,
        selected_models=selected_models,
        selected_feature_columns=approved_feature_columns,
        selected_model_feature_columns=model_feature_columns,
    )

    oof = bt["oof_predictions"]
    if oof.empty:
        logger.warning("Backtest produced no folds.")
        return

    feature_set_rows = db.query("SELECT feature_set_version FROM feature_sets ORDER BY created_at_utc DESC LIMIT 1")
    feature_set_version = feature_set_rows[0]["feature_set_version"] if feature_set_rows else "unknown_feature_set"
    model_run_id = f"backtest_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    snapshot_id = latest_snapshot_id(db)

    pred_rows = []
    for r in oof.itertuples(index=False):
        game_date = pd.to_datetime(r.game_date_utc)
        as_of = (game_date - pd.Timedelta(days=1)).replace(hour=0, minute=0, second=0).isoformat()
        for model_name in [c for c in oof.columns if c not in {"fold", "home_win", "game_id", "game_date_utc"}]:
            prob = float(getattr(r, model_name))
            pred_rows.append(
                (
                    int(r.game_id),
                    as_of,
                    model_name,
                    f"{model_run_id}__{model_name}",
                    feature_set_version,
                    snapshot_id,
                    str(r.game_date_utc),
                    None,
                    None,
                    prob,
                    None,
                    None,
                    None,
                    to_json({"backtest_fold": int(r.fold)}),
                    to_json({"source": "walk_forward_backtest"}),
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

    score_info = score_predictions(db, windows_days=cfg.modeling.rolling_windows_days)

    pred_df = pd.DataFrame(db.query("SELECT * FROM predictions"))
    res_df = pd.DataFrame(db.query("SELECT * FROM results"))
    integrity = run_backtest_integrity_checks(pred_df, res_df, embargo_days=cfg.runtime.embargo_days)
    out_path = Path(cfg.paths.artifacts_dir) / "validation" / "backtest" / "backtest_integrity.json"
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(integrity, indent=2, sort_keys=True))

    logger.info(
        "Backtest complete | oof_rows=%d scored=%d selected_models=%s",
        len(oof),
        score_info.get("n_scored", 0),
        selected_models if selected_models is not None else ["all"],
    )
