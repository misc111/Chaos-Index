"""CLI wrappers for training and backtest flows."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd

from src.common.config import AppConfig
from src.common.logging import get_logger
from src.common.utils import ensure_dir
from src.services import backtest as backtest_service
from src.services import ingest, train as train_service
from src.storage.db import Database
from src.training.prequential import score_predictions

logger = get_logger(__name__)


def train(cfg: AppConfig, args: Namespace) -> None:
    train_service.train_models(
        cfg,
        models_arg=getattr(args, "models", None),
        approve_feature_changes=bool(getattr(args, "approve_feature_changes", False)),
    )


def backtest(cfg: AppConfig, args: Namespace) -> None:
    backtest_service.run_backtest(
        cfg,
        models_arg=getattr(args, "models", None),
        approve_feature_changes=bool(getattr(args, "approve_feature_changes", False)),
    )


def run_daily(cfg: AppConfig, args: Namespace) -> None:
    ingest.fetch_data(cfg)
    ingest.build_features(cfg)
    if cfg.runtime.retrain_daily:
        train(cfg, args)

    db = Database(cfg.paths.db_path)
    score_info = score_predictions(db, windows_days=cfg.modeling.rolling_windows_days)

    perf = pd.DataFrame(db.query("SELECT * FROM performance_aggregates ORDER BY as_of_utc DESC"))
    if not perf.empty:
        out = Path(cfg.paths.artifacts_dir) / "reports" / "performance_aggregates_latest.csv"
        ensure_dir(out.parent)
        perf.to_csv(out, index=False)

    logger.info("Daily run complete | scored=%s", score_info)
