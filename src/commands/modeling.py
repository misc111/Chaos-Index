"""CLI wrappers for training and backtest flows."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd

from src.common.config import AppConfig
from src.common.logging import get_logger
from src.common.utils import ensure_dir
from src.services import backtest as backtest_service
from src.services import ingest, model_compare as model_compare_service, train as train_service
from src.services import validate as validate_service
from src.storage.db import Database
from src.training.prequential import score_predictions

logger = get_logger(__name__)


def _apply_validation_split_overrides(cfg: AppConfig, args: Namespace) -> None:
    split_mode = getattr(args, "validation_split_mode", None)
    split_method = getattr(args, "validation_split_method", None)
    split_seed = getattr(args, "validation_split_seed", None)
    if split_mode is not None:
        cfg.validation_split.mode = split_mode
    if split_method is not None:
        cfg.validation_split.method = split_method
    if split_seed is not None:
        cfg.validation_split.random_seed = int(split_seed)


def train(cfg: AppConfig, args: Namespace) -> None:
    _apply_validation_split_overrides(cfg, args)
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


def validate(cfg: AppConfig, args: Namespace) -> None:
    _apply_validation_split_overrides(cfg, args)
    validate_service.run_saved_validation(
        cfg,
        models_arg=getattr(args, "models", None),
        model_run_id=getattr(args, "model_run_id", None),
    )


def compare_candidates(cfg: AppConfig, args: Namespace) -> None:
    result = model_compare_service.compare_candidate_models(
        cfg,
        report_slug=getattr(args, "report_slug", None),
        bootstrap_samples=int(getattr(args, "bootstrap_samples", 1000)),
    )
    print(
        f"CANDIDATE_MODEL_COMPARISON::{result.league}::{result.recommendation_model}::{result.report_path}",
        flush=True,
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
