"""CLI wrappers for ingest and feature commands."""

from __future__ import annotations

from argparse import Namespace

from src.common.config import AppConfig
from src.services import historical_odds_backfill as historical_odds_backfill_service
from src.services import history_import as history_import_service, ingest, train


def init_db(cfg: AppConfig, args: Namespace) -> None:
    """Initialize the SQLite schema for the selected league config."""

    del args
    ingest.initialize_database(cfg)


def fetch(cfg: AppConfig, args: Namespace) -> None:
    """Fetch the latest league data and persist ingest outputs."""

    del args
    ingest.fetch_data(cfg)


def refresh_data(cfg: AppConfig, args: Namespace) -> None:
    """Run the league-scoped refresh-data flow."""

    del args
    ingest.refresh_data(cfg)


def fetch_odds(cfg: AppConfig, args: Namespace) -> None:
    """Fetch the latest standalone odds snapshot."""

    del args
    ingest.fetch_odds(cfg)


def import_history(cfg: AppConfig, args: Namespace) -> None:
    """Import historical data into the research dataset."""

    history_import_service.import_historical_data(
        cfg,
        history_seasons=getattr(args, "history_seasons", None),
        source_manifest=getattr(args, "source_manifest", None),
    )


def backfill_historical_odds(cfg: AppConfig, args: Namespace) -> None:
    """Download historical odds bundles into a regenerable manifest-backed cache."""

    result = historical_odds_backfill_service.backfill_historical_odds_cache(
        cfg,
        start_date=getattr(args, "start_date", None),
        end_date=getattr(args, "end_date", None),
        history_seasons=getattr(args, "history_seasons", None),
        chunk_days=int(getattr(args, "chunk_days", 30)),
    )
    print(
        f"HISTORICAL_ODDS_CACHE::{result.league}::{result.manifest_path}::{result.chunk_count}::{result.fetched_chunks}::{result.skipped_chunks}",
        flush=True,
    )


def features(cfg: AppConfig, args: Namespace) -> None:
    """Build processed features from the current interim snapshot."""

    del args
    ingest.build_features(cfg)


def research_features(cfg: AppConfig, args: Namespace) -> None:
    """Score and optionally promote per-model feature maps."""

    train.research_features(
        cfg,
        models_arg=getattr(args, "models", None),
        approve_feature_changes=bool(getattr(args, "approve_feature_changes", False)),
    )
