"""CLI wrappers for ingest and feature commands."""

from __future__ import annotations

from argparse import Namespace

from src.common.config import AppConfig
from src.services import ingest, train


def init_db(cfg: AppConfig, args: Namespace) -> None:
    del args
    ingest.initialize_database(cfg)


def fetch(cfg: AppConfig, args: Namespace) -> None:
    del args
    ingest.fetch_data(cfg)


def refresh_data(cfg: AppConfig, args: Namespace) -> None:
    del args
    ingest.refresh_data(cfg)


def fetch_odds(cfg: AppConfig, args: Namespace) -> None:
    del args
    ingest.fetch_odds(cfg)


def features(cfg: AppConfig, args: Namespace) -> None:
    del args
    ingest.build_features(cfg)


def research_features(cfg: AppConfig, args: Namespace) -> None:
    train.research_features(
        cfg,
        models_arg=getattr(args, "models", None),
        approve_feature_changes=bool(getattr(args, "approve_feature_changes", False)),
    )
