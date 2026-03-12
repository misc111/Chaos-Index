"""Smoke command that exercises the full local pipeline."""

from __future__ import annotations

from argparse import Namespace

from src.common.config import AppConfig
from src.common.logging import get_logger
from src.league_registry import canonicalize_league
from src.query.answer import answer_question
from src.services import ingest, train as train_service
from src.storage.db import Database
from src.training.prequential import score_predictions

logger = get_logger(__name__)


def run(cfg: AppConfig, args: Namespace) -> None:
    """Run the reduced-window smoke pipeline for the selected league."""

    del args

    old_hist = cfg.data.history_days
    old_upc = cfg.data.upcoming_days
    cfg.data.history_days = min(60, old_hist)
    cfg.data.upcoming_days = min(7, old_upc)

    ingest.initialize_database(cfg)
    ingest.fetch_data(cfg)
    ingest.build_features(cfg)
    train_service.train_models(cfg, approve_feature_changes=True)
    score_info = score_predictions(Database(cfg.paths.db_path), windows_days=cfg.modeling.rolling_windows_days)

    league = canonicalize_league(cfg.data.league)
    db = Database(cfg.paths.db_path)
    question_by_league = {
        "NHL": "What's the chance the Leafs win the next game?",
        "NBA": "What's the chance the Raptors win the next game?",
        "NCAAM": "What's the chance Duke wins the next game?",
    }
    questions = [
        question_by_league.get(league, "What's the chance the Raptors win the next game?"),
        "Which model has performed best the last 60 days?",
    ]
    logger.info("Smoke scoring info: %s", score_info)
    for question in questions:
        answer, payload = answer_question(db, question, default_league=league)
        logger.info("Smoke query | question=%s answer=%s payload_intent=%s", question, answer, payload.get("intent"))

    cfg.data.history_days = old_hist
    cfg.data.upcoming_days = old_upc
