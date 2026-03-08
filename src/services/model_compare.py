"""Standalone research comparison service for candidate predictive models."""

from __future__ import annotations

from src.common.config import AppConfig
from src.common.logging import get_logger
from src.research.model_comparison import run_candidate_model_comparison

logger = get_logger(__name__)


def compare_candidate_models(
    cfg: AppConfig,
    *,
    report_slug: str | None = None,
    bootstrap_samples: int = 1000,
):
    result = run_candidate_model_comparison(
        cfg,
        report_slug=report_slug,
        bootstrap_samples=bootstrap_samples,
    )
    logger.info(
        "Candidate model comparison complete | league=%s recommendation=%s report=%s",
        result.league,
        result.recommendation_model,
        result.report_path,
    )
    return result
