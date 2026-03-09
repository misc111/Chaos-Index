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
    candidate_models: list[str] | None = None,
    feature_pool: str = "full_screened",
    feature_map_model: str = "glm_ridge",
):
    result = run_candidate_model_comparison(
        cfg,
        report_slug=report_slug,
        bootstrap_samples=bootstrap_samples,
        candidate_models=candidate_models,
        feature_pool=feature_pool,
        feature_map_model=feature_map_model,
    )
    logger.info(
        "Candidate model comparison complete | league=%s recommendation=%s report=%s",
        result.league,
        result.recommendation_model,
        result.report_path,
    )
    return result
