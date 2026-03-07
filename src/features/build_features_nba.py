"""Backward-compatible NBA wrapper over the shared feature pipeline."""

from __future__ import annotations

from src.features.pipeline import build_features_from_interim_with_strategy
from src.features.strategies.nba import NbaFeatureStrategy


def build_nba_features_from_interim(interim_dir: str, processed_dir: str):
    return build_features_from_interim_with_strategy(
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        strategy=NbaFeatureStrategy(),
    )
