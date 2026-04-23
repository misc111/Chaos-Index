"""Public feature builder entry point over the shared pipeline."""

from __future__ import annotations

from src.features.pipeline import build_features_from_interim_with_strategy
from src.features.strategies.mlb import MlbFeatureStrategy


def build_features_from_interim(interim_dir: str, processed_dir: str, league: str = "MLB"):
    league_code = str(league or "MLB").strip().upper()
    if league_code != "MLB":
        raise ValueError(f"Unsupported league '{league}'. Expected only: MLB.")
    strategy = MlbFeatureStrategy()
    return build_features_from_interim_with_strategy(interim_dir=interim_dir, processed_dir=processed_dir, strategy=strategy)
