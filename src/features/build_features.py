"""Public feature builder entry point over the shared pipeline."""

from __future__ import annotations

from src.features.pipeline import build_features_from_interim_with_strategy
from src.features.strategies.mlb import MlbFeatureStrategy
from src.features.strategies.nba import NbaFeatureStrategy
from src.features.strategies.nhl import NhlFeatureStrategy


def build_features_from_interim(interim_dir: str, processed_dir: str, league: str = "MLB"):
    league_code = str(league or "MLB").strip().upper()
    if league_code == "MLB":
        strategy = MlbFeatureStrategy()
    elif league_code == "NBA":
        strategy = NbaFeatureStrategy()
    elif league_code == "NHL":
        strategy = NhlFeatureStrategy()
    else:
        raise ValueError(f"Unsupported league '{league}'. Expected one of: MLB, NHL, NBA.")
    return build_features_from_interim_with_strategy(interim_dir=interim_dir, processed_dir=processed_dir, strategy=strategy)
