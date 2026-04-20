"""Contracts for the shared feature pipeline.

The pipeline owns staged orchestration. Each league strategy only overrides the
domain transforms that differ across supported sports, with the active rebuild focused on MLB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class FeatureStrategy(Protocol):
    league: str
    team_stats_artifact_name: str
    starter_context_artifact_name: str
    players_artifact_name: str
    injuries_artifact_name: str
    context_metrics_artifact_name: str | None
    summary_aggregations: dict[str, tuple[str, str]]
    starter_aggregations: dict[str, tuple[str, str]]
    team_value_for_column: str
    team_value_against_column: str
    team_result_column: str
    rolling_value_columns: list[str]
    diff_pairs: list[tuple[str, str]]
    direct_event_drop_columns: list[str]

    def prepare_team_games(self, team_games: pd.DataFrame, players_df: pd.DataFrame, injuries_df: pd.DataFrame) -> pd.DataFrame:
        ...

    def finalize_team_games(self, team_games: pd.DataFrame) -> pd.DataFrame:
        ...

    def enrich_game_level(
        self,
        merged: pd.DataFrame,
        games_df: pd.DataFrame,
        team_games: pd.DataFrame,
        context_df: pd.DataFrame,
    ) -> pd.DataFrame:
        ...

    def add_model_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def feature_hash_payload(self, feature_columns: list[str]) -> dict:
        ...


@dataclass(frozen=True)
class BaseFeatureStrategy:
    league: str
    summary_aggregations: dict[str, tuple[str, str]]
    starter_aggregations: dict[str, tuple[str, str]]
    team_value_for_column: str
    team_value_against_column: str
    team_result_column: str
    rolling_value_columns: list[str]
    diff_pairs: list[tuple[str, str]]
    direct_event_drop_columns: list[str]
    team_stats_artifact_name: str = "team_stats"
    starter_context_artifact_name: str = "team_stats"
    players_artifact_name: str = "players"
    injuries_artifact_name: str = "injuries"
    context_metrics_artifact_name: str | None = None
