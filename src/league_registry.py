"""Central league registry for shared orchestration contracts.

The CLI, ingest services, and query layer all need the same league-specific
facts and fetch hooks. Keeping them here prevents league-switch logic from
drifting across entry points during the MLB-first rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import pandas as pd

from src.data_sources.base import HttpClient, SourceFetchResult
from src.registry.leagues import (
    canonicalize_league as canonicalize_registry_league,
    get_league_registry_entry,
    league_codes,
)

FetchGamesFn = Callable[[HttpClient], SourceFetchResult]
FetchTeamStatsFn = Callable[[HttpClient, list[int], int], SourceFetchResult]
FetchStarterContextFn = Callable[[HttpClient, list[int], int], SourceFetchResult]
FetchInjuriesFn = Callable[[HttpClient, list[str], pd.DataFrame | None], SourceFetchResult]
FetchOddsFn = Callable[[HttpClient, pd.DataFrame, str], SourceFetchResult]
FetchPlayersFn = Callable[[HttpClient, list[str], str, pd.DataFrame], SourceFetchResult]
BuildResultsFn = Callable[[pd.DataFrame], pd.DataFrame]
FetchScheduleFn = Callable[[HttpClient, int], SourceFetchResult]
FetchTeamsFn = Callable[[HttpClient], SourceFetchResult]
FetchContextMetricsFn = Callable[[HttpClient], SourceFetchResult]


@dataclass(frozen=True)
class LeagueMetadata:
    """Canonical league metadata exposed to shared orchestration layers."""

    code: str
    slug: str
    default_config_path: str
    config_env_var: str
    project_name: str
    db_path: str
    db_env_var: str
    display_label: str
    championship_name: str
    championship_probability_key: str
    uncertainty_policy_name: str


@dataclass(frozen=True)
class LeagueAdapter:
    """Typed adapter for league-specific ingest implementations.

    Shared orchestration should only depend on this surface. League-specific
    modules may add richer behavior internally, but cross-league services
    should only call the responsibilities listed here.
    """

    metadata: LeagueMetadata
    fetch_games: Callable[..., SourceFetchResult]
    fetch_team_game_stats: Callable[..., SourceFetchResult]
    fetch_starter_context: Callable[..., SourceFetchResult]
    fetch_injuries_report: Callable[..., SourceFetchResult]
    fetch_public_odds_optional: Callable[..., SourceFetchResult]
    fetch_players: Callable[..., SourceFetchResult]
    build_results_from_games: BuildResultsFn
    fetch_upcoming_schedule: Callable[..., SourceFetchResult]
    fetch_teams: Callable[..., SourceFetchResult]
    fetch_context_metrics_optional: Callable[..., SourceFetchResult]

    @property
    def code(self) -> str:
        return self.metadata.code


def canonicalize_league(league: str | None) -> str:
    """Normalize a user or config league value into a supported code."""

    return canonicalize_registry_league(league)


@lru_cache(maxsize=1)
def _registry() -> dict[str, LeagueAdapter]:
    mlb = get_league_registry_entry("MLB")
    from src.data_sources.mlb.games import fetch_games as fetch_mlb_games
    from src.data_sources.mlb.injuries import fetch_injuries_report as fetch_mlb_injuries
    from src.data_sources.mlb.odds import fetch_public_odds_optional as fetch_mlb_odds
    from src.data_sources.mlb.players import fetch_players as fetch_mlb_players
    from src.data_sources.mlb.results import build_results_from_games as build_mlb_results
    from src.data_sources.mlb.schedule import fetch_upcoming_schedule as fetch_mlb_schedule
    from src.data_sources.mlb.starting_pitchers import fetch_starter_context as fetch_mlb_starter_context
    from src.data_sources.mlb.team_stats import fetch_team_game_stats as fetch_mlb_team_stats
    from src.data_sources.mlb.teams import fetch_teams as fetch_mlb_teams
    from src.data_sources.mlb.weather import fetch_weather_context_optional as fetch_mlb_weather

    return {
        "MLB": LeagueAdapter(
            metadata=LeagueMetadata(
                code=mlb.code,
                slug=mlb.slug,
                default_config_path=mlb.default_config_path,
                config_env_var=mlb.config_env_var,
                project_name=mlb.project_name,
                db_path=mlb.db_path,
                db_env_var=mlb.db_env_var,
                display_label=mlb.display_label,
                championship_name=mlb.championship_name,
                championship_probability_key=mlb.championship_probability_key,
                uncertainty_policy_name=mlb.uncertainty_policy_name,
            ),
            fetch_games=fetch_mlb_games,
            fetch_team_game_stats=fetch_mlb_team_stats,
            fetch_starter_context=fetch_mlb_starter_context,
            fetch_injuries_report=fetch_mlb_injuries,
            fetch_public_odds_optional=fetch_mlb_odds,
            fetch_players=fetch_mlb_players,
            build_results_from_games=build_mlb_results,
            fetch_upcoming_schedule=fetch_mlb_schedule,
            fetch_teams=fetch_mlb_teams,
            fetch_context_metrics_optional=fetch_mlb_weather,
        ),
    }


def get_league_adapter(league: str | None) -> LeagueAdapter:
    """Return the typed ingest/query adapter for a supported league."""

    return _registry()[canonicalize_league(league)]


def get_league_metadata(league: str | None) -> LeagueMetadata:
    """Return canonical metadata for a supported league."""

    return get_league_adapter(league).metadata


def supported_leagues() -> tuple[str, ...]:
    """Return supported league codes in the canonical orchestration order."""

    return league_codes()
