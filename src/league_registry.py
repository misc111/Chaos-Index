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

from src.common.time import utc_now_iso
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


def _empty_fetch_result(source: str, note: str) -> SourceFetchResult:
    as_of_utc = utc_now_iso()
    metadata = {"fetched_at_utc": as_of_utc, "fallback_used": 1, "note": note}
    return SourceFetchResult(
        source=source,
        snapshot_id=f"{source}_empty",
        extracted_at_utc=as_of_utc,
        raw_path="",
        metadata=metadata,
        dataframe=pd.DataFrame(),
    )


def _empty_starter_context(*args: object, **kwargs: object) -> SourceFetchResult:
    del args, kwargs
    return _empty_fetch_result("starter_context_empty", "league_has_no_dedicated_starter_context_feed")


def _wrap_injuries_proxy(fn: Callable[..., SourceFetchResult]) -> Callable[..., SourceFetchResult]:
    def _wrapped(client: HttpClient, teams: list[str] | None = None, games_df: pd.DataFrame | None = None) -> SourceFetchResult:
        del games_df
        return fn(client, teams=teams or [])

    return _wrapped


@lru_cache(maxsize=1)
def _registry() -> dict[str, LeagueAdapter]:
    mlb = get_league_registry_entry("MLB")
    nhl = get_league_registry_entry("NHL")
    nba = get_league_registry_entry("NBA")
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
    from src.data_sources.nba.games import fetch_games as fetch_nba_games
    from src.data_sources.nba.goalies import fetch_goalie_game_stats as fetch_nba_goalie_stats
    from src.data_sources.nba.injuries import fetch_injuries_proxy as fetch_nba_injuries
    from src.data_sources.nba.odds import fetch_public_odds_optional as fetch_nba_odds
    from src.data_sources.nba.players import fetch_players as fetch_nba_players
    from src.data_sources.nba.results import build_results_from_games as build_nba_results
    from src.data_sources.nba.schedule import fetch_upcoming_schedule as fetch_nba_schedule
    from src.data_sources.nba.teams import fetch_teams as fetch_nba_teams
    from src.data_sources.nba.xg import fetch_xg_optional as fetch_nba_xg
    from src.data_sources.nhl.games import fetch_games as fetch_nhl_games
    from src.data_sources.nhl.goalies import fetch_goalie_game_stats as fetch_nhl_goalie_stats
    from src.data_sources.nhl.injuries import fetch_injuries_proxy as fetch_nhl_injuries
    from src.data_sources.nhl.odds import fetch_public_odds_optional as fetch_nhl_odds
    from src.data_sources.nhl.players import fetch_players as fetch_nhl_players
    from src.data_sources.nhl.results import build_results_from_games as build_nhl_results
    from src.data_sources.nhl.schedule import fetch_upcoming_schedule as fetch_nhl_schedule
    from src.data_sources.nhl.teams import fetch_teams as fetch_nhl_teams
    from src.data_sources.nhl.xg import fetch_xg_optional as fetch_nhl_xg

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
        "NHL": LeagueAdapter(
            metadata=LeagueMetadata(
                code=nhl.code,
                slug=nhl.slug,
                default_config_path=nhl.default_config_path,
                config_env_var=nhl.config_env_var,
                project_name=nhl.project_name,
                db_path=nhl.db_path,
                db_env_var=nhl.db_env_var,
                display_label=nhl.display_label,
                championship_name=nhl.championship_name,
                championship_probability_key=nhl.championship_probability_key,
                uncertainty_policy_name=nhl.uncertainty_policy_name,
            ),
            fetch_games=fetch_nhl_games,
            fetch_team_game_stats=fetch_nhl_goalie_stats,
            fetch_starter_context=_empty_starter_context,
            fetch_injuries_report=_wrap_injuries_proxy(fetch_nhl_injuries),
            fetch_public_odds_optional=fetch_nhl_odds,
            fetch_players=fetch_nhl_players,
            build_results_from_games=build_nhl_results,
            fetch_upcoming_schedule=fetch_nhl_schedule,
            fetch_teams=fetch_nhl_teams,
            fetch_context_metrics_optional=fetch_nhl_xg,
        ),
        "NBA": LeagueAdapter(
            metadata=LeagueMetadata(
                code=nba.code,
                slug=nba.slug,
                default_config_path=nba.default_config_path,
                config_env_var=nba.config_env_var,
                project_name=nba.project_name,
                db_path=nba.db_path,
                db_env_var=nba.db_env_var,
                display_label=nba.display_label,
                championship_name=nba.championship_name,
                championship_probability_key=nba.championship_probability_key,
                uncertainty_policy_name=nba.uncertainty_policy_name,
            ),
            fetch_games=fetch_nba_games,
            fetch_team_game_stats=fetch_nba_goalie_stats,
            fetch_starter_context=_empty_starter_context,
            fetch_injuries_report=_wrap_injuries_proxy(fetch_nba_injuries),
            fetch_public_odds_optional=fetch_nba_odds,
            fetch_players=fetch_nba_players,
            build_results_from_games=build_nba_results,
            fetch_upcoming_schedule=fetch_nba_schedule,
            fetch_teams=fetch_nba_teams,
            fetch_context_metrics_optional=fetch_nba_xg,
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
