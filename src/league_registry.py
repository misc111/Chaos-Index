"""Central league registry for shared orchestration contracts.

The CLI, ingest services, and query layer all need the same league-specific
facts and fetch hooks. Keeping them here prevents the NHL/NBA switch logic from
drifting across entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import pandas as pd

from src.common.manifests import load_league_manifest
from src.data_sources.base import HttpClient, SourceFetchResult

FetchGamesFn = Callable[[HttpClient], SourceFetchResult]
FetchGoalieStatsFn = Callable[[HttpClient, list[int], int], SourceFetchResult]
FetchInjuriesFn = Callable[[HttpClient, list[str]], SourceFetchResult]
FetchOddsFn = Callable[[HttpClient, pd.DataFrame, str], SourceFetchResult]
FetchPlayersFn = Callable[[HttpClient, list[str], str, pd.DataFrame], SourceFetchResult]
BuildResultsFn = Callable[[pd.DataFrame], pd.DataFrame]
FetchScheduleFn = Callable[[HttpClient, int], SourceFetchResult]
FetchTeamsFn = Callable[[HttpClient], SourceFetchResult]
FetchXgFn = Callable[[HttpClient], SourceFetchResult]


@dataclass(frozen=True)
class LeagueMetadata:
    code: str
    slug: str
    default_config_path: str
    env_config_var: str
    project_name: str
    db_path: str
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
    fetch_goalie_game_stats: Callable[..., SourceFetchResult]
    fetch_injuries_proxy: Callable[..., SourceFetchResult]
    fetch_public_odds_optional: Callable[..., SourceFetchResult]
    fetch_players: Callable[..., SourceFetchResult]
    build_results_from_games: BuildResultsFn
    fetch_upcoming_schedule: Callable[..., SourceFetchResult]
    fetch_teams: Callable[..., SourceFetchResult]
    fetch_xg_optional: Callable[..., SourceFetchResult]

    @property
    def code(self) -> str:
        return self.metadata.code


def canonicalize_league(league: str | None) -> str:
    token = str(league or "").strip().upper()
    if token in {"NHL", "NBA"}:
        return token
    raise ValueError(f"Unsupported league '{league}'. Expected one of: NHL, NBA.")


@lru_cache(maxsize=1)
def _registry() -> dict[str, LeagueAdapter]:
    league_manifest = load_league_manifest()["leagues"]
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
        "NHL": LeagueAdapter(
            metadata=LeagueMetadata(
                **league_manifest["NHL"],
            ),
            fetch_games=fetch_nhl_games,
            fetch_goalie_game_stats=fetch_nhl_goalie_stats,
            fetch_injuries_proxy=fetch_nhl_injuries,
            fetch_public_odds_optional=fetch_nhl_odds,
            fetch_players=fetch_nhl_players,
            build_results_from_games=build_nhl_results,
            fetch_upcoming_schedule=fetch_nhl_schedule,
            fetch_teams=fetch_nhl_teams,
            fetch_xg_optional=fetch_nhl_xg,
        ),
        "NBA": LeagueAdapter(
            metadata=LeagueMetadata(
                **league_manifest["NBA"],
            ),
            fetch_games=fetch_nba_games,
            fetch_goalie_game_stats=fetch_nba_goalie_stats,
            fetch_injuries_proxy=fetch_nba_injuries,
            fetch_public_odds_optional=fetch_nba_odds,
            fetch_players=fetch_nba_players,
            build_results_from_games=build_nba_results,
            fetch_upcoming_schedule=fetch_nba_schedule,
            fetch_teams=fetch_nba_teams,
            fetch_xg_optional=fetch_nba_xg,
        ),
    }


def get_league_adapter(league: str | None) -> LeagueAdapter:
    return _registry()[canonicalize_league(league)]


def get_league_metadata(league: str | None) -> LeagueMetadata:
    return get_league_adapter(league).metadata


def supported_leagues() -> tuple[str, ...]:
    return tuple(_registry().keys())
