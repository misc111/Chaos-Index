"""Canonical league registry and runtime resolution helpers."""

from __future__ import annotations

import os
from pathlib import Path

from src.registry.types import LeagueRegistryEntry


ROOT_DIR = Path(__file__).resolve().parents[2]
PRIMARY_LEAGUE_CODE = "MLB"


LEAGUE_REGISTRY: tuple[LeagueRegistryEntry, ...] = (
    LeagueRegistryEntry(
        code="MLB",
        slug="mlb",
        display_label="MLB",
        default_config_path="configs/mlb.yaml",
        config_env_var="MLB_CONFIG_PATH",
        project_name="mlb_forecast",
        db_path="data/processed/mlb/mlb_forecast.db",
        db_env_var="MLB_DB_PATH",
        championship_name="World Series",
        championship_probability_key="world_series_prob",
        uncertainty_policy_name="mlb_starting_pitcher_bullpen",
        primary_rebuild_lane=True,
    ),
    LeagueRegistryEntry(
        code="NHL",
        slug="nhl",
        display_label="NHL",
        default_config_path="configs/nhl.yaml",
        config_env_var="NHL_CONFIG_PATH",
        project_name="nhl_forecast",
        db_path="data/processed/nhl_forecast.db",
        db_env_var="NHL_DB_PATH",
        championship_name="Stanley Cup",
        championship_probability_key="stanley_cup_prob",
        uncertainty_policy_name="nhl_goalie_rink",
    ),
    LeagueRegistryEntry(
        code="NBA",
        slug="nba",
        display_label="NBA",
        default_config_path="configs/nba.yaml",
        config_env_var="NBA_CONFIG_PATH",
        project_name="nba_forecast",
        db_path="data/processed/nba_forecast.db",
        db_env_var="NBA_DB_PATH",
        championship_name="NBA Finals",
        championship_probability_key="nba_finals_prob",
        uncertainty_policy_name="nba_availability_market",
    ),
)

_LEAGUE_BY_CODE = {entry.code: entry for entry in LEAGUE_REGISTRY}
_LEAGUE_ALIAS_MAP = {
    alias.upper(): entry.code
    for entry in LEAGUE_REGISTRY
    for alias in (entry.code, entry.slug.upper(), *entry.aliases)
}


def ordered_league_entries() -> tuple[LeagueRegistryEntry, ...]:
    """Return supported leagues in the canonical orchestration order."""

    return LEAGUE_REGISTRY


def league_codes() -> tuple[str, ...]:
    """Return the canonical league-code tuple."""

    return tuple(entry.code for entry in LEAGUE_REGISTRY)


def primary_league_code() -> str:
    """Return the canonical primary rebuild league."""

    return PRIMARY_LEAGUE_CODE


def primary_rebuild_league_codes() -> tuple[str, ...]:
    """Return leagues that participate in the primary shipped rebuild lane."""

    return tuple(entry.code for entry in LEAGUE_REGISTRY if entry.primary_rebuild_lane)


def canonicalize_league(value: str | None) -> str:
    """Normalize user or config league input into a supported code."""

    token = str(value or "").strip().upper()
    resolved = _LEAGUE_ALIAS_MAP.get(token, token)
    if resolved in _LEAGUE_BY_CODE:
        return resolved
    raise ValueError(f"Unsupported league '{value}'. Expected one of: {', '.join(league_codes())}.")


def get_league_registry_entry(value: str | None) -> LeagueRegistryEntry:
    """Resolve a league alias or code into registry metadata."""

    return _LEAGUE_BY_CODE[canonicalize_league(value)]


def default_config_path(value: str | None = "MLB") -> str:
    """Return the canonical default config path for a league."""

    return get_league_registry_entry(value).default_config_path


def default_db_path(value: str | None = "MLB") -> str:
    """Return the canonical default DB path for a league."""

    return get_league_registry_entry(value).db_path


def resolve_config_path(value: str | None, *, root_dir: Path = ROOT_DIR) -> Path:
    """Resolve a config path with per-league environment overrides applied."""

    entry = get_league_registry_entry(value)
    override = os.getenv(entry.config_env_var)
    raw_path = override or entry.default_config_path
    return (root_dir / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path).resolve()


def resolve_db_path(value: str | None, *, root_dir: Path = ROOT_DIR) -> Path:
    """Resolve a DB path with per-league environment overrides applied."""

    entry = get_league_registry_entry(value)
    override = os.getenv(entry.db_env_var)
    raw_path = override or entry.db_path
    return (root_dir / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path).resolve()


def league_manifest_payload() -> dict[str, object]:
    """Render the deterministic league manifest payload."""

    return {
        "version": 1,
        "source": "code_registry",
        "primary_league": PRIMARY_LEAGUE_CODE,
        "primary_rebuild_leagues": [entry.code for entry in LEAGUE_REGISTRY if entry.primary_rebuild_lane],
        "leagues": {
            entry.code: {
                "code": entry.code,
                "slug": entry.slug,
                "display_label": entry.display_label,
                "default_config_path": entry.default_config_path,
                "config_env_var": entry.config_env_var,
                "project_name": entry.project_name,
                "db_path": entry.db_path,
                "db_env_var": entry.db_env_var,
                "championship_name": entry.championship_name,
                "championship_probability_key": entry.championship_probability_key,
                "uncertainty_policy_name": entry.uncertainty_policy_name,
                "primary_rebuild_lane": entry.primary_rebuild_lane,
                "aliases": list(entry.aliases),
            }
            for entry in LEAGUE_REGISTRY
        },
    }
