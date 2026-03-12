"""Code-first registries for deterministic repo metadata and docs generation."""

from src.registry.commands import command_registry
from src.registry.dashboard_routes import dashboard_routes
from src.registry.leagues import canonicalize_league, league_codes, ordered_league_entries
from src.registry.models import ordered_model_entries, trainable_model_names

__all__ = [
    "canonicalize_league",
    "command_registry",
    "dashboard_routes",
    "league_codes",
    "ordered_league_entries",
    "ordered_model_entries",
    "trainable_model_names",
]
