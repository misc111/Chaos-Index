"""Backward-compatible parse entry point over modular query components."""

from __future__ import annotations

from src.query.intent_parser import QueryIntent, parse_question
from src.query.team_aliases import (
    NBA_TEAM_ALIAS_GROUPS,
    NHL_TEAM_ALIAS_GROUPS,
    TEAM_ABBREV_ALIASES_BY_LEAGUE,
    TEAM_ABBREV_PATTERN_BY_LEAGUE,
    TEAM_ALIAS_GROUPS,
    TEAM_ALIAS_GROUPS_BY_LEAGUE,
    TEAM_ALIASES_BY_LEAGUE,
)

__all__ = [
    "NBA_TEAM_ALIAS_GROUPS",
    "NHL_TEAM_ALIAS_GROUPS",
    "QueryIntent",
    "TEAM_ABBREV_ALIASES_BY_LEAGUE",
    "TEAM_ABBREV_PATTERN_BY_LEAGUE",
    "TEAM_ALIAS_GROUPS",
    "TEAM_ALIAS_GROUPS_BY_LEAGUE",
    "TEAM_ALIASES_BY_LEAGUE",
    "parse_question",
]
