from src.query import parse
from src.query.team_aliases import (
    LEGACY_COMPARISON_TEAM_ALIAS_GROUPS_BY_LEAGUE,
    NBA_TEAM_ALIAS_GROUPS,
    NHL_TEAM_ALIAS_GROUPS,
    TEAM_ALIAS_GROUPS_BY_LEAGUE,
)


def test_query_alias_registry_keeps_retired_leagues_quarantined() -> None:
    assert sorted(TEAM_ALIAS_GROUPS_BY_LEAGUE) == ["MLB"]
    assert NBA_TEAM_ALIAS_GROUPS == {}
    assert NHL_TEAM_ALIAS_GROUPS == {}
    assert LEGACY_COMPARISON_TEAM_ALIAS_GROUPS_BY_LEAGUE == {
        "NBA": {},
        "NHL": {},
    }


def test_parse_compatibility_exports_do_not_reactivate_nba_or_nhl() -> None:
    assert parse.NBA_TEAM_ALIAS_GROUPS == {}
    assert parse.NHL_TEAM_ALIAS_GROUPS == {}
    assert sorted(parse.TEAM_ALIAS_GROUPS_BY_LEAGUE) == ["MLB"]
