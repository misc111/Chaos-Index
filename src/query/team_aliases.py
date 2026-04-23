"""MLB team alias resolution for casual queries."""

from __future__ import annotations

import re
from typing import Any


MLB_TEAM_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "ARI": ("arizona", "arizona diamondbacks", "diamondbacks", "dbacks", "d backs"),
    "ATL": ("atlanta", "atlanta braves", "braves"),
    "ATH": ("athletics", "sacramento athletics", "oakland athletics", "oakland a s", "a s"),
    "BAL": ("baltimore", "baltimore orioles", "orioles"),
    "BOS": ("boston", "boston red sox", "red sox", "bosox"),
    "CHC": ("chicago cubs", "cubs", "north siders"),
    "CHW": ("chicago white sox", "white sox", "south siders"),
    "CIN": ("cincinnati", "cincinnati reds", "reds"),
    "CLE": ("cleveland", "cleveland guardians", "guardians"),
    "COL": ("colorado", "colorado rockies", "rockies"),
    "DET": ("detroit", "detroit tigers", "tigers"),
    "HOU": ("houston", "houston astros", "astros"),
    "KC": ("kansas city", "kansas city royals", "royals", "kc"),
    "LAA": ("anaheim", "los angeles angels", "la angels", "angels"),
    "LAD": ("los angeles dodgers", "la dodgers", "dodgers"),
    "MIA": ("miami", "miami marlins", "marlins"),
    "MIL": ("milwaukee", "milwaukee brewers", "brewers"),
    "MIN": ("minnesota", "minnesota twins", "twins"),
    "NYM": ("new york mets", "mets"),
    "NYY": ("new york yankees", "yankees"),
    "PHI": ("philadelphia", "philadelphia phillies", "phillies"),
    "PIT": ("pittsburgh", "pittsburgh pirates", "pirates", "bucs"),
    "SD": ("san diego", "san diego padres", "padres", "sd"),
    "SEA": ("seattle", "seattle mariners", "mariners"),
    "SF": ("san francisco", "san francisco giants", "giants", "sf"),
    "STL": ("st louis", "st louis cardinals", "cardinals", "cards"),
    "TB": ("tampa bay rays", "tampa bay", "rays", "tb"),
    "TEX": ("texas", "texas rangers", "rangers"),
    "TOR": ("toronto", "toronto blue jays", "blue jays", "jays"),
    "WSH": ("washington nationals", "nationals", "nats"),
}

TEAM_ALIAS_GROUPS_BY_LEAGUE: dict[str, dict[str, tuple[str, ...]]] = {"MLB": MLB_TEAM_ALIAS_GROUPS}
TEAM_ALIAS_GROUPS = MLB_TEAM_ALIAS_GROUPS

TEAM_ALIASES_BY_LEAGUE = {
    "MLB": {alias: team for team, aliases in MLB_TEAM_ALIAS_GROUPS.items() for alias in aliases}
}
TEAM_ALIASES_BY_LEAGUE["MLB"].update(
    {
        "bluejays": "TOR",
        "d backs": "ARI",
        "dbacks": "ARI",
        "whitesox": "CHW",
        "redsox": "BOS",
    }
)

TEAM_ABBREV_ALIASES_BY_LEAGUE = {
    "MLB": {
        "ATH": "ATH",
        "CWS": "CHW",
        "KC": "KC",
        "KCR": "KC",
        "OAK": "ATH",
        "SD": "SD",
        "SDP": "SD",
        "SF": "SF",
        "SFG": "SF",
        "TB": "TB",
        "TBR": "TB",
        "WAS": "WSH",
        "WSN": "WSH",
    }
}

TEAM_ABBREVIATIONS_BY_LEAGUE = {
    "MLB": sorted(set(MLB_TEAM_ALIAS_GROUPS.keys()) | set(TEAM_ABBREV_ALIASES_BY_LEAGUE["MLB"].keys()))
}

TEAM_ABBREV_PATTERN_BY_LEAGUE = {
    league: re.compile(rf"\b({'|'.join(sorted(abbrevs))})\b") if abbrevs else re.compile(r"(?!x)x")
    for league, abbrevs in TEAM_ABBREVIATIONS_BY_LEAGUE.items()
}

_ALIAS_ITEMS = [
    (alias, league, team)
    for league, aliases in TEAM_ALIASES_BY_LEAGUE.items()
    for alias, team in aliases.items()
]
_ALIAS_ITEMS.sort(key=lambda item: len(item[0]), reverse=True)
TEAM_ALIAS_REGEX = [
    (re.compile(rf"\b{re.escape(alias)}\b"), league, team, len(alias))
    for alias, league, team in _ALIAS_ITEMS
]


def canonical_league(league: str | None) -> str | None:
    token = str(league or "").strip().upper()
    return "MLB" if token == "MLB" else None


def canonical_team_code(team: Any, league: str) -> str:
    token = str(team or "").strip().upper()
    if not token or str(league or "").strip().upper() != "MLB":
        return ""
    return TEAM_ABBREV_ALIASES_BY_LEAGUE["MLB"].get(token, token)
