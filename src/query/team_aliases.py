"""League-aware team alias resolution for casual queries.

These tables intentionally stay separate from answer construction so parsing
and reporting can evolve independently.
"""

from __future__ import annotations

import re
from typing import Any


NHL_TEAM_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "ANA": ("anaheim", "anaheim ducks", "ducks"),
    "BOS": ("boston", "boston bruins", "bruins"),
    "BUF": ("buffalo", "buffalo sabres", "sabres", "sabers"),
    "CAR": ("carolina", "carolina hurricanes", "hurricanes", "canes"),
    "CBJ": ("columbus", "columbus blue jackets", "blue jackets", "jackets"),
    "CGY": ("calgary", "calgary flames", "flames"),
    "CHI": ("chicago", "chicago blackhawks", "blackhawks", "black hawks"),
    "COL": ("colorado", "colorado avalanche", "avalanche", "avs"),
    "DAL": ("dallas", "dallas stars", "stars"),
    "DET": ("detroit", "detroit red wings", "red wings", "wings"),
    "EDM": ("edmonton", "edmonton oilers", "oilers"),
    "FLA": ("florida", "florida panthers", "panthers"),
    "LAK": ("los angeles", "los angeles kings", "kings"),
    "MIN": ("minnesota", "minnesota wild", "wild"),
    "MTL": ("montreal", "montreal canadiens", "canadiens", "habs"),
    "NJD": ("new jersey", "new jersey devils", "devils", "jersey"),
    "NSH": ("nashville", "nashville predators", "predators", "preds"),
    "NYI": ("new york islanders", "islanders"),
    "NYR": ("new york rangers", "rangers"),
    "OTT": ("ottawa", "ottawa senators", "senators", "sens"),
    "PHI": ("philadelphia", "philadelphia flyers", "flyers"),
    "PIT": ("pittsburgh", "pittsburgh penguins", "penguins", "pens"),
    "SEA": ("seattle", "seattle kraken", "kraken"),
    "SJS": ("san jose", "san jose sharks", "sharks"),
    "STL": ("st louis", "st louis blues", "blues"),
    "TBL": ("tampa", "tampa bay", "tampa bay lightning", "lightning", "bolts"),
    "TOR": ("toronto", "toronto maple leafs", "maple leafs", "leafs"),
    "UTA": ("utah", "utah hockey club", "utah hc", "arizona", "arizona coyotes", "coyotes"),
    "VAN": ("vancouver", "vancouver canucks", "canucks"),
    "VGK": ("vegas", "vegas golden knights", "golden knights", "knights"),
    "WPG": ("winnipeg", "winnipeg jets", "jets"),
    "WSH": ("washington", "washington capitals", "capitals", "caps"),
}

NBA_TEAM_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "ATL": ("atlanta", "atlanta hawks", "hawks"),
    "BOS": ("boston", "boston celtics", "celtics"),
    "BKN": ("brooklyn", "brooklyn nets", "nets"),
    "CHA": ("charlotte", "charlotte hornets", "hornets"),
    "CHI": ("chicago", "chicago bulls", "bulls"),
    "CLE": ("cleveland", "cleveland cavaliers", "cavaliers", "cavs"),
    "DAL": ("dallas", "dallas mavericks", "mavericks", "mavs"),
    "DEN": ("denver", "denver nuggets", "nuggets"),
    "DET": ("detroit", "detroit pistons", "pistons"),
    "GSW": ("golden state", "golden state warriors", "warriors", "dubs", "gsw"),
    "HOU": ("houston", "houston rockets", "rockets"),
    "IND": ("indiana", "indiana pacers", "pacers"),
    "LAC": ("la clippers", "los angeles clippers", "clippers"),
    "LAL": ("la lakers", "los angeles lakers", "lakers"),
    "MEM": ("memphis", "memphis grizzlies", "grizzlies", "grizz"),
    "MIA": ("miami", "miami heat", "heat"),
    "MIL": ("milwaukee", "milwaukee bucks", "bucks"),
    "MIN": ("minnesota", "minnesota timberwolves", "timberwolves", "wolves"),
    "NOP": ("new orleans", "new orleans pelicans", "pelicans", "pels", "nola"),
    "NYK": ("new york", "new york knicks", "knicks"),
    "OKC": ("okc", "oklahoma city", "oklahoma city thunder", "thunder"),
    "ORL": ("orlando", "orlando magic", "magic"),
    "PHI": ("philadelphia", "philadelphia 76ers", "philly", "76ers", "sixers"),
    "PHX": ("phoenix", "phoenix suns", "suns"),
    "POR": ("portland", "portland trail blazers", "trail blazers", "trailblazers", "blazers"),
    "SAC": ("sacramento", "sacramento kings", "kings"),
    "SAS": ("san antonio", "san antonio spurs", "spurs"),
    "TOR": ("toronto", "toronto raptors", "raptors"),
    "UTA": ("utah", "utah jazz", "jazz"),
    "WAS": ("washington", "washington wizards", "wizards"),
}

NCAAM_TEAM_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "DUKE": ("duke", "duke blue devils", "blue devils"),
    "UNC": ("unc", "north carolina", "north carolina tar heels", "tar heels"),
    "UK": ("kentucky", "kentucky wildcats", "wildcats"),
    "KU": ("kansas", "kansas jayhawks", "jayhawks"),
    "UCONN": ("uconn", "uconn huskies"),
    "CONN": ("connecticut", "connecticut huskies"),
    "MSU": ("michigan state", "michigan state spartans", "spartans"),
    "UCLA": ("ucla", "ucla bruins", "bruins"),
    "USC": ("usc", "southern california", "usc trojans", "trojans"),
    "GONZ": ("gonzaga", "gonzaga bulldogs", "bulldogs"),
    "PUR": ("purdue", "purdue boilermakers", "boilermakers"),
    "BAMA": ("alabama", "alabama crimson tide", "crimson tide"),
}

TEAM_ALIAS_GROUPS_BY_LEAGUE: dict[str, dict[str, tuple[str, ...]]] = {
    "NHL": NHL_TEAM_ALIAS_GROUPS,
    "NBA": NBA_TEAM_ALIAS_GROUPS,
    "NCAAM": NCAAM_TEAM_ALIAS_GROUPS,
}
TEAM_ALIAS_GROUPS = NHL_TEAM_ALIAS_GROUPS

TEAM_ALIASES_BY_LEAGUE = {
    league: {alias: team for team, aliases in groups.items() for alias in aliases}
    for league, groups in TEAM_ALIAS_GROUPS_BY_LEAGUE.items()
}
TEAM_ALIASES_BY_LEAGUE["NHL"]["mapleleafs"] = "TOR"
TEAM_ALIASES_BY_LEAGUE["NHL"]["devs"] = "NJD"
TEAM_ALIASES_BY_LEAGUE["NBA"]["6ers"] = "PHI"
TEAM_ALIASES_BY_LEAGUE["NBA"]["brk"] = "BKN"

TEAM_ABBREV_ALIASES_BY_LEAGUE = {
    "NHL": {"ARI": "UTA"},
    "NBA": {
        "BRK": "BKN",
        "GS": "GSW",
        "NO": "NOP",
        "NY": "NYK",
        "PHO": "PHX",
        "SA": "SAS",
        "UTAH": "UTA",
        "WSH": "WAS",
    },
    "NCAAM": {},
}

TEAM_ABBREVIATIONS_BY_LEAGUE = {
    league: sorted(set(groups.keys()) | set(TEAM_ABBREV_ALIASES_BY_LEAGUE.get(league, {}).keys()))
    for league, groups in TEAM_ALIAS_GROUPS_BY_LEAGUE.items()
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
    if not league:
        return None
    token = str(league).strip().upper()
    if token in TEAM_ALIAS_GROUPS_BY_LEAGUE:
        return token
    return None


def canonical_team_code(team: Any, league: str) -> str:
    token = str(team or "").strip().upper()
    if not token:
        return ""
    return TEAM_ABBREV_ALIASES_BY_LEAGUE.get(league, {}).get(token, token)
