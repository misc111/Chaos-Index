from __future__ import annotations

import re
from dataclasses import dataclass


TEAM_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
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
    "UTA": (
        "utah",
        "utah hockey club",
        "utah hc",
        # Legacy references still appear in casual questions.
        "arizona",
        "arizona coyotes",
        "coyotes",
    ),
    "VAN": ("vancouver", "vancouver canucks", "canucks"),
    "VGK": ("vegas", "vegas golden knights", "golden knights", "knights"),
    "WPG": ("winnipeg", "winnipeg jets", "jets"),
    "WSH": ("washington", "washington capitals", "capitals", "caps"),
}

TEAM_ALIASES = {alias: team for team, aliases in TEAM_ALIAS_GROUPS.items() for alias in aliases}
TEAM_ALIASES["mapleleafs"] = "TOR"  # common no-space typo
TEAM_ALIASES["devs"] = "NJD"  # common shorthand

TEAM_ABBREV_PATTERN = re.compile(
    r"\b(ANA|ARI|BOS|BUF|CAR|CBJ|CGY|CHI|COL|DAL|DET|EDM|FLA|LAK|MIN|MTL|NJD|NSH|NYI|NYR|OTT|PHI|PIT|SEA|SJS|STL|TBL|TOR|UTA|VAN|VGK|WPG|WSH)\b"
)

_TEAM_ALIAS_REGEX = [
    (re.compile(rf"\b{re.escape(alias)}\b"), team)
    for alias, team in sorted(TEAM_ALIASES.items(), key=lambda item: len(item[0]), reverse=True)
]


@dataclass
class QueryIntent:
    intent_type: str
    team: str | None = None
    window_days: int = 60
    n_games: int = 1



def _parse_window_days(question: str) -> int:
    m = re.search(r"last\s+(\d+)\s+day", question.lower())
    if m:
        return int(m.group(1))
    return 60



_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "couple": 2,
    "few": 3,
}



def _normalize(question: str) -> str:
    lowered = question.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()



def _token_to_int(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)



def _parse_next_games_count(question: str) -> int | None:
    q = _normalize(question)

    m = re.search(r"\bnext\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|couple|few)\s+games?\b", q)
    if m:
        n = _token_to_int(m.group(1))
        return max(1, min(int(n), 10)) if n is not None else None

    # Casual phrasing like "next three" or "next couple" without explicit "games".
    m = re.search(r"\bnext\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|couple|few)\b", q)
    if m:
        n = _token_to_int(m.group(1))
        return max(1, min(int(n), 10)) if n is not None else None

    if re.search(r"\bnext\s+game\b", q) or "tonight" in q or "tomorrow" in q:
        return 1

    if re.search(r"\bnext\s+games\b", q):
        return 3

    return None



def _is_stanley_cup_question(question: str) -> bool:
    q = _normalize(question)
    if "stanley cup" in q:
        return True
    if "cup" in q and any(k in q for k in ["chance", "probability", "odds", "win", "wins"]):
        return True
    cup_signals = [
        "win the cup",
        "win cup",
        "hoist the cup",
        "lift the cup",
        "win it all",
        "championship",
        "champion",
        "title",
    ]
    return any(s in q for s in cup_signals)



def _parse_team(question: str) -> str | None:
    q = _normalize(question)
    for pattern, abbr in _TEAM_ALIAS_REGEX:
        if pattern.search(q):
            return abbr
    # Direct team abbrev mention.
    m = TEAM_ABBREV_PATTERN.search(question.upper())
    if m:
        abbr = m.group(1)
        return "UTA" if abbr == "ARI" else abbr
    return None



def parse_question(question: str) -> QueryIntent:
    q = question.lower().strip()
    team = _parse_team(question)
    n_games = _parse_next_games_count(question)

    if any(k in q for k in ["which model", "best model", "performed best", "leaderboard"]):
        return QueryIntent(intent_type="best_model", window_days=_parse_window_days(q))

    if _is_stanley_cup_question(question):
        return QueryIntent(intent_type="team_stanley_cup", team=team)

    # In this NHL-only command, if the user mentioned a team in any casual way,
    # default to the team's next-game win probability.
    if team:
        if n_games and n_games > 1:
            return QueryIntent(intent_type="team_next_n_games", team=team, n_games=n_games)
        return QueryIntent(intent_type="team_next_game", team=team)

    if n_games and n_games > 1 and any(k in q for k in ["chance", "probability", "odds", "likely", "win", "wins"]):
        return QueryIntent(intent_type="team_next_n_games", team=None, n_games=n_games)

    if (n_games == 1 or "next game" in q) and any(k in q for k in ["chance", "probability", "odds", "likely"]):
        return QueryIntent(intent_type="team_next_game", team=None)

    return QueryIntent(intent_type="help")
