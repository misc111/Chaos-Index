from __future__ import annotations

import re
from dataclasses import dataclass


TEAM_ALIASES = {
    "leafs": "TOR",
    "maple leafs": "TOR",
    "toronto": "TOR",
    "bruins": "BOS",
    "canadiens": "MTL",
    "habs": "MTL",
    "rangers": "NYR",
    "islanders": "NYI",
    "devils": "NJD",
    "flyers": "PHI",
    "penguins": "PIT",
    "capitals": "WSH",
    "hurricanes": "CAR",
    "lightning": "TBL",
    "panthers": "FLA",
    "sabres": "BUF",
    "senators": "OTT",
    "red wings": "DET",
    "blue jackets": "CBJ",
    "blackhawks": "CHI",
    "wild": "MIN",
    "jets": "WPG",
    "predators": "NSH",
    "blues": "STL",
    "stars": "DAL",
    "avalanche": "COL",
    "utah": "UTA",
    "knights": "VGK",
    "golden knights": "VGK",
    "kings": "LAK",
    "ducks": "ANA",
    "sharks": "SJS",
    "kraken": "SEA",
    "canucks": "VAN",
    "flames": "CGY",
    "oilers": "EDM",
}


@dataclass
class QueryIntent:
    intent_type: str
    team: str | None = None
    window_days: int = 60



def _parse_window_days(question: str) -> int:
    m = re.search(r"last\s+(\d+)\s+day", question.lower())
    if m:
        return int(m.group(1))
    return 60



def _parse_team(question: str) -> str | None:
    q = question.lower()
    for name, abbr in TEAM_ALIASES.items():
        if re.search(rf"\b{re.escape(name)}\b", q):
            return abbr
    # Direct team abbrev mention.
    m = re.search(r"\b(ANA|ARI|BOS|BUF|CAR|CBJ|CGY|CHI|COL|DAL|DET|EDM|FLA|LAK|MIN|MTL|NJD|NSH|NYI|NYR|OTT|PHI|PIT|SEA|SJS|STL|TBL|TOR|UTA|VAN|VGK|WPG|WSH)\b", question.upper())
    if m:
        return m.group(1)
    return None



def parse_question(question: str) -> QueryIntent:
    q = question.lower().strip()

    if any(k in q for k in ["which model", "best model", "performed best", "leaderboard"]):
        return QueryIntent(intent_type="best_model", window_days=_parse_window_days(q))

    if any(k in q for k in ["chance", "probability", "what's the chance", "what is the chance"]):
        team = _parse_team(q)
        if "next game" in q or team:
            return QueryIntent(intent_type="team_next_game", team=team)

    return QueryIntent(intent_type="help")
