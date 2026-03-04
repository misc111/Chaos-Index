from __future__ import annotations

import re
from dataclasses import dataclass, field


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

TEAM_ALIAS_GROUPS_BY_LEAGUE: dict[str, dict[str, tuple[str, ...]]] = {
    "NHL": NHL_TEAM_ALIAS_GROUPS,
    "NBA": NBA_TEAM_ALIAS_GROUPS,
}

# Backward compatibility for older imports.
TEAM_ALIAS_GROUPS = NHL_TEAM_ALIAS_GROUPS

TEAM_ALIASES_BY_LEAGUE = {
    league: {alias: team for team, aliases in groups.items() for alias in aliases}
    for league, groups in TEAM_ALIAS_GROUPS_BY_LEAGUE.items()
}
TEAM_ALIASES_BY_LEAGUE["NHL"]["mapleleafs"] = "TOR"  # common no-space typo
TEAM_ALIASES_BY_LEAGUE["NHL"]["devs"] = "NJD"  # common shorthand
TEAM_ALIASES_BY_LEAGUE["NBA"]["6ers"] = "PHI"  # common shorthand
TEAM_ALIASES_BY_LEAGUE["NBA"]["brk"] = "BKN"  # common alt abbreviation

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
}

TEAM_ABBREVIATIONS_BY_LEAGUE = {
    league: sorted(set(groups.keys()) | set(TEAM_ABBREV_ALIASES_BY_LEAGUE.get(league, {}).keys()))
    for league, groups in TEAM_ALIAS_GROUPS_BY_LEAGUE.items()
}

TEAM_ABBREV_PATTERN_BY_LEAGUE = {
    league: re.compile(rf"\b({'|'.join(sorted(abbrevs))})\b")
    for league, abbrevs in TEAM_ABBREVIATIONS_BY_LEAGUE.items()
}

_ALIAS_ITEMS = [
    (alias, league, team)
    for league, aliases in TEAM_ALIASES_BY_LEAGUE.items()
    for alias, team in aliases.items()
]
_ALIAS_ITEMS.sort(key=lambda item: len(item[0]), reverse=True)
_TEAM_ALIAS_REGEX = [
    (re.compile(rf"\b{re.escape(alias)}\b"), league, team, len(alias))
    for alias, league, team in _ALIAS_ITEMS
]


@dataclass
class QueryIntent:
    intent_type: str
    team: str | None = None
    league: str | None = None
    competition: str | None = None
    window_days: int = 60
    n_games: int = 1
    team_candidates: tuple[tuple[str, str], ...] = field(default_factory=tuple)


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



def _canonical_league(league: str | None) -> str | None:
    if not league:
        return None
    token = str(league).strip().upper()
    if token in TEAM_ALIAS_GROUPS_BY_LEAGUE:
        return token
    return None



def _parse_window_days(question: str) -> int:
    m = re.search(r"last\s+(\d+)\s+day", question.lower())
    if m:
        return int(m.group(1))
    return 60



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



def _explicit_league_hint(question: str) -> str | None:
    q = _normalize(question)

    nba_signals = bool(
        re.search(r"\bnba\b", q)
        or "basketball" in q
        or "nba finals" in q
        or "larry o brien" in q
        or "larry obrien" in q
    )
    nhl_signals = bool(re.search(r"\bnhl\b", q) or "hockey" in q or "stanley cup" in q)

    if nba_signals and not nhl_signals:
        return "NBA"
    if nhl_signals and not nba_signals:
        return "NHL"
    return None



def _team_candidates(question: str) -> list[tuple[str, str, int]]:
    q = _normalize(question)
    out: list[tuple[str, str, int]] = []

    for pattern, league, team, alias_len in _TEAM_ALIAS_REGEX:
        if pattern.search(q):
            out.append((league, team, alias_len))

    upper = question.upper()
    for league, pattern in TEAM_ABBREV_PATTERN_BY_LEAGUE.items():
        m = pattern.search(upper)
        if not m:
            continue
        abbr = m.group(1).upper()
        abbr = TEAM_ABBREV_ALIASES_BY_LEAGUE.get(league, {}).get(abbr, abbr)
        out.append((league, abbr, len(abbr)))

    deduped: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()
    for cand in out:
        key = (cand[0], cand[1])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)
    return deduped



def _resolve_team(question: str, default_league: str | None) -> tuple[str | None, str | None, tuple[tuple[str, str], ...]]:
    league_hint = _explicit_league_hint(question)
    candidates = _team_candidates(question)
    pair_candidates = tuple((league, team) for league, team, _ in candidates)

    if league_hint:
        hinted = [c for c in candidates if c[0] == league_hint]
        if len(hinted) == 1:
            league, team, _ = hinted[0]
            return team, league, pair_candidates
        if len(hinted) > 1:
            return None, league_hint, pair_candidates

    if not candidates:
        return None, league_hint or _canonical_league(default_league), tuple()

    if len(candidates) == 1:
        league, team, _ = candidates[0]
        return team, league, pair_candidates

    strongest_len = max(c[2] for c in candidates)
    strongest = [c for c in candidates if c[2] == strongest_len]
    if len(strongest) == 1:
        league, team, _ = strongest[0]
        return team, league, pair_candidates

    preferred = _canonical_league(default_league) or "NHL"
    preferred_hits = [c for c in strongest if c[0] == preferred]
    if len(preferred_hits) == 1:
        league, team, _ = preferred_hits[0]
        return team, league, pair_candidates

    if preferred == "NHL" and preferred_hits:
        league, team, _ = preferred_hits[0]
        return team, league, pair_candidates

    return None, preferred, pair_candidates



def _competition_for_question(
    question: str,
    team_league: str | None,
    default_league: str | None,
) -> tuple[str | None, str | None]:
    q = _normalize(question)

    if "stanley cup" in q:
        return "NHL", "Stanley Cup"
    if "nba finals" in q or "larry o brien" in q or "larry obrien" in q:
        return "NBA", "NBA Finals"

    championship_signals = [
        "win the cup",
        "win cup",
        "hoist the cup",
        "lift the cup",
        "win it all",
        "championship",
        "champion",
        "title",
        "finals",
    ]
    if any(s in q for s in championship_signals):
        league = team_league or _explicit_league_hint(question) or _canonical_league(default_league) or "NHL"
        return ("NBA", "NBA Finals") if league == "NBA" else ("NHL", "Stanley Cup")

    return None, None



def _is_league_report_request(question: str) -> bool:
    q = _normalize(question)

    direct_report_phrases = (
        "give me the report",
        "team report",
        "league report",
        "all teams report",
        "full report",
    )
    if any(phrase in q for phrase in direct_report_phrases):
        return True

    has_reportish = any(token in q for token in ("report", "table", "breakdown"))
    has_scope = any(token in q for token in ("all teams", "every team", "by division"))
    has_schedule_context = any(token in q for token in ("next game", "next opponent", "upcoming"))
    return has_reportish and (has_scope or has_schedule_context)


def parse_question(question: str, default_league: str | None = "NHL") -> QueryIntent:
    q = question.lower().strip()
    canonical_default = _canonical_league(default_league) or "NHL"
    league_hint = _explicit_league_hint(question) or canonical_default

    if _is_league_report_request(question):
        return QueryIntent(intent_type="league_report", league=league_hint)

    team, team_league, candidates = _resolve_team(question, default_league=canonical_default)
    n_games = _parse_next_games_count(question)

    if any(k in q for k in ["which model", "best model", "performed best", "leaderboard"]):
        return QueryIntent(intent_type="best_model", league=canonical_default, window_days=_parse_window_days(q))

    competition_league, competition = _competition_for_question(question, team_league=team_league, default_league=canonical_default)
    if competition:
        return QueryIntent(
            intent_type="team_championship",
            team=team,
            league=competition_league,
            competition=competition,
            team_candidates=candidates,
        )

    if not team and candidates:
        return QueryIntent(
            intent_type="clarify_team",
            league=team_league,
            team_candidates=candidates,
        )

    if team:
        if n_games and n_games > 1:
            return QueryIntent(intent_type="team_next_n_games", team=team, league=team_league, n_games=n_games)
        return QueryIntent(intent_type="team_next_game", team=team, league=team_league)

    if n_games and n_games > 1 and any(k in q for k in ["chance", "probability", "odds", "likely", "win", "wins"]):
        return QueryIntent(intent_type="team_next_n_games", team=None, league=canonical_default, n_games=n_games)

    if (n_games == 1 or "next game" in q) and any(k in q for k in ["chance", "probability", "odds", "likely"]):
        return QueryIntent(intent_type="team_next_game", team=None, league=canonical_default)

    return QueryIntent(intent_type="help", league=canonical_default)
