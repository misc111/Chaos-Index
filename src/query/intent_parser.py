"""Intent parsing separated from answer construction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.query.team_aliases import (
    TEAM_ABBREV_ALIASES_BY_LEAGUE,
    TEAM_ABBREV_PATTERN_BY_LEAGUE,
    TEAM_ALIAS_GROUPS_BY_LEAGUE,
    TEAM_ALIAS_REGEX,
    canonical_league,
)


@dataclass
class QueryIntent:
    intent_type: str
    team: str | None = None
    league: str | None = None
    competition: str | None = None
    window_days: int = 60
    n_games: int = 1
    history_period: str | None = None
    include_games: bool = False
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


def normalize_question(question: str) -> str:
    lowered = question.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def parse_window_days(question: str) -> int:
    match = re.search(r"last\s+(\d+)\s+day", question.lower())
    if match:
        return int(match.group(1))
    return 60


def _token_to_int(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


def parse_next_games_count(question: str) -> int | None:
    normalized = normalize_question(question)

    match = re.search(r"\bnext\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|couple|few)\s+games?\b", normalized)
    if match:
        n = _token_to_int(match.group(1))
        return max(1, min(int(n), 10)) if n is not None else None

    match = re.search(r"\bnext\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|couple|few)\b", normalized)
    if match:
        n = _token_to_int(match.group(1))
        return max(1, min(int(n), 10)) if n is not None else None

    if re.search(r"\bnext\s+game\b", normalized) or "tonight" in normalized or "tomorrow" in normalized:
        return 1
    if re.search(r"\bnext\s+games\b", normalized):
        return 3
    return None


def explicit_league_hint(question: str) -> str | None:
    normalized = normalize_question(question)

    nba_signals = bool(
        re.search(r"\bnba\b", normalized)
        or "basketball" in normalized
        or "nba finals" in normalized
        or "larry o brien" in normalized
        or "larry obrien" in normalized
    )
    nhl_signals = bool(re.search(r"\bnhl\b", normalized) or "hockey" in normalized or "stanley cup" in normalized)

    if nba_signals and not nhl_signals:
        return "NBA"
    if nhl_signals and not nba_signals:
        return "NHL"
    return None


def team_candidates(question: str) -> list[tuple[str, str, int]]:
    normalized = normalize_question(question)
    out: list[tuple[str, str, int]] = []

    for pattern, league, team, alias_len in TEAM_ALIAS_REGEX:
        if pattern.search(normalized):
            out.append((league, team, alias_len))

    upper = question.upper()
    for league, pattern in TEAM_ABBREV_PATTERN_BY_LEAGUE.items():
        match = pattern.search(upper)
        if not match:
            continue
        abbr = match.group(1).upper()
        abbr = TEAM_ABBREV_ALIASES_BY_LEAGUE.get(league, {}).get(abbr, abbr)
        out.append((league, abbr, len(abbr)))

    deduped: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in out:
        key = (candidate[0], candidate[1])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def resolve_team(question: str, default_league: str | None) -> tuple[str | None, str | None, tuple[tuple[str, str], ...]]:
    league_hint = explicit_league_hint(question)
    candidates = team_candidates(question)
    pair_candidates = tuple((league, team) for league, team, _ in candidates)

    if league_hint:
        hinted = [c for c in candidates if c[0] == league_hint]
        if len(hinted) == 1:
            league, team, _ = hinted[0]
            return team, league, pair_candidates
        if len(hinted) > 1:
            return None, league_hint, pair_candidates

    if not candidates:
        return None, league_hint or canonical_league(default_league), tuple()
    if len(candidates) == 1:
        league, team, _ = candidates[0]
        return team, league, pair_candidates

    strongest_len = max(c[2] for c in candidates)
    strongest = [c for c in candidates if c[2] == strongest_len]
    if len(strongest) == 1:
        league, team, _ = strongest[0]
        return team, league, pair_candidates

    preferred = canonical_league(default_league) or "NBA"
    preferred_hits = [c for c in strongest if c[0] == preferred]
    if len(preferred_hits) == 1:
        league, team, _ = preferred_hits[0]
        return team, league, pair_candidates
    if preferred_hits:
        league, team, _ = preferred_hits[0]
        return team, league, pair_candidates
    return None, preferred, pair_candidates


def competition_for_question(
    question: str,
    team_league: str | None,
    default_league: str | None,
) -> tuple[str | None, str | None]:
    normalized = normalize_question(question)

    if "stanley cup" in normalized:
        return "NHL", "Stanley Cup"
    if "nba finals" in normalized or "larry o brien" in normalized or "larry obrien" in normalized:
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
    if any(signal in normalized for signal in championship_signals):
        league = team_league or explicit_league_hint(question) or canonical_league(default_league) or "NBA"
        return ("NBA", "NBA Finals") if league == "NBA" else ("NHL", "Stanley Cup")

    return None, None


def is_league_report_request(question: str) -> bool:
    normalized = normalize_question(question)
    direct_report_phrases = ("give me the report", "team report", "league report", "all teams report", "full report")
    if any(phrase in normalized for phrase in direct_report_phrases):
        return True

    has_reportish = any(token in normalized for token in ("report", "table", "breakdown"))
    has_scope = any(token in normalized for token in ("all teams", "every team", "by division"))
    has_schedule_context = any(token in normalized for token in ("next game", "next opponent", "upcoming"))
    return has_reportish and (has_scope or has_schedule_context)


def _has_bet_history_time_scope(normalized: str) -> bool:
    return any(
        term in normalized
        for term in (
            "last night",
            "yesterday",
            "since the beginning",
            "since beginning",
            "since the start",
            "since start",
            "since the beginning of tracking",
            "since the beginning of the tracking",
            "tracking",
            "all time",
            "cumulative",
            "to date",
            "so far",
        )
    )


def _is_casual_last_night_bet_recap(normalized: str) -> bool:
    has_last_night_scope = "last night" in normalized or "yesterday" in normalized
    if not has_last_night_scope:
        return False

    direct_phrases = (
        "how did i do",
        "how d i do",
        "howd i do",
        "how did my bets do",
        "how d my bets do",
        "howd my bets do",
        "recap my bets",
        "bet recap",
        "betting recap",
        "last night recap",
        "yesterday recap",
        "what happened with my bets",
    )
    if any(term in normalized for term in direct_phrases):
        return True

    has_bet_context = any(term in normalized for term in ("bet", "bets", "betting", "wager", "wagers", "pick", "picks"))
    has_recap_context = any(term in normalized for term in ("recap", "summary", "breakdown", "results", "result"))
    return has_bet_context and has_recap_context


def is_bet_history_request(question: str) -> bool:
    normalized = normalize_question(question)
    money_terms = (
        "money i won or lost",
        "money did i win",
        "money did i lose",
        "won or lost",
        "win lose",
        "bets",
        "net profit",
        "net profits",
        "net loss",
        "net losses",
        "cumulative net",
        "profit",
        "profits",
        "pnl",
        "risked",
        "amount bet",
        "bet history",
        "betting history",
        "won on",
        "lost on",
        "bet on",
        "didn t bet",
        "didnt bet",
        "no bet",
    )
    has_money_term = any(term in normalized for term in money_terms)
    has_time_term = _has_bet_history_time_scope(normalized)
    has_history_scope = any(
        term in normalized
        for term in ("bet history", "betting history", "specific games", "which games", "those bets were related")
    )
    return (has_money_term and (has_time_term or has_history_scope)) or _is_casual_last_night_bet_recap(normalized)


def parse_bet_history_period(question: str) -> str:
    normalized = normalize_question(question)
    if "last night" in normalized or "yesterday" in normalized:
        return "yesterday"
    return "all_time"


def parse_bet_history_include_games(question: str) -> bool:
    normalized = normalize_question(question)
    detail_terms = (
        "by game",
        "by games",
        "per game",
        "game by game",
        "each game",
        "specific games",
        "won on",
        "lost on",
        "bet on",
        "didn t bet",
        "didnt bet",
        "no bet",
        "why we bet",
        "why did we bet",
        "why we didn t bet",
        "why we didnt bet",
    )
    if any(term in normalized for term in detail_terms):
        return True

    has_last_night_scope = "last night" in normalized or "yesterday" in normalized
    has_win_loss_summary = any(
        term in normalized
        for term in ("money i won or lost", "money did i win", "money did i lose", "won or lost", "win lose")
    )
    return (has_last_night_scope and has_win_loss_summary) or _is_casual_last_night_bet_recap(normalized)


def parse_question(question: str, default_league: str | None = "NBA") -> QueryIntent:
    lowered = question.lower().strip()
    canonical_default = canonical_league(default_league) or "NBA"
    league_hint = explicit_league_hint(question) or canonical_default

    if is_bet_history_request(question):
        return QueryIntent(
            intent_type="bet_history_summary",
            league=league_hint,
            history_period=parse_bet_history_period(question),
            include_games=parse_bet_history_include_games(question),
        )

    if is_league_report_request(question):
        return QueryIntent(intent_type="league_report", league=league_hint)

    team, team_league, candidates = resolve_team(question, default_league=canonical_default)
    n_games = parse_next_games_count(question)

    if any(k in lowered for k in ["which model", "best model", "performed best", "leaderboard"]):
        return QueryIntent(intent_type="best_model", league=canonical_default, window_days=parse_window_days(lowered))

    competition_league, competition = competition_for_question(question, team_league=team_league, default_league=canonical_default)
    if competition:
        return QueryIntent(
            intent_type="team_championship",
            team=team,
            league=competition_league,
            competition=competition,
            team_candidates=candidates,
        )

    if not team and candidates:
        return QueryIntent(intent_type="clarify_team", league=team_league, team_candidates=candidates)

    if team:
        if n_games and n_games > 1:
            return QueryIntent(intent_type="team_next_n_games", team=team, league=team_league, n_games=n_games)
        return QueryIntent(intent_type="team_next_game", team=team, league=team_league)

    if n_games and n_games > 1 and any(k in lowered for k in ["chance", "probability", "odds", "likely", "win", "wins"]):
        return QueryIntent(intent_type="team_next_n_games", team=None, league=canonical_default, n_games=n_games)
    if (n_games == 1 or "next game" in lowered) and any(k in lowered for k in ["chance", "probability", "odds", "likely"]):
        return QueryIntent(intent_type="team_next_game", team=None, league=canonical_default)
    return QueryIntent(intent_type="help", league=canonical_default)
