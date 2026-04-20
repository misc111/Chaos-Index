from __future__ import annotations

from collections.abc import Callable

from src.training.contracts import LassoCredibilityMetadata

FeatureMatcher = Callable[[str], bool]


def _lower_feature_name(feature: str) -> str:
    return str(feature or "").strip().lower()


def _contains_any(token: str, needles: tuple[str, ...]) -> bool:
    return any(needle in token for needle in needles)


def _startswith_any(token: str, prefixes: tuple[str, ...]) -> bool:
    return any(token.startswith(prefix) for prefix in prefixes)


def _ordered_rules_for(
    league: str,
    *,
    credibility: LassoCredibilityMetadata | None,
) -> list[tuple[str, FeatureMatcher]]:
    complement_column = _lower_feature_name(credibility.complement_column) if credibility is not None else ""
    market_matcher = lambda token: token == complement_column or _contains_any(  # noqa: E731
        token,
        ("market", "vig", "implied", "odds", "price", "spread", "total", "offset"),
    )

    league_code = str(league or "").strip().upper()
    if league_code == "MLB":
        return [
            ("market_credibility_block", market_matcher),
            ("starting_pitcher_block", lambda token: _contains_any(token, ("starter", "starting_pitcher", "pitcher_hand"))),
            ("bullpen_block", lambda token: _contains_any(token, ("bullpen", "pitcher_out_count"))),
            ("lineup_block", lambda token: _contains_any(token, ("lineup", "position_player", "slugging"))),
            (
                "park_weather_block",
                lambda token: _contains_any(token, ("park", "weather", "wind", "temperature", "humidity", "umpire")),
            ),
            (
                "schedule_travel_block",
                lambda token: _contains_any(
                    token,
                    ("travel", "rest", "series", "day_game", "days_into_season", "season_game", "home_field"),
                ),
            ),
            ("rating_block", lambda token: _startswith_any(token, ("elo_", "dyn_")) or "rating" in token),
            (
                "offense_form_block",
                lambda token: _contains_any(
                    token,
                    ("form_", "run_diff", "win_rate", "hits", "walks", "strikeouts", "home_runs", "total_bases"),
                ),
            ),
        ]

    if league_code == "NBA":
        return [
            ("market_credibility_block", market_matcher),
            ("availability_block", lambda token: _contains_any(token, ("availability", "absence", "roster_depth"))),
            ("shot_profile_block", lambda token: _contains_any(token, ("shot", "scoring_efficiency"))),
            ("discipline_block", lambda token: _contains_any(token, ("discipline", "foul", "free_throw"))),
            ("travel_block", lambda token: _contains_any(token, ("travel", "rest", "tz_"))),
            ("arena_block", lambda token: _contains_any(token, ("arena", "altitude"))),
            ("rating_block", lambda token: _startswith_any(token, ("elo_", "dyn_")) or "rating" in token),
        ]

    return [
        ("market_credibility_block", market_matcher),
        ("goalie_block", lambda token: _contains_any(token, ("goalie", "starter_save_pct"))),
        ("xg_block", lambda token: "xg" in token),
        ("special_teams_block", lambda token: _contains_any(token, ("special", "penalty", "pp_", "pk_"))),
        ("travel_block", lambda token: _contains_any(token, ("travel", "rest", "tz_"))),
        ("lineup_block", lambda token: _contains_any(token, ("lineup", "roster", "man_games"))),
        ("rink_block", lambda token: "rink" in token),
        ("rating_block", lambda token: _startswith_any(token, ("elo_", "dyn_")) or "rating" in token),
    ]


def build_feature_blocks(
    league: str,
    feature_columns: list[str],
    *,
    credibility: LassoCredibilityMetadata | None = None,
) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    leftovers: list[str] = []
    rules = _ordered_rules_for(league, credibility=credibility)

    for feature in feature_columns:
        token = _lower_feature_name(feature)
        matched = False
        for block_name, matcher in rules:
            if matcher(token):
                blocks.setdefault(block_name, []).append(feature)
                matched = True
                break
        if not matched:
            leftovers.append(feature)

    if leftovers:
        blocks["other_context_block"] = leftovers
    return {block_name: cols for block_name, cols in blocks.items() if cols}
