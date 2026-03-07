"""League-specific championship heuristics kept outside shared responders."""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone

from src.league_registry import get_league_metadata
from src.query.contracts import Queryable
from src.query.team_aliases import TEAM_ALIAS_GROUPS_BY_LEAGUE
from src.query.team_handlers import latest_upcoming_as_of
from src.query.templates import championship_answer


def probability_key_for_league(league: str) -> str:
    return get_league_metadata(league).championship_probability_key


def competition_name_for_league(league: str) -> str:
    return get_league_metadata(league).championship_name


def championship_probs_from_win_rates(win_rates: list[float]) -> list[float]:
    clipped = [max(0.15, min(0.85, float(w))) for w in win_rates]
    strength = [math.log(w / (1 - w)) for w in clipped]
    n_teams = len(strength)
    playoff_slots = min(16, n_teams)
    cutoff = sorted(strength, reverse=True)[playoff_slots - 1] if playoff_slots > 0 else 0.0

    raw = []
    for strength_value in strength:
        qual_prob = 1.0 / (1.0 + math.exp(-(strength_value - cutoff) / 0.30))
        raw.append(qual_prob * math.exp(1.75 * strength_value))
    denom = float(sum(raw))
    if denom <= 0:
        if n_teams == 0:
            return []
        return [1.0 / n_teams] * n_teams
    return [r / denom for r in raw]


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float(sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight)


def answer_team_championship(
    db: Queryable,
    team: str | None,
    league: str,
    competition: str,
) -> tuple[str, dict]:
    probability_key = probability_key_for_league(league)

    if not team:
        return (
            f"Team not recognized. Include the {league} team for the {competition} question.",
            {"error": "team_not_recognized", "league": league, "competition": competition},
        )

    season_rows = db.query("SELECT MAX(season) AS season FROM results")
    season = season_rows[0]["season"] if season_rows and season_rows[0]["season"] is not None else None
    if season is None:
        return (
            f"Cannot estimate {competition} probability yet: no finalized results are stored.",
            {"error": "no_results_data", "league": league, "competition": competition},
        )

    as_of_rows = db.query("SELECT MAX(game_date_utc) AS as_of_date FROM results WHERE season = ?", (season,))
    as_of_date = as_of_rows[0]["as_of_date"] if as_of_rows and as_of_rows[0]["as_of_date"] else datetime.now(timezone.utc).date().isoformat()

    allowed_teams = set(TEAM_ALIAS_GROUPS_BY_LEAGUE.get(league, {}).keys())
    team_set: set[str] = set()
    league_rows = db.query(
        """
        SELECT team
        FROM (
          SELECT home_team AS team FROM results WHERE season = ?
          UNION
          SELECT away_team AS team FROM results WHERE season = ?
        )
        WHERE team IS NOT NULL
        """,
        (season, season),
    )
    for row in league_rows:
        if row.get("team"):
            team_code = str(row["team"])
            if team_code in allowed_teams:
                team_set.add(team_code)

    latest_as_of = latest_upcoming_as_of(db)
    if latest_as_of:
        upcoming_rows = db.query(
            """
            SELECT team
            FROM (
              SELECT home_team AS team FROM upcoming_game_forecasts WHERE as_of_utc = ?
              UNION
              SELECT away_team AS team FROM upcoming_game_forecasts WHERE as_of_utc = ?
            )
            WHERE team IS NOT NULL
            """,
            (latest_as_of, latest_as_of),
        )
        for row in upcoming_rows:
            if row.get("team"):
                team_code = str(row["team"])
                if team_code in allowed_teams:
                    team_set.add(team_code)

    if team not in team_set:
        return (
            f"Cannot estimate {competition} probability for {team}: team not found in current-season {league} database snapshot.",
            {
                "error": "team_not_found_in_snapshot",
                "team": team,
                "league": league,
                "competition": competition,
                "season": season,
            },
        )

    record_rows = db.query(
        """
        WITH team_games AS (
          SELECT home_team AS team, CASE WHEN home_win = 1 THEN 1 ELSE 0 END AS win
          FROM results
          WHERE season = ?
          UNION ALL
          SELECT away_team AS team, CASE WHEN home_win = 0 THEN 1 ELSE 0 END AS win
          FROM results
          WHERE season = ?
        )
        SELECT team, SUM(win) AS wins, COUNT(*) AS games
        FROM team_games
        GROUP BY team
        """,
        (season, season),
    )
    records = {
        str(r["team"]): {"wins": float(r["wins"]), "games": float(r["games"])}
        for r in record_rows
        if r.get("team") and str(r["team"]) in allowed_teams
    }

    teams = sorted(team_set)
    wins = [float(records.get(t, {}).get("wins", 0.0)) for t in teams]
    games = [float(records.get(t, {}).get("games", 0.0)) for t in teams]
    losses = [max(g - w, 0.0) for g, w in zip(games, wins)]

    alpha = [1.0 + w for w in wins]
    beta = [1.0 + l for l in losses]
    mean_win_rates = [a / (a + b) for a, b in zip(alpha, beta)]
    championship_probs = championship_probs_from_win_rates(mean_win_rates)

    team_idx = teams.index(team)
    point_estimate = float(championship_probs[team_idx])

    rng = random.Random(42)
    n_sims = 2000
    sampled: list[float] = []
    for _ in range(n_sims):
        draw = [rng.betavariate(a, b) for a, b in zip(alpha, beta)]
        sampled.append(float(championship_probs_from_win_rates(draw)[team_idx]))
    sampled.sort()
    low_90 = quantile(sampled, 0.05)
    high_90 = quantile(sampled, 0.95)

    rank_order = sorted(range(len(championship_probs)), key=lambda idx: championship_probs[idx], reverse=True)
    rank_lookup = {idx: rank + 1 for rank, idx in enumerate(rank_order)}
    rank = int(rank_lookup[team_idx])
    top_teams = [
        {
            "team": teams[idx],
            "championship_prob": float(championship_probs[idx]),
            probability_key: float(championship_probs[idx]),
        }
        for idx in rank_order[:5]
    ]

    answer = championship_answer(
        team=team,
        competition=competition,
        championship_prob=point_estimate,
        low_90=float(low_90),
        high_90=float(high_90),
        as_of_date=str(as_of_date),
    )
    payload = {
        "intent": "team_championship",
        "league": league,
        "competition": competition,
        "team": team,
        "season": int(season),
        "championship_prob": point_estimate,
        probability_key: point_estimate,
        "interval_90": {"low": float(low_90), "high": float(high_90)},
        "rank_by_estimated_championship_prob": rank,
        "standings_proxy": {
            "games_played": int(games[team_idx]),
            "wins": int(wins[team_idx]),
            "losses": int(losses[team_idx]),
            "posterior_mean_win_rate": float(mean_win_rates[team_idx]),
        },
        "top_teams": top_teams,
        "methodology": {
            "type": "heuristic_results_based",
            "summary": "Uses current-season win/loss results with Beta shrinkage, converts to strength scores, applies a smooth playoff-qualification proxy, then normalizes championship shares.",
            "monte_carlo_draws": n_sims,
            "as_of_date": as_of_date,
            "upcoming_snapshot_as_of_utc": latest_as_of,
        },
        "notes": {"caveat": "High-variance estimate. This is not a full playoff bracket simulator."},
    }
    return answer, payload
