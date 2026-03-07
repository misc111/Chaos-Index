"""Handlers for team-centric forecast questions."""

from __future__ import annotations

import json
import math

from src.query.contracts import Queryable
from src.query.model_reporting import canonicalize_model_probabilities
from src.query.templates import team_multi_game_answer, team_prob_answer


def latest_upcoming_as_of(db: Queryable) -> str | None:
    rows = db.query("SELECT MAX(as_of_utc) AS as_of_utc FROM upcoming_game_forecasts")
    return rows[0]["as_of_utc"] if rows and rows[0]["as_of_utc"] else None


def query_team_upcoming_rows(db: Queryable, team: str, as_of_utc: str, limit: int) -> list[dict]:
    query = """
    WITH team_games AS (
      SELECT *
      FROM upcoming_game_forecasts
      WHERE as_of_utc = ?
        AND home_team = ?
      UNION ALL
      SELECT *
      FROM upcoming_game_forecasts
      WHERE as_of_utc = ?
        AND away_team = ?
    )
    SELECT *
    FROM team_games
    ORDER BY game_date_utc ASC, game_id ASC
    LIMIT ?
    """
    return db.query(query, (as_of_utc, team, as_of_utc, team, int(limit)))


def oriented_game_forecast(row: dict, team: str) -> dict:
    is_home = row["home_team"] == team
    p_home = float(row["ensemble_prob_home_win"])
    p_team = p_home if is_home else 1 - p_home
    opponent = row["away_team"] if is_home else row["home_team"]

    per_model = json.loads(row["per_model_probs_json"]) if row.get("per_model_probs_json") else {}
    per_model = canonicalize_model_probabilities(per_model)
    team_oriented = {}
    for key, value in per_model.items():
        team_oriented[key] = float(value) if is_home else float(1 - float(value))

    return {
        "game_id": row["game_id"],
        "date": row["game_date_utc"],
        "opponent": opponent,
        "is_home": bool(is_home),
        "ensemble_prob_team_win": float(p_team),
        "ensemble_prob_home_win": float(p_home),
        "per_model_probs_team_win": team_oriented,
        "spread": {
            "min": row.get("spread_min"),
            "median": row.get("spread_median"),
            "max": row.get("spread_max"),
            "mean": row.get("spread_mean"),
            "sd": row.get("spread_sd"),
            "iqr": row.get("spread_iqr"),
        },
        "bayes_credible_interval": {
            "low": row.get("bayes_ci_low"),
            "high": row.get("bayes_ci_high"),
        },
        "notes": json.loads(row.get("uncertainty_flags_json") or "{}"),
        "meta": {
            "as_of_utc": row.get("as_of_utc"),
            "snapshot_id": row.get("snapshot_id"),
            "feature_set_version": row.get("feature_set_version"),
            "model_run_id": row.get("model_run_id"),
        },
    }


def bernoulli_sum_distribution(probs: list[float]) -> list[float]:
    dist = [1.0]
    for prob in probs:
        nxt = [0.0] * (len(dist) + 1)
        for k, prob_mass in enumerate(dist):
            nxt[k] += prob_mass * (1 - prob)
            nxt[k + 1] += prob_mass * prob
        dist = nxt
    return dist


def answer_team_next_game(db: Queryable, team: str | None, league: str | None) -> tuple[str, dict]:
    if not team:
        league_msg = f" for {league}" if league else ""
        return f"Team not recognized{league_msg}. Include team name or abbreviation.", {"error": "team_not_recognized", "league": league}

    latest_as_of = latest_upcoming_as_of(db)
    if not latest_as_of:
        return "No upcoming forecasts are currently stored.", {"error": "no_upcoming_forecasts", "league": league}

    rows = query_team_upcoming_rows(db, team=team, as_of_utc=latest_as_of, limit=1)
    if not rows:
        return f"No upcoming forecast currently stored for {team}.", {"error": "no_upcoming_forecast", "team": team, "league": league}

    game = oriented_game_forecast(rows[0], team=team)
    answer = team_prob_answer(team=team, opponent=game["opponent"], date=game["date"], win_prob=game["ensemble_prob_team_win"])
    payload = {
        "intent": "team_next_game",
        "league": league,
        "team": team,
        "opponent": game["opponent"],
        "game_id": game["game_id"],
        "game_date_utc": game["date"],
        "ensemble_prob_team_win": game["ensemble_prob_team_win"],
        "ensemble_prob_home_win": game["ensemble_prob_home_win"],
        "per_model_probs_team_win": game["per_model_probs_team_win"],
        "spread": game["spread"],
        "bayes_credible_interval": game["bayes_credible_interval"],
        "meta": game["meta"],
        "notes": game["notes"],
    }
    return answer, payload


def answer_team_next_n_games(db: Queryable, team: str | None, n_games: int, league: str | None) -> tuple[str, dict]:
    if not team:
        league_msg = f" for {league}" if league else ""
        return f"Team not recognized{league_msg}. Include team name or abbreviation.", {"error": "team_not_recognized", "league": league}

    n = max(1, int(n_games))
    if n == 1:
        return answer_team_next_game(db, team=team, league=league)

    latest_as_of = latest_upcoming_as_of(db)
    if not latest_as_of:
        return "No upcoming forecasts are currently stored.", {"error": "no_upcoming_forecasts", "league": league}

    rows = query_team_upcoming_rows(db, team=team, as_of_utc=latest_as_of, limit=n)
    if not rows:
        return f"No upcoming forecast currently stored for {team}.", {"error": "no_upcoming_forecast", "team": team, "league": league}

    games = [oriented_game_forecast(row, team=team) for row in rows]
    probs = [float(game["ensemble_prob_team_win"]) for game in games]
    prob_win_all = float(math.prod(probs)) if probs else 0.0
    expected_wins = float(sum(probs))
    wins_dist = bernoulli_sum_distribution(probs)

    games_summary = "; ".join([f"{game['date']} vs {game['opponent']} {game['ensemble_prob_team_win']:.1%}" for game in games])
    answer = team_multi_game_answer(
        team=team,
        n_games_requested=n,
        n_games_returned=len(games),
        prob_win_all=prob_win_all,
        expected_wins=expected_wins,
        games_summary=games_summary,
    )
    payload = {
        "intent": "team_next_n_games",
        "league": league,
        "team": team,
        "n_games_requested": n,
        "n_games_returned": len(games),
        "games": games,
        "aggregate": {
            "prob_win_all": prob_win_all,
            "expected_wins": expected_wins,
            "prob_by_total_wins": {str(i): float(p) for i, p in enumerate(wins_dist)},
            "prob_at_least_one_win": float(1 - wins_dist[0]),
        },
        "meta": {"as_of_utc": latest_as_of},
    }
    return answer, payload
