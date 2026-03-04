from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import numpy as np

from src.common.config import load_config
from src.query.parse import TEAM_ALIAS_GROUPS, parse_question
from src.query.templates import (
    best_model_answer,
    help_answer,
    stanley_cup_answer,
    team_multi_game_answer,
    team_prob_answer,
)
from src.storage.db import Database


def _latest_upcoming_as_of(db: Database) -> str | None:
    rows = db.query("SELECT MAX(as_of_utc) AS as_of_utc FROM upcoming_game_forecasts")
    return rows[0]["as_of_utc"] if rows and rows[0]["as_of_utc"] else None


def _query_team_upcoming_rows(db: Database, team: str, as_of_utc: str, limit: int) -> list[dict]:
    q = """
    SELECT *
    FROM upcoming_game_forecasts
    WHERE as_of_utc = ?
      AND (home_team = ? OR away_team = ?)
    ORDER BY game_date_utc ASC, game_id ASC
    LIMIT ?
    """
    return db.query(q, (as_of_utc, team, team, int(limit)))


def _oriented_game_forecast(row: dict, team: str) -> dict:
    is_home = row["home_team"] == team
    p_home = float(row["ensemble_prob_home_win"])
    p_team = p_home if is_home else 1 - p_home
    opp = row["away_team"] if is_home else row["home_team"]

    per_model = json.loads(row["per_model_probs_json"]) if row.get("per_model_probs_json") else {}
    team_oriented = {}
    for k, v in per_model.items():
        team_oriented[k] = float(v) if is_home else float(1 - float(v))

    return {
        "game_id": row["game_id"],
        "date": row["game_date_utc"],
        "opponent": opp,
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


def _bernoulli_sum_distribution(probs: list[float]) -> list[float]:
    dist = [1.0]
    for p in probs:
        nxt = [0.0] * (len(dist) + 1)
        for k, prob_mass in enumerate(dist):
            nxt[k] += prob_mass * (1 - p)
            nxt[k + 1] += prob_mass * p
        dist = nxt
    return dist


def _answer_team_next_game(db: Database, team: str | None) -> tuple[str, dict]:
    if not team:
        return "Team not recognized. Include team name or abbreviation.", {"error": "team_not_recognized"}

    latest_as_of = _latest_upcoming_as_of(db)
    if not latest_as_of:
        return "No upcoming forecasts are currently stored.", {"error": "no_upcoming_forecasts"}

    rows = _query_team_upcoming_rows(db, team=team, as_of_utc=latest_as_of, limit=1)
    if not rows:
        return f"No upcoming forecast currently stored for {team}.", {"error": "no_upcoming_forecast", "team": team}

    game = _oriented_game_forecast(rows[0], team=team)
    answer = team_prob_answer(team=team, opponent=game["opponent"], date=game["date"], win_prob=game["ensemble_prob_team_win"])
    payload = {
        "intent": "team_next_game",
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


def _answer_team_next_n_games(db: Database, team: str | None, n_games: int) -> tuple[str, dict]:
    if not team:
        return "Team not recognized. Include team name or abbreviation.", {"error": "team_not_recognized"}

    n = max(1, int(n_games))
    if n == 1:
        return _answer_team_next_game(db, team=team)

    latest_as_of = _latest_upcoming_as_of(db)
    if not latest_as_of:
        return "No upcoming forecasts are currently stored.", {"error": "no_upcoming_forecasts"}

    rows = _query_team_upcoming_rows(db, team=team, as_of_utc=latest_as_of, limit=n)
    if not rows:
        return f"No upcoming forecast currently stored for {team}.", {"error": "no_upcoming_forecast", "team": team}

    games = [_oriented_game_forecast(row, team=team) for row in rows]
    probs = [float(g["ensemble_prob_team_win"]) for g in games]
    prob_win_all = float(np.prod(probs)) if probs else 0.0
    expected_wins = float(np.sum(probs))
    wins_dist = _bernoulli_sum_distribution(probs)

    games_summary = "; ".join([f"{g['date']} vs {g['opponent']} {g['ensemble_prob_team_win']:.1%}" for g in games])
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
        "meta": {
            "as_of_utc": latest_as_of,
        },
    }
    return answer, payload


def _cup_probs_from_win_rates(win_rates: np.ndarray) -> np.ndarray:
    clipped = np.clip(win_rates, 0.15, 0.85)
    strength = np.log(clipped / (1 - clipped))
    n_teams = int(strength.shape[0])
    playoff_slots = min(16, n_teams)
    cutoff = np.sort(strength)[::-1][playoff_slots - 1] if playoff_slots > 0 else 0.0

    # Smooth playoff-qualification proxy, then strength-weighted champion share.
    qual_prob = 1.0 / (1.0 + np.exp(-(strength - cutoff) / 0.30))
    raw = qual_prob * np.exp(1.75 * strength)
    denom = float(np.sum(raw))
    if denom <= 0:
        return np.full(n_teams, 1.0 / max(n_teams, 1), dtype=float)
    return raw / denom


def _answer_team_stanley_cup(db: Database, team: str | None) -> tuple[str, dict]:
    if not team:
        return "Team not recognized. Include the NHL team for the Stanley Cup question.", {"error": "team_not_recognized"}

    season_rows = db.query("SELECT MAX(season) AS season FROM results")
    season = season_rows[0]["season"] if season_rows and season_rows[0]["season"] is not None else None
    if season is None:
        return "Cannot estimate Stanley Cup probability yet: no finalized results are stored.", {"error": "no_results_data"}

    as_of_rows = db.query("SELECT MAX(game_date_utc) AS as_of_date FROM results WHERE season = ?", (season,))
    as_of_date = as_of_rows[0]["as_of_date"] if as_of_rows and as_of_rows[0]["as_of_date"] else datetime.now(timezone.utc).date().isoformat()

    allowed_teams = set(TEAM_ALIAS_GROUPS.keys())
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

    latest_as_of = _latest_upcoming_as_of(db)
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
            f"Cannot estimate Stanley Cup probability for {team}: team not found in current-season database snapshot.",
            {"error": "team_not_found_in_snapshot", "team": team, "season": season},
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
    wins = np.array([records.get(t, {}).get("wins", 0.0) for t in teams], dtype=float)
    games = np.array([records.get(t, {}).get("games", 0.0) for t in teams], dtype=float)
    losses = np.maximum(games - wins, 0.0)

    alpha = 1.0 + wins
    beta = 1.0 + losses
    mean_win_rates = alpha / (alpha + beta)
    cup_probs = _cup_probs_from_win_rates(mean_win_rates)

    team_idx = teams.index(team)
    point_estimate = float(cup_probs[team_idx])

    rng = np.random.default_rng(42)
    n_sims = 2000
    draws = rng.beta(alpha, beta, size=(n_sims, len(teams)))
    sampled = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        sampled[i] = float(_cup_probs_from_win_rates(draws[i])[team_idx])
    low_90, high_90 = np.quantile(sampled, [0.05, 0.95]).tolist()

    rank_order = np.argsort(cup_probs)[::-1]
    rank_lookup = {int(idx): rank + 1 for rank, idx in enumerate(rank_order)}
    rank = int(rank_lookup[team_idx])
    top_teams = [{"team": teams[int(idx)], "stanley_cup_prob": float(cup_probs[int(idx)])} for idx in rank_order[:5]]

    answer = stanley_cup_answer(
        team=team,
        cup_prob=point_estimate,
        low_90=float(low_90),
        high_90=float(high_90),
        as_of_date=str(as_of_date),
    )
    payload = {
        "intent": "team_stanley_cup",
        "team": team,
        "season": int(season),
        "stanley_cup_prob": point_estimate,
        "interval_90": {
            "low": float(low_90),
            "high": float(high_90),
        },
        "rank_by_estimated_cup_prob": rank,
        "standings_proxy": {
            "games_played": int(games[team_idx]),
            "wins": int(wins[team_idx]),
            "losses": int(losses[team_idx]),
            "posterior_mean_win_rate": float(mean_win_rates[team_idx]),
        },
        "top_teams": top_teams,
        "methodology": {
            "type": "heuristic_results_based",
            "summary": (
                "Uses current-season win/loss results with Beta shrinkage, converts to strength scores, "
                "applies a smooth playoff-qualification proxy, then normalizes championship shares."
            ),
            "monte_carlo_draws": n_sims,
            "as_of_date": as_of_date,
            "upcoming_snapshot_as_of_utc": latest_as_of,
        },
        "notes": {
            "caveat": "High-variance estimate. This is not a full playoff bracket simulator.",
        },
    }
    return answer, payload


def _answer_best_model(db: Database, window_days: int) -> tuple[str, dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).date().isoformat()
    q = """
    WITH ranked AS (
      SELECT
        score_id,
        game_id,
        model_name,
        log_loss,
        brier,
        accuracy,
        game_date_utc,
        scored_at_utc,
        ROW_NUMBER() OVER (
          PARTITION BY game_id, model_name
          ORDER BY DATETIME(scored_at_utc) DESC, score_id DESC
        ) AS rn
      FROM model_scores
      WHERE DATE(game_date_utc) >= DATE(?)
    )
    SELECT model_name,
           AVG(log_loss) AS log_loss,
           AVG(brier) AS brier,
           AVG(accuracy) AS accuracy,
           COUNT(*) AS n_games
    FROM ranked
    WHERE rn = 1
    GROUP BY model_name
    HAVING COUNT(*) >= 1
    ORDER BY AVG(log_loss) ASC
    """
    rows = db.query(q, (cutoff,))
    if not rows:
        return f"No model_scores rows found in the last {window_days} days.", {"error": "no_scores", "window_days": window_days}

    best = rows[0]
    answer = best_model_answer(model=best["model_name"], window_days=window_days, log_loss=float(best["log_loss"]))
    payload = {
        "intent": "best_model",
        "window_days": window_days,
        "best_model": best,
        "leaderboard": rows,
        "meta": {
            "evaluated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "cutoff_date": cutoff,
        },
    }
    return answer, payload


def answer_question(db: Database, question: str) -> tuple[str, dict]:
    intent = parse_question(question)

    if intent.intent_type == "team_next_game":
        return _answer_team_next_game(db, intent.team)

    if intent.intent_type == "team_next_n_games":
        return _answer_team_next_n_games(db, intent.team, intent.n_games)

    if intent.intent_type == "team_stanley_cup":
        return _answer_team_stanley_cup(db, intent.team)

    if intent.intent_type == "best_model":
        return _answer_best_model(db, intent.window_days)

    return help_answer(), {"intent": "help"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Local deterministic NHL query command")
    parser.add_argument("--config", type=str, default="configs/nhl.yaml")
    parser.add_argument("--question", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    db = Database(cfg.paths.db_path)

    answer, payload = answer_question(db, args.question)
    print(answer)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
