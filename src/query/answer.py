from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.common.config import load_config
from src.query.parse import parse_question
from src.query.templates import best_model_answer, help_answer, team_prob_answer
from src.storage.db import Database



def _answer_team_next_game(db: Database, team: str | None) -> tuple[str, dict]:
    if not team:
        return "Team not recognized. Include team name or abbreviation.", {"error": "team_not_recognized"}

    latest_rows = db.query("SELECT MAX(as_of_utc) AS as_of_utc FROM upcoming_game_forecasts")
    latest_as_of = latest_rows[0]["as_of_utc"] if latest_rows and latest_rows[0]["as_of_utc"] else None
    if not latest_as_of:
        return "No upcoming forecasts are currently stored.", {"error": "no_upcoming_forecasts"}

    q = """
    SELECT *
    FROM upcoming_game_forecasts
    WHERE as_of_utc = ?
      AND (home_team = ? OR away_team = ?)
    ORDER BY game_date_utc ASC
    LIMIT 1
    """
    rows = db.query(q, (latest_as_of, team, team))
    if not rows:
        return f"No upcoming forecast currently stored for {team}.", {"error": "no_upcoming_forecast", "team": team}

    row = rows[0]
    is_home = row["home_team"] == team
    p_home = float(row["ensemble_prob_home_win"])
    p_team = p_home if is_home else 1 - p_home
    opp = row["away_team"] if is_home else row["home_team"]

    per_model = json.loads(row["per_model_probs_json"]) if row.get("per_model_probs_json") else {}
    team_oriented = {}
    for k, v in per_model.items():
        team_oriented[k] = float(v) if is_home else float(1 - float(v))

    answer = team_prob_answer(team=team, opponent=opp, date=row["game_date_utc"], win_prob=p_team)
    payload = {
        "intent": "team_next_game",
        "team": team,
        "opponent": opp,
        "game_id": row["game_id"],
        "game_date_utc": row["game_date_utc"],
        "ensemble_prob_team_win": p_team,
        "ensemble_prob_home_win": p_home,
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
        "meta": {
            "as_of_utc": row.get("as_of_utc"),
            "snapshot_id": row.get("snapshot_id"),
            "feature_set_version": row.get("feature_set_version"),
            "model_run_id": row.get("model_run_id"),
        },
        "notes": json.loads(row.get("uncertainty_flags_json") or "{}"),
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
