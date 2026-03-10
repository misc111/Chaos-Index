"""Thin query responder over intent-specific handlers."""

from __future__ import annotations

import argparse
import json

from src.common.config import load_config
from src.league_registry import get_league_metadata
from src.query.bet_history_handlers import answer_bet_history_summary
from src.query.championship_estimators import answer_team_championship, competition_name_for_league
from src.query.contracts import ConnectionQueryAdapter, Queryable
from src.query.intent_parser import parse_question
from src.query.report_handlers import answer_best_model, answer_league_report
from src.query.team_handlers import answer_team_next_game, answer_team_next_n_games
from src.query.templates import help_answer
from src.storage.db import Database


def clarify_team_answer(team_candidates: tuple[tuple[str, str], ...]) -> tuple[str, dict]:
    if not team_candidates:
        return "Team not recognized. Include team name or abbreviation.", {"error": "team_not_recognized"}

    options = [f"{league}:{team}" for league, team in team_candidates]
    answer = (
        "That team wording matches multiple teams across leagues. "
        "Please specify league in your question (for example: 'NHL Boston' or 'NBA Boston')."
    )
    return answer, {"intent": "clarify_team", "team_candidates": options}


def answer_question(db: Queryable, question: str, default_league: str | None = "NBA") -> tuple[str, dict]:
    intent = parse_question(question, default_league=default_league)

    if intent.intent_type == "bet_history_summary":
        league = intent.league or (default_league or "NBA")
        return answer_bet_history_summary(
            db,
            league=league,
            history_period=intent.history_period or "all_time",
            include_games=bool(intent.include_games),
        )
    if intent.intent_type == "team_next_game":
        return answer_team_next_game(db, intent.team, intent.league)
    if intent.intent_type == "team_next_n_games":
        return answer_team_next_n_games(db, intent.team, intent.n_games, intent.league)
    if intent.intent_type == "team_championship":
        league = intent.league or (default_league or "NBA")
        competition = intent.competition or competition_name_for_league(league)
        return answer_team_championship(db, intent.team, league=league, competition=competition)
    if intent.intent_type == "league_report":
        return answer_league_report(db, league=intent.league or default_league)
    if intent.intent_type == "best_model":
        return answer_best_model(db, intent.window_days)
    if intent.intent_type == "clarify_team":
        return clarify_team_answer(intent.team_candidates)
    return help_answer(), {"intent": "help"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Local deterministic sports query command")
    parser.add_argument("--config", type=str, default="configs/nba.yaml")
    parser.add_argument("--league", type=str, choices=["NHL", "NBA", "NCAAM"], default=None)
    parser.add_argument("--question", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    default_league = args.league or get_league_metadata(cfg.data.league).code

    db = Database(cfg.paths.db_path)
    with db.connect() as conn:
        session = ConnectionQueryAdapter(conn)
        answer, payload = answer_question(session, args.question, default_league=default_league)
    print(answer)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
