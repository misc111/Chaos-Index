"""Handlers for report-style query responses."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from src.query.contracts import Queryable
from src.query.model_reporting import MODEL_TRUST_NOTES, ordered_model_names
from src.query.team_aliases import TEAM_ALIAS_GROUPS_BY_LEAGUE, canonical_team_code
from src.query.team_handlers import latest_upcoming_as_of, oriented_game_forecast
from src.query.templates import best_model_answer


def available_team_leagues(db: Queryable) -> set[str]:
    try:
        rows = db.query("SELECT DISTINCT league FROM teams WHERE league IS NOT NULL")
    except Exception:
        return set()
    return {str(row.get("league") or "").strip().upper() for row in rows if str(row.get("league") or "").strip()}


def latest_team_metadata(db: Queryable, league: str) -> dict[str, dict[str, Any]]:
    try:
        as_of_rows = db.query("SELECT MAX(as_of_utc) AS as_of_utc FROM teams WHERE league = ?", (league,))
        team_as_of = as_of_rows[0]["as_of_utc"] if as_of_rows and as_of_rows[0].get("as_of_utc") else None
        if not team_as_of:
            return {}
        rows = db.query(
            """
            SELECT team_abbrev, team_name, conference, division
            FROM teams
            WHERE league = ? AND as_of_utc = ?
            """,
            (league, team_as_of),
        )
    except Exception:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        team = canonical_team_code(row.get("team_abbrev"), league=league)
        if not team:
            continue
        out[team] = {
            "team_name": row.get("team_name"),
            "conference": row.get("conference"),
            "division": row.get("division"),
        }
    return out


def _format_probability(probability: float | None) -> str:
    if probability is None:
        return "-"
    return f"{float(probability):.1%}"


def build_league_report_markdown(rows: list[dict[str, Any]], model_columns: list[str]) -> str:
    headers = ["Home Team", "Away Team", "Date"] + model_columns
    sep = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]

    for row in rows:
        if str(row.get("home_or_away") or "").strip().lower() != "home":
            continue
        home_team = row.get("home_team") or "-"
        away_team = row.get("away_team") or "-"
        game_date = row.get("next_game_date_utc") or "-"
        team_probs = row.get("model_win_probabilities", {})
        cells = [home_team, away_team, game_date] + [_format_probability(team_probs.get(model)) for model in model_columns]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def answer_league_report(db: Queryable, league: str | None) -> tuple[str, dict]:
    league_code = str(league or "NBA").upper()
    if league_code not in TEAM_ALIAS_GROUPS_BY_LEAGUE:
        league_code = "NBA"

    latest_as_of = latest_upcoming_as_of(db)
    if not latest_as_of:
        return "No upcoming forecasts are currently stored.", {"error": "no_upcoming_forecasts", "league": league_code}

    upcoming_rows = db.query(
        """
        SELECT game_id, game_date_utc, home_team, away_team, ensemble_prob_home_win, per_model_probs_json
        FROM upcoming_game_forecasts
        WHERE as_of_utc = ?
        ORDER BY game_date_utc ASC, game_id ASC
        """,
        (latest_as_of,),
    )
    if not upcoming_rows:
        return "No upcoming forecasts are currently stored.", {"error": "no_upcoming_forecasts", "league": league_code}

    available_leagues = available_team_leagues(db)
    if available_leagues and league_code not in available_leagues:
        return (
            f"No {league_code} team metadata is stored in this database snapshot. "
            f"Available league snapshots: {', '.join(sorted(available_leagues))}.",
            {
                "error": "league_not_available_in_db",
                "league_requested": league_code,
                "available_leagues": sorted(available_leagues),
            },
        )

    team_meta = latest_team_metadata(db, league=league_code)
    allowed_teams = set(TEAM_ALIAS_GROUPS_BY_LEAGUE.get(league_code, {}).keys())
    team_scope = set(team_meta.keys()) if team_meta else allowed_teams

    teams_seen_in_games: set[str] = set()
    next_game_by_team: dict[str, dict[str, Any]] = {}
    model_names: set[str] = {"ensemble"}
    for row in upcoming_rows:
        home_team = canonical_team_code(row.get("home_team"), league=league_code)
        away_team = canonical_team_code(row.get("away_team"), league=league_code)
        if not home_team or not away_team:
            continue
        if team_scope and (home_team not in team_scope or away_team not in team_scope):
            continue
        if home_team in allowed_teams:
            teams_seen_in_games.add(home_team)
        if away_team in allowed_teams:
            teams_seen_in_games.add(away_team)

        for team in (home_team, away_team):
            if team not in allowed_teams or team in next_game_by_team:
                continue
            canonical_row = dict(row)
            canonical_row["home_team"] = home_team
            canonical_row["away_team"] = away_team
            game = oriented_game_forecast(canonical_row, team=team)
            probs = {"ensemble": float(game["ensemble_prob_team_win"])} | game["per_model_probs_team_win"]
            model_names.update(probs.keys())
            next_game_by_team[team] = {
                "game_id": game["game_id"],
                "next_opponent": game["opponent"],
                "next_game_date_utc": game["date"],
                "is_home": game["is_home"],
                "model_win_probabilities": probs,
            }

    model_columns = ordered_model_names(model_names=model_names)
    teams_from_meta = set(team_meta.keys())
    if teams_from_meta:
        teams = sorted((teams_from_meta | teams_seen_in_games) & allowed_teams)
    else:
        teams = sorted((set(TEAM_ALIAS_GROUPS_BY_LEAGUE.get(league_code, {}).keys()) | teams_seen_in_games) & allowed_teams)

    report_rows = []
    for team in teams:
        meta = team_meta.get(team, {})
        next_game = next_game_by_team.get(team)
        if next_game:
            raw_probs = dict(next_game.get("model_win_probabilities", {}))
            model_probs = {name: raw_probs.get(name) for name in model_columns}
            is_home = bool(next_game.get("is_home"))
            home_or_away = "Home" if is_home else "Away"
            opponent = next_game.get("next_opponent")
            home_team = team if is_home else opponent
            away_team = opponent if is_home else team
        else:
            model_probs = {name: None for name in model_columns}
            home_or_away = None
            home_team = None
            away_team = None

        report_rows.append(
            {
                "team": team,
                "team_name": meta.get("team_name"),
                "conference": meta.get("conference"),
                "division": meta.get("division"),
                "next_opponent": next_game.get("next_opponent") if next_game else None,
                "home_team": home_team,
                "away_team": away_team,
                "next_game_date_utc": next_game.get("next_game_date_utc") if next_game else None,
                "home_or_away": home_or_away,
                "model_win_probabilities": model_probs,
            }
        )

    report_rows.sort(
        key=lambda row: (
            0 if row.get("model_win_probabilities", {}).get("ensemble") is not None else 1,
            -float(row["model_win_probabilities"]["ensemble"])
            if row.get("model_win_probabilities", {}).get("ensemble") is not None
            else 0.0,
            row["team"],
        )
    )
    table_md = build_league_report_markdown(report_rows, model_columns=model_columns)
    trust_notes = {
        model: MODEL_TRUST_NOTES.get(
            model,
            "Built on: that model's own rule set. Good at: extra perspective. Watch out: trust less if it is far from ensemble.",
        )
        for model in model_columns
    }
    trust_lines = [f"- `{model}`: {note}" for model, note in trust_notes.items()]
    answer = (
        f"{league_code} all-teams next-game report as of {latest_as_of}.\n\n"
        f"{table_md}\n\n"
        "Model trust guide (super brief): each line is Built on / Good at / Watch out.\n"
        + "\n".join(trust_lines)
    )
    payload = {
        "intent": "league_report",
        "league": league_code,
        "as_of_utc": latest_as_of,
        "model_columns": model_columns,
        "rows": report_rows,
        "model_trust_notes": trust_notes,
    }
    return answer, payload


def answer_best_model(db: Queryable, window_days: int) -> tuple[str, dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).date().isoformat()
    query = """
    WITH ranked AS (
      SELECT
        score_id,
        game_id,
        CASE
          WHEN model_name = 'glm_logit' THEN 'glm_ridge'
          ELSE model_name
        END AS model_name,
        log_loss,
        brier,
        accuracy,
        game_date_utc,
        scored_at_utc,
        ROW_NUMBER() OVER (
          PARTITION BY game_id,
                       CASE
                         WHEN model_name = 'glm_logit' THEN 'glm_ridge'
                         ELSE model_name
                       END
          ORDER BY scored_at_utc DESC, score_id DESC
        ) AS rn
      FROM model_scores
      WHERE game_date_utc >= ?
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
    rows = db.query(query, (cutoff,))
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
