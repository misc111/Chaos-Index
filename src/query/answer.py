from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml

from src.query.parse import TEAM_ABBREV_ALIASES_BY_LEAGUE, TEAM_ALIAS_GROUPS_BY_LEAGUE, parse_question
from src.query.templates import (
    best_model_answer,
    championship_answer,
    help_answer,
    team_multi_game_answer,
    team_prob_answer,
)
from src.storage.db import Database


class Queryable(Protocol):
    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        ...


class _ConnectionQueryAdapter:
    def __init__(self, conn):
        self.conn = conn

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        cur = self.conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


PROBABILITY_KEY_BY_LEAGUE = {
    "NHL": "stanley_cup_prob",
    "NBA": "nba_finals_prob",
}

COMPETITION_NAME_BY_LEAGUE = {
    "NHL": "Stanley Cup",
    "NBA": "NBA Finals",
}

MODEL_REPORT_ORDER = [
    "ensemble",
    "elo_baseline",
    "glm_ridge",
    "dynamic_rating",
    "rf",
    "goals_poisson",
    "gbdt",
    "two_stage",
    "bayes_bt_state_space",
    "bayes_goals",
    "simulation_first",
    "nn_mlp",
]

MODEL_TRUST_NOTES = {
    "ensemble": (
        "All models combined. Best default pick. Can share the same blind spot."
    ),
    "elo_baseline": (
        "Standard sports betting baseline based on past wins/losses. Good long-run read. Slow on sudden changes."
    ),
    "glm_ridge": (
        "Statistical model that uses a checklist. Usually steady. Weird matchups can slip through."
    ),
    "dynamic_rating": (
        "Hot/cold meter. Good for momentum. Can overreact to short streaks."
    ),
    "gbdt": (
        "Machine learning model that finds hidden combos. Sometimes too confident."
    ),
    "rf": (
        "Machine learning model that blends many different predictions from random slices of past games. Good at smoothing out flukes. Can be too cautious on close matchups."
    ),
    "two_stage": (
        "Machine learning model with two steps: first predicts game type (fast/slow, close/lopsided), then predicts winner. Good when style matchups matter. If step 1 is wrong, final pick can be wrong."
    ),
    "goals_poisson": (
        "Score-based model. Good for normal scoring games. Messy games hurt it."
    ),
    "simulation_first": (
        "Runs the matchup thousands of times using set assumptions (team strength, pace, and scoring). Good for seeing different paths. If those assumptions are off, this number can be off."
    ),
    "bayes_bt_state_space": (
        "Tracks team strength after every game and gives a range, not just one number. Good for spotting rising/falling teams with uncertainty shown. Can move fast after injuries, trades, or short weird stretches."
    ),
    "bayes_goals": (
        "Scoring strength + confidence meter. Good trend read. Can lag sudden lineup changes."
    ),
    "nn_mlp": (
        "Machine learning model that finds subtle patterns. Hardest to explain."
    ),
}

LEGACY_MODEL_NAME_MAP = {
    "glm_logit": "glm_ridge",
}



def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _canonical_model_name(model_name: Any) -> str:
    token = str(model_name or "").strip()
    return LEGACY_MODEL_NAME_MAP.get(token, token)


def _canonicalize_model_probabilities(per_model: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw_name, raw_value in per_model.items():
        model_name = _canonical_model_name(raw_name)
        if not model_name:
            continue
        try:
            out[model_name] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return out



def _load_query_config(config_path: str) -> tuple[str, str]:
    path = Path(config_path)
    data = yaml.safe_load(path.read_text()) or {}

    extends = data.get("extends")
    if extends:
        parent_path = Path(extends)
        if not parent_path.is_absolute():
            candidate = (path.parent / parent_path).resolve()
            if candidate.exists():
                parent_path = candidate
            else:
                parent_path = Path(extends).resolve()
        parent_data = yaml.safe_load(parent_path.read_text()) or {}
        data = _deep_update(parent_data, {k: v for k, v in data.items() if k != "extends"})

    db_path = str(data.get("paths", {}).get("db_path", "data/processed/nhl_forecast.db"))
    league = str(data.get("data", {}).get("league", "NHL")).upper()
    if league not in TEAM_ALIAS_GROUPS_BY_LEAGUE:
        league = "NHL"
    return db_path, league



def _latest_upcoming_as_of(db: Queryable) -> str | None:
    rows = db.query("SELECT MAX(as_of_utc) AS as_of_utc FROM upcoming_game_forecasts")
    return rows[0]["as_of_utc"] if rows and rows[0]["as_of_utc"] else None



def _query_team_upcoming_rows(db: Queryable, team: str, as_of_utc: str, limit: int) -> list[dict]:
    q = """
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
    return db.query(q, (as_of_utc, team, as_of_utc, team, int(limit)))



def _oriented_game_forecast(row: dict, team: str) -> dict:
    is_home = row["home_team"] == team
    p_home = float(row["ensemble_prob_home_win"])
    p_team = p_home if is_home else 1 - p_home
    opp = row["away_team"] if is_home else row["home_team"]

    per_model = json.loads(row["per_model_probs_json"]) if row.get("per_model_probs_json") else {}
    per_model = _canonicalize_model_probabilities(per_model)
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



def _ordered_model_names(model_names: set[str]) -> list[str]:
    canonical_names = {_canonical_model_name(name) for name in model_names if _canonical_model_name(name)}
    ordered = [name for name in MODEL_REPORT_ORDER if name in canonical_names]
    seen = set(ordered)
    ordered.extend(sorted(name for name in canonical_names if name not in seen))
    return ordered


def _canonical_team_code(team: Any, league: str) -> str:
    token = str(team or "").strip().upper()
    if not token:
        return ""
    return TEAM_ABBREV_ALIASES_BY_LEAGUE.get(league, {}).get(token, token)


def _latest_team_metadata(db: Queryable, league: str) -> dict[str, dict[str, Any]]:
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
        team = _canonical_team_code(row.get("team_abbrev"), league=league)
        if not team:
            continue
        out[team] = {
            "team_name": row.get("team_name"),
            "conference": row.get("conference"),
            "division": row.get("division"),
        }
    return out


def _available_team_leagues(db: Queryable) -> set[str]:
    try:
        rows = db.query("SELECT DISTINCT league FROM teams WHERE league IS NOT NULL")
    except Exception:
        return set()
    return {str(row.get("league") or "").strip().upper() for row in rows if str(row.get("league") or "").strip()}


def _format_probability(p: float | None) -> str:
    if p is None:
        return "-"
    return f"{float(p):.1%}"


def _build_league_report_markdown(rows: list[dict[str, Any]], model_columns: list[str]) -> str:
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


def _answer_league_report(db: Queryable, league: str | None) -> tuple[str, dict]:
    league_code = str(league or "NHL").upper()
    if league_code not in TEAM_ALIAS_GROUPS_BY_LEAGUE:
        league_code = "NHL"

    latest_as_of = _latest_upcoming_as_of(db)
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

    available_leagues = _available_team_leagues(db)
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

    team_meta = _latest_team_metadata(db, league=league_code)
    allowed_teams = set(TEAM_ALIAS_GROUPS_BY_LEAGUE.get(league_code, {}).keys())
    team_scope = set(team_meta.keys()) if team_meta else allowed_teams

    teams_seen_in_games: set[str] = set()
    next_game_by_team: dict[str, dict[str, Any]] = {}
    model_names: set[str] = {"ensemble"}
    for row in upcoming_rows:
        home_team = _canonical_team_code(row.get("home_team"), league=league_code)
        away_team = _canonical_team_code(row.get("away_team"), league=league_code)
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
            game = _oriented_game_forecast(canonical_row, team=team)
            probs = {"ensemble": float(game["ensemble_prob_team_win"])} | game["per_model_probs_team_win"]
            model_names.update(probs.keys())
            next_game_by_team[team] = {
                "game_id": game["game_id"],
                "next_opponent": game["opponent"],
                "next_game_date_utc": game["date"],
                "is_home": game["is_home"],
                "model_win_probabilities": probs,
            }

    model_columns = _ordered_model_names(model_names=model_names)
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
            if is_home:
                home_team = team
                away_team = opponent
            else:
                home_team = opponent
                away_team = team
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
        key=lambda r: (
            0 if r.get("model_win_probabilities", {}).get("ensemble") is not None else 1,
            -float(r["model_win_probabilities"]["ensemble"])
            if r.get("model_win_probabilities", {}).get("ensemble") is not None
            else 0.0,
            r["team"],
        )
    )
    table_md = _build_league_report_markdown(report_rows, model_columns=model_columns)

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


def _answer_team_next_game(db: Queryable, team: str | None, league: str | None) -> tuple[str, dict]:
    if not team:
        league_msg = f" for {league}" if league else ""
        return f"Team not recognized{league_msg}. Include team name or abbreviation.", {"error": "team_not_recognized", "league": league}

    latest_as_of = _latest_upcoming_as_of(db)
    if not latest_as_of:
        return "No upcoming forecasts are currently stored.", {"error": "no_upcoming_forecasts", "league": league}

    rows = _query_team_upcoming_rows(db, team=team, as_of_utc=latest_as_of, limit=1)
    if not rows:
        return (
            f"No upcoming forecast currently stored for {team}.",
            {"error": "no_upcoming_forecast", "team": team, "league": league},
        )

    game = _oriented_game_forecast(rows[0], team=team)
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



def _answer_team_next_n_games(db: Queryable, team: str | None, n_games: int, league: str | None) -> tuple[str, dict]:
    if not team:
        league_msg = f" for {league}" if league else ""
        return f"Team not recognized{league_msg}. Include team name or abbreviation.", {"error": "team_not_recognized", "league": league}

    n = max(1, int(n_games))
    if n == 1:
        return _answer_team_next_game(db, team=team, league=league)

    latest_as_of = _latest_upcoming_as_of(db)
    if not latest_as_of:
        return "No upcoming forecasts are currently stored.", {"error": "no_upcoming_forecasts", "league": league}

    rows = _query_team_upcoming_rows(db, team=team, as_of_utc=latest_as_of, limit=n)
    if not rows:
        return (
            f"No upcoming forecast currently stored for {team}.",
            {"error": "no_upcoming_forecast", "team": team, "league": league},
        )

    games = [_oriented_game_forecast(row, team=team) for row in rows]
    probs = [float(g["ensemble_prob_team_win"]) for g in games]
    prob_win_all = float(math.prod(probs)) if probs else 0.0
    expected_wins = float(sum(probs))
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
        "meta": {
            "as_of_utc": latest_as_of,
        },
    }
    return answer, payload



def _championship_probs_from_win_rates(win_rates: list[float]) -> list[float]:
    clipped = [max(0.15, min(0.85, float(w))) for w in win_rates]
    strength = [math.log(w / (1 - w)) for w in clipped]
    n_teams = len(strength)
    playoff_slots = min(16, n_teams)
    cutoff = sorted(strength, reverse=True)[playoff_slots - 1] if playoff_slots > 0 else 0.0

    # Smooth playoff-qualification proxy, then strength-weighted champion share.
    raw = []
    for s in strength:
        qual_prob = 1.0 / (1.0 + math.exp(-(s - cutoff) / 0.30))
        raw.append(qual_prob * math.exp(1.75 * s))
    denom = float(sum(raw))
    if denom <= 0:
        if n_teams == 0:
            return []
        return [1.0 / n_teams] * n_teams
    return [r / denom for r in raw]



def _quantile(sorted_values: list[float], q: float) -> float:
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



def _answer_team_championship(
    db: Queryable,
    team: str | None,
    league: str,
    competition: str,
) -> tuple[str, dict]:
    probability_key = PROBABILITY_KEY_BY_LEAGUE.get(league, "championship_prob")

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
    championship_probs = _championship_probs_from_win_rates(mean_win_rates)

    team_idx = teams.index(team)
    point_estimate = float(championship_probs[team_idx])

    rng = random.Random(42)
    n_sims = 2000
    sampled: list[float] = []
    for _ in range(n_sims):
        draw = [rng.betavariate(a, b) for a, b in zip(alpha, beta)]
        sampled.append(float(_championship_probs_from_win_rates(draw)[team_idx]))
    sampled.sort()
    low_90 = _quantile(sampled, 0.05)
    high_90 = _quantile(sampled, 0.95)

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
        "interval_90": {
            "low": float(low_90),
            "high": float(high_90),
        },
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



def _answer_best_model(db: Queryable, window_days: int) -> tuple[str, dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).date().isoformat()
    q = """
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



def _clarify_team_answer(team_candidates: tuple[tuple[str, str], ...]) -> tuple[str, dict]:
    if not team_candidates:
        return "Team not recognized. Include team name or abbreviation.", {"error": "team_not_recognized"}

    options = [f"{league}:{team}" for league, team in team_candidates]
    answer = (
        "That team wording matches multiple teams across leagues. "
        "Please specify league in your question (for example: 'NHL Boston' or 'NBA Boston')."
    )
    return answer, {"intent": "clarify_team", "team_candidates": options}



def answer_question(db: Queryable, question: str, default_league: str | None = "NHL") -> tuple[str, dict]:
    intent = parse_question(question, default_league=default_league)

    if intent.intent_type == "team_next_game":
        return _answer_team_next_game(db, intent.team, intent.league)

    if intent.intent_type == "team_next_n_games":
        return _answer_team_next_n_games(db, intent.team, intent.n_games, intent.league)

    if intent.intent_type == "team_championship":
        league = intent.league or (default_league or "NHL")
        competition = intent.competition or COMPETITION_NAME_BY_LEAGUE.get(league, "Championship")
        return _answer_team_championship(db, intent.team, league=league, competition=competition)

    if intent.intent_type == "league_report":
        return _answer_league_report(db, league=intent.league or default_league)

    if intent.intent_type == "best_model":
        return _answer_best_model(db, intent.window_days)

    if intent.intent_type == "clarify_team":
        return _clarify_team_answer(intent.team_candidates)

    return help_answer(), {"intent": "help"}



def main() -> None:
    parser = argparse.ArgumentParser(description="Local deterministic sports query command")
    parser.add_argument("--config", type=str, default="configs/nhl.yaml")
    parser.add_argument("--league", type=str, choices=["NHL", "NBA"], default=None)
    parser.add_argument("--question", type=str, required=True)
    args = parser.parse_args()

    db_path, config_league = _load_query_config(args.config)
    default_league = args.league or config_league

    db = Database(db_path)
    with db.connect() as conn:
        session = _ConnectionQueryAdapter(conn)
        answer, payload = answer_question(session, args.question, default_league=default_league)
    print(answer)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
