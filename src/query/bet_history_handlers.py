"""Handlers for betting-history summary and breakdown questions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from src.query.contracts import Queryable
from src.query.team_aliases import TEAM_ALIAS_GROUPS_BY_LEAGUE, canonical_league, canonical_team_code

DEFAULT_PROFILE_PREFERENCES: tuple[tuple[str, str], ...] = (
    ("riskAdjusted", "default"),
    ("balanced", "default"),
    ("riskAdjusted", "continuous"),
    ("balanced", "continuous"),
    ("riskAdjusted", "bucketed"),
    ("balanced", "bucketed"),
)
NBA_PROFILE_PREFERENCES: tuple[tuple[str, str], ...] = (
    ("capitalPreservation", "default"),
    ("riskAdjusted", "default"),
    ("balanced", "default"),
    ("capitalPreservation", "continuous"),
    ("riskAdjusted", "continuous"),
    ("balanced", "continuous"),
    ("capitalPreservation", "bucketed"),
    ("riskAdjusted", "bucketed"),
    ("balanced", "bucketed"),
)
CENTRAL_TZ = ZoneInfo("America/Chicago")
PROFILE_TABLE_V2 = "historical_bet_decisions_by_profile_v2"
PROFILE_TABLE_LEGACY = "historical_bet_decisions_by_profile"


def _format_signed_usd(amount: float) -> str:
    sign = "+" if amount >= 0 else "-"
    return f"{sign}${abs(float(amount)):.2f}"


def _format_usd(amount: float) -> str:
    return f"${float(amount):.2f}"


def _bet_role(odds: float | None) -> str | None:
    if odds is None:
        return None
    if odds < 0:
        return "favorite"
    if odds > 0:
        return "underdog"
    return None


def _winner_for_row(row: dict) -> str | None:
    home_team = row.get("home_team")
    away_team = row.get("away_team")
    home_score = row.get("home_score")
    away_score = row.get("away_score")
    if home_team is None or away_team is None or home_score is None or away_score is None:
        return None
    return str(home_team) if int(home_score) > int(away_score) else str(away_team)


def _display_team_name(team: str | None, league: str) -> str | None:
    if not team:
        return None
    league_code = canonical_league(league) or "NBA"
    team_code = canonical_team_code(team, league_code)
    aliases = TEAM_ALIAS_GROUPS_BY_LEAGUE.get(league_code, {}).get(team_code, ())
    if not aliases:
        return str(team)
    display = aliases[0]
    if display.startswith("la "):
        return f"LA {display[3:].title()}"
    if display == "okc":
        return "OKC"
    return display.title()


def _bet_rationale(row: dict, *, league: str) -> str:
    reason = str(row.get("reason") or "").strip()
    team = row.get("team")
    if not team:
        return "No bet because the game was too close." if reason.lower() == "too close" else "No bet."

    display_team = _display_team_name(str(team), league) or str(team)
    role = _bet_role(float(row["odds"])) if row.get("odds") is not None else None
    lowered_reason = reason.lower()
    if "underpriced" in lowered_reason and role:
        return f"{display_team} was the {role} but underpriced."
    if reason:
        return f"{display_team} was bet because the model flagged: {lowered_reason}."
    return f"{display_team} was the selected side."


def _render_bet_history_table(rows: list[dict]) -> str:
    header = [
        "| Game | Bet on | Winner | P/L | Bet rationale |",
        "|---|---|---|---:|---|",
    ]
    body = [
        f"| {row['away_team']} @ {row['home_team']} | {row['bet_on'] or '-'} | {row['winner'] or '-'} | {_format_signed_usd(row['profit'])} | {row['bet_rationale']} |"
        for row in rows
    ]
    return "\n".join([*header, *body])


def _central_date_key(dt: datetime | None = None) -> str:
    current = dt or datetime.now(timezone.utc)
    return current.astimezone(CENTRAL_TZ).date().isoformat()


def _yesterday_central_date_key() -> str:
    return (datetime.now(timezone.utc).astimezone(CENTRAL_TZ).date() - timedelta(days=1)).isoformat()


def _table_exists(db: Queryable, table_name: str) -> bool:
    rows = db.query(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (table_name,),
    )
    return bool(rows)


def _profile_preferences(league: str | None) -> tuple[tuple[str, str], ...]:
    return NBA_PROFILE_PREFERENCES if canonical_league(league) == "NBA" else DEFAULT_PROFILE_PREFERENCES


def _select_profile(db: Queryable, league: str | None = None) -> tuple[str, str, str, str] | None:
    preferences = _profile_preferences(league)

    if _table_exists(db, PROFILE_TABLE_V2):
        rows = db.query(
            f"""
            SELECT
              strategy,
              sizing_style,
              COALESCE(strategy_config_signature, '') AS strategy_config_signature,
              MAX(created_at_utc) AS latest_created_at
            FROM {PROFILE_TABLE_V2}
            GROUP BY strategy, sizing_style, COALESCE(strategy_config_signature, '')
            """
        )
        profiles = [
            {
                "strategy": str(row.get("strategy") or ""),
                "sizing_style": str(row.get("sizing_style") or ""),
                "strategy_config_signature": str(row.get("strategy_config_signature") or ""),
                "latest_created_at": str(row.get("latest_created_at") or ""),
            }
            for row in rows
        ]
        for preferred_strategy, preferred_style in preferences:
            matching = [
                profile
                for profile in profiles
                if profile["strategy"] == preferred_strategy and profile["sizing_style"] == preferred_style
            ]
            if matching:
                matching.sort(key=lambda profile: profile["latest_created_at"], reverse=True)
                selected = matching[0]
                return (
                    selected["strategy"],
                    selected["sizing_style"],
                    selected["strategy_config_signature"],
                    PROFILE_TABLE_V2,
                )
        if profiles:
            profiles.sort(
                key=lambda profile: (
                    profile["sizing_style"] == "continuous",
                    profile["latest_created_at"],
                    profile["strategy"],
                    profile["sizing_style"],
                ),
                reverse=True,
            )
            selected = profiles[0]
            return (
                selected["strategy"],
                selected["sizing_style"],
                selected["strategy_config_signature"],
                PROFILE_TABLE_V2,
            )

    if not _table_exists(db, PROFILE_TABLE_LEGACY):
        return None

    rows = db.query(f"SELECT DISTINCT strategy, sizing_style FROM {PROFILE_TABLE_LEGACY}")
    profiles = {(str(row.get("strategy") or ""), str(row.get("sizing_style") or "")) for row in rows}
    for profile in preferences:
        if profile in profiles:
            return profile[0], profile[1], "", PROFILE_TABLE_LEGACY
    if not profiles:
        return None

    continuous_profiles = sorted(profile for profile in profiles if profile[1] == "continuous")
    if continuous_profiles:
        strategy, sizing_style = continuous_profiles[0]
        return strategy, sizing_style, "", PROFILE_TABLE_LEGACY
    strategy, sizing_style = sorted(profiles)[0]
    return strategy, sizing_style, "", PROFILE_TABLE_LEGACY


def _query_bet_history_rows(
    db: Queryable,
    *,
    league: str,
    history_period: str,
) -> tuple[list[dict], dict[str, str | None]]:
    params: list[object] = []
    date_filter_sql = ""
    period_start = None
    period_end = None

    if history_period == "yesterday":
        target_date = _yesterday_central_date_key()
        date_filter_sql = "AND d.date_central = ?"
        params.append(target_date)
        period_start = target_date
        period_end = target_date

    selected_profile = _select_profile(db, league=league)
    if selected_profile:
        strategy, sizing_style, strategy_config_signature, source_table = selected_profile
        signature_filter_sql = "AND COALESCE(d.strategy_config_signature, '') = ?" if source_table == PROFILE_TABLE_V2 else ""
        signature_params = [strategy_config_signature] if source_table == PROFILE_TABLE_V2 else []
        rows = db.query(
            f"""
            SELECT
              d.game_id,
              d.date_central,
              d.away_team,
              d.home_team,
              d.team,
              d.side,
              d.stake,
              d.odds,
              d.bet_label,
              d.reason,
              d.model_probability,
              d.market_probability,
              d.edge,
              d.expected_value,
              r.home_win,
              r.away_score,
              r.home_score,
              CASE
                WHEN d.stake <= 0 OR d.side = 'none' OR d.odds IS NULL THEN 0
                WHEN ((d.side = 'home' AND CAST(r.home_win AS INTEGER) = 1) OR (d.side = 'away' AND CAST(r.home_win AS INTEGER) = 0))
                  THEN CASE WHEN d.odds > 0 THEN d.stake * (d.odds / 100.0) ELSE d.stake * (100.0 / ABS(d.odds)) END
                ELSE -d.stake
              END AS profit,
              CASE
                WHEN d.stake <= 0 OR d.side = 'none' OR d.odds IS NULL THEN 'no_bet'
                WHEN ((d.side = 'home' AND CAST(r.home_win AS INTEGER) = 1) OR (d.side = 'away' AND CAST(r.home_win AS INTEGER) = 0))
                  THEN 'win'
                ELSE 'loss'
              END AS outcome
            FROM {source_table} d
            JOIN results r
              ON r.game_id = d.game_id
            WHERE d.strategy = ?
              AND d.sizing_style = ?
              {signature_filter_sql}
              AND r.home_win IS NOT NULL
              {date_filter_sql}
            ORDER BY d.date_central ASC, d.game_id ASC
            """,
            tuple([strategy, sizing_style, *signature_params, *params]),
        )
        return rows, {
            "source_table": source_table,
            "strategy": strategy,
            "sizing_style": sizing_style,
            "period_start_central": period_start,
            "period_end_central": period_end,
        }

    if not _table_exists(db, "historical_bet_decisions"):
        return [], {
            "source_table": None,
            "strategy": None,
            "sizing_style": None,
            "period_start_central": period_start,
            "period_end_central": period_end,
        }

    rows = db.query(
        f"""
        SELECT
          d.game_id,
          d.date_central,
          d.away_team,
          d.home_team,
          d.team,
          d.side,
          d.stake,
          d.odds,
          d.bet_label,
          d.reason,
          d.model_probability,
          d.market_probability,
          d.edge,
          d.expected_value,
          r.home_win,
          r.away_score,
          r.home_score,
          CASE
            WHEN d.stake <= 0 OR d.side = 'none' OR d.odds IS NULL THEN 0
            WHEN ((d.side = 'home' AND CAST(r.home_win AS INTEGER) = 1) OR (d.side = 'away' AND CAST(r.home_win AS INTEGER) = 0))
              THEN CASE WHEN d.odds > 0 THEN d.stake * (d.odds / 100.0) ELSE d.stake * (100.0 / ABS(d.odds)) END
            ELSE -d.stake
          END AS profit,
          CASE
            WHEN d.stake <= 0 OR d.side = 'none' OR d.odds IS NULL THEN 'no_bet'
            WHEN ((d.side = 'home' AND CAST(r.home_win AS INTEGER) = 1) OR (d.side = 'away' AND CAST(r.home_win AS INTEGER) = 0))
              THEN 'win'
            ELSE 'loss'
          END AS outcome
        FROM historical_bet_decisions d
        JOIN results r
          ON r.game_id = d.game_id
        WHERE r.home_win IS NOT NULL
          {date_filter_sql}
        ORDER BY d.date_central ASC, d.game_id ASC
        """,
        tuple(params),
    )
    return rows, {
        "source_table": "historical_bet_decisions",
        "strategy": "riskAdjusted",
        "sizing_style": "continuous",
        "period_start_central": period_start,
        "period_end_central": period_end,
    }


def answer_bet_history_summary(
    db: Queryable,
    *,
    league: str,
    history_period: str,
    include_games: bool,
) -> tuple[str, dict]:
    rows, meta = _query_bet_history_rows(db, league=league, history_period=history_period)
    if meta["source_table"] is None:
        return (
            "No tracked betting history is available in this database yet.",
            {"error": "no_bet_history", "league": league, "intent": "bet_history_summary"},
        )

    if not rows:
        period_desc = "last night" if history_period == "yesterday" else "the requested period"
        return (
            f"No tracked betting results are available for {league} {period_desc}.",
            {
                "intent": "bet_history_summary",
                "league": league,
                "period": history_period,
                "strategy": meta["strategy"],
                "sizing_style": meta["sizing_style"],
                "summary": {
                    "tracked_games": 0,
                    "settled_bets": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_risked": 0.0,
                    "total_profit": 0.0,
                    "roi": 0.0,
                },
                "games": [],
            },
        )

    tracked_games = len(rows)
    settled_rows = [row for row in rows if str(row.get("outcome") or "") != "no_bet"]
    wins = sum(1 for row in settled_rows if row.get("outcome") == "win")
    losses = sum(1 for row in settled_rows if row.get("outcome") == "loss")
    total_risked = float(sum(float(row.get("stake") or 0.0) for row in settled_rows))
    total_profit = float(sum(float(row.get("profit") or 0.0) for row in rows))
    roi = total_profit / total_risked if total_risked > 0 else 0.0
    coverage_start = str(rows[0].get("date_central") or "")
    coverage_end = str(rows[-1].get("date_central") or "")
    period_label = f"last night ({meta['period_start_central']})" if history_period == "yesterday" else f"since tracking started ({coverage_start})"

    summary_line = (
        f"{league} {period_label}: {_format_signed_usd(total_profit)} net, "
        f"{_format_usd(total_risked)} risked, {wins}-{losses} on {len(settled_rows)} bets."
    )

    game_rows = [
        {
            "game_id": int(row["game_id"]),
            "date_central": str(row["date_central"]),
            "away_team": str(row["away_team"]),
            "home_team": str(row["home_team"]),
            "team": str(row["team"]) if row.get("team") is not None else None,
            "side": str(row["side"]),
            "stake": float(row["stake"] or 0.0),
            "odds": float(row["odds"]) if row.get("odds") is not None else None,
            "bet_label": str(row["bet_label"]),
            "reason": str(row["reason"]),
            "outcome": str(row["outcome"]),
            "profit": float(row["profit"] or 0.0),
            "home_score": int(row["home_score"]) if row.get("home_score") is not None else None,
            "away_score": int(row["away_score"]) if row.get("away_score") is not None else None,
            "model_probability": float(row["model_probability"]) if row.get("model_probability") is not None else None,
            "market_probability": float(row["market_probability"]) if row.get("market_probability") is not None else None,
            "edge": float(row["edge"]) if row.get("edge") is not None else None,
            "expected_value": float(row["expected_value"]) if row.get("expected_value") is not None else None,
        }
        for row in rows
    ]
    for row in game_rows:
        row["bet_on"] = row["team"]
        row["winner"] = _winner_for_row(row)
        row["bet_role"] = _bet_role(row["odds"])
        row["bet_rationale"] = _bet_rationale(row, league=league)

    if include_games:
        settled_game_rows = [row for row in game_rows if row["outcome"] != "no_bet"]
        if settled_game_rows:
            answer = f"{summary_line}\n\n{_render_bet_history_table(settled_game_rows)}"
        else:
            answer = summary_line
    else:
        answer = summary_line

    payload = {
        "intent": "bet_history_summary",
        "league": league,
        "period": history_period,
        "strategy": meta["strategy"],
        "sizing_style": meta["sizing_style"],
        "source_table": meta["source_table"],
        "coverage_start_central": coverage_start,
        "coverage_end_central": coverage_end,
        "period_start_central": meta["period_start_central"] or coverage_start,
        "period_end_central": meta["period_end_central"] or coverage_end,
        "summary": {
            "tracked_games": tracked_games,
            "settled_bets": len(settled_rows),
            "wins": wins,
            "losses": losses,
            "total_risked": total_risked,
            "total_profit": total_profit,
            "roi": roi,
        },
        "games": game_rows if include_games else [],
    }
    return answer, payload
