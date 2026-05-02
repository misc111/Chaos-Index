"""Scoring logic for current-season MLB predictiveness evidence.

The functions here load frozen live predictions, quarantine rows that cannot
prove a pregame timestamp, and compute probability-skill summaries.  They avoid
filesystem writes so they can be unit-tested independently from the artifact
contract.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from src.common.utils import from_json
from src.services.current_season_predictiveness_contract import (
    PRIMARY_TARGET,
    RUNLINE_TARGET,
    SCORES_COLUMNS,
    TARGET_ORDER,
    TOTALS_TARGET,
    clean_payload,
)
from src.services.current_season_predictiveness_metrics import (
    headline_row,
    metric_record,
    reliability_rows as build_reliability_rows,
    rolling_records,
)
from src.storage.db import Database
from src.storage.prediction_history import FROZEN_PREDICTION_SOURCE
from src.training.model_catalog import MODEL_ALIASES


def latest_result_season(db: Database) -> int | None:
    """Resolve the most recent season with settled outcomes, if one exists."""

    rows = db.query(
        """
        SELECT MAX(season) AS season
        FROM results
        WHERE season IS NOT NULL
          AND home_win IS NOT NULL
        """
    )
    value = rows[0]["season"] if rows else None
    return None if value is None else int(value)


def regular_result_count(db: Database, *, start_date: str, end_date: str, season: int | None) -> int:
    """Count settled MLB outcomes inside the requested evidence window."""

    params: list[Any] = [start_date, end_date]
    season_filter = ""
    if season is not None:
        season_filter = "AND r.season = ?"
        params.append(int(season))
    rows = db.query(
        f"""
        SELECT COUNT(*) AS n
        FROM results r
        WHERE r.home_win IS NOT NULL
          AND r.game_date_utc BETWEEN ? AND ?
          {season_filter}
        """,
        tuple(params),
    )
    return int(rows[0]["n"] or 0)


def prediction_rows(db: Database, *, start_date: str, end_date: str, season: int | None) -> pd.DataFrame:
    """Load all prediction rows that could be relevant to the current season."""

    params: list[Any] = [start_date, end_date]
    season_filter = ""
    if season is not None:
        season_filter = "AND r.season = ?"
        params.append(int(season))
    rows = db.query(
        f"""
        SELECT
          p.prediction_id,
          p.game_id,
          p.model_name,
          p.model_run_id,
          p.as_of_utc,
          p.game_date_utc,
          p.prob_home_win,
          p.metadata_json,
          COALESCE(json_extract(p.metadata_json, '$.source'), '') AS prediction_source,
          COALESCE(json_extract(p.metadata_json, '$.target_name'), '{PRIMARY_TARGET}') AS target_name,
          g.start_time_utc,
          r.home_win AS outcome_home_win
        FROM predictions p
        JOIN results r ON r.game_id = p.game_id
        LEFT JOIN games g ON g.game_id = p.game_id
        WHERE r.home_win IS NOT NULL
          AND r.game_date_utc BETWEEN ? AND ?
          {season_filter}
        """,
        tuple(params),
    )
    return pd.DataFrame(rows)


def eligible_scoring_frame(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter raw ledger rows down to frozen, strictly pregame moneyline scores."""

    counts = {
        "joined_prediction_rows": int(len(rows)),
        "ignored_non_live_predictions": 0,
        "quarantined_post_start_predictions": 0,
        "quarantined_missing_start_time_predictions": 0,
        "quarantined_invalid_probability_predictions": 0,
        "unsupported_target_predictions": 0,
    }
    if rows.empty:
        return pd.DataFrame(columns=SCORES_COLUMNS + ["market_baseline_prob"]), counts

    work = rows.copy()
    work["prediction_source"] = work["prediction_source"].fillna("").astype(str)
    live_mask = work["prediction_source"].eq(FROZEN_PREDICTION_SOURCE)
    counts["ignored_non_live_predictions"] = int((~live_mask).sum())

    work["as_of_ts"] = pd.to_datetime(work["as_of_utc"], utc=True, errors="coerce")
    work["start_ts"] = pd.to_datetime(work["start_time_utc"], utc=True, errors="coerce")
    missing_start_mask = live_mask & work["start_ts"].isna()
    pregame_mask = live_mask & work["as_of_ts"].notna() & work["start_ts"].notna() & (work["as_of_ts"] < work["start_ts"])
    counts["quarantined_missing_start_time_predictions"] = int(missing_start_mask.sum())
    counts["quarantined_post_start_predictions"] = int((live_mask & ~missing_start_mask & ~pregame_mask).sum())

    work["target_name"] = work["target_name"].fillna(PRIMARY_TARGET).astype(str)
    target_mask = work["target_name"].eq(PRIMARY_TARGET)
    counts["unsupported_target_predictions"] = int((live_mask & pregame_mask & ~target_mask).sum())

    work["prob_home_win"] = pd.to_numeric(work["prob_home_win"], errors="coerce")
    prob_mask = work["prob_home_win"].between(0.0, 1.0, inclusive="neither")
    counts["quarantined_invalid_probability_predictions"] = int((live_mask & pregame_mask & target_mask & ~prob_mask).sum())

    eligible = work[live_mask & pregame_mask & target_mask & prob_mask].copy()
    if eligible.empty:
        return pd.DataFrame(columns=SCORES_COLUMNS + ["market_baseline_prob"]), counts

    eligible["model_name"] = eligible["model_name"].map(_canonical_model_name)
    eligible = eligible[eligible["model_name"].astype(str) != ""].copy()
    eligible = eligible.sort_values(["game_id", "model_name", "as_of_ts", "prediction_id"]).drop_duplicates(
        subset=["game_id", "model_name"],
        keep="last",
    )
    eligible["outcome_home_win"] = eligible["outcome_home_win"].astype(int)
    eligible["market_baseline_prob"] = eligible["metadata_json"].map(_parse_market_baseline)
    eligible["target_name"] = PRIMARY_TARGET
    return eligible.reset_index(drop=True), counts


def target_payloads(
    scored: pd.DataFrame,
    *,
    as_of_utc: str,
    windows_days: Iterable[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create per-target JSON, headline CSV rows, and reliability rows."""

    targets: dict[str, Any] = {
        RUNLINE_TARGET: {
            "status": "ledger_not_available",
            "reason": "The frozen live ledger currently stores moneyline prob_home_win only.",
        },
        TOTALS_TARGET: {
            "status": "ledger_not_available",
            "reason": "The frozen live ledger currently stores moneyline prob_home_win only.",
        },
    }
    metric_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []

    if scored.empty:
        targets[PRIMARY_TARGET] = {
            "status": "not_measurable",
            "reason": "No settled frozen pregame moneyline predictions are available for the current season window.",
            "models": {},
            "baselines": {},
            "model_vs_baseline": {},
        }
        return {target: targets[target] for target in TARGET_ORDER}, metric_rows, reliability_rows

    moneyline = scored[scored["target_name"].eq(PRIMARY_TARGET)].copy()
    baseline_payloads, baseline_row = _baseline_payloads(moneyline, as_of_utc=as_of_utc, windows_days=windows_days)
    if baseline_row:
        metric_rows.append(baseline_row)

    model_payloads: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for model_name, group in moneyline.groupby("model_name", sort=True):
        group = group.sort_values(["game_date_utc", "game_id"]).copy()
        model_payloads[model_name] = rolling_records(group, model_name=model_name, as_of_utc=as_of_utc, windows_days=windows_days)
        cumulative = model_payloads[model_name].get("cumulative")
        if not cumulative:
            continue
        metric_rows.append({"row_type": "model", "target_name": PRIMARY_TARGET, "model_name": model_name, **headline_row(cumulative)})
        comparisons[model_name] = {"intercept_only": _intercept_comparison(group, cumulative, as_of_utc=as_of_utc)}
        reliability_rows.extend(build_reliability_rows(group, target_name=PRIMARY_TARGET, model_name=model_name))

    targets[PRIMARY_TARGET] = {
        "status": "measurable",
        "models": model_payloads,
        "baselines": baseline_payloads,
        "model_vs_baseline": clean_payload(comparisons),
    }
    return {target: targets[target] for target in TARGET_ORDER}, metric_rows, reliability_rows


def _canonical_model_name(model_name: object) -> str:
    """Normalize legacy aliases so current-season evidence matches scoring tables."""

    token = str(model_name or "").strip().lower()
    if not token:
        return ""
    return str(MODEL_ALIASES.get(token, token))


def _parse_market_baseline(metadata_json: object) -> float | None:
    """Extract a prediction-time market baseline from prediction metadata."""

    metadata = from_json(str(metadata_json or ""), default={})
    if not isinstance(metadata, dict):
        return None
    candidates = [
        metadata.get("market_vig_free_home_prob"),
        metadata.get("market_implied_home_prob"),
        metadata.get("no_vig_home_prob"),
        (metadata.get("baselines") or {}).get("market_vig_free_home_prob")
        if isinstance(metadata.get("baselines"), dict)
        else None,
    ]
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if 0.0 < value < 1.0:
            return value
    return None


def _baseline_payloads(
    moneyline: pd.DataFrame,
    *,
    as_of_utc: str,
    windows_days: Iterable[int],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Build intercept and optional market baseline evidence for unique games."""

    baseline_payloads: dict[str, Any] = {}
    unique_games = moneyline.sort_values(["game_id", "as_of_ts"]).drop_duplicates("game_id", keep="last")
    intercept_frame = unique_games.copy()
    intercept_frame["prob_home_win"] = float(unique_games["outcome_home_win"].mean())
    baseline_payloads["intercept_only"] = rolling_records(
        intercept_frame,
        model_name="intercept_only",
        as_of_utc=as_of_utc,
        windows_days=windows_days,
    )

    market_frame = unique_games[unique_games["market_baseline_prob"].notna()].copy()
    if market_frame.empty:
        baseline_payloads["market_implied"] = {
            "status": "not_available",
            "reason": "Prediction metadata did not include prediction-time no-vig market probabilities.",
        }
    else:
        market_frame["prob_home_win"] = market_frame["market_baseline_prob"].astype(float)
        baseline_payloads["market_implied"] = rolling_records(
            market_frame,
            model_name="market_implied",
            as_of_utc=as_of_utc,
            windows_days=windows_days,
        )

    row = None
    if "cumulative" in baseline_payloads["intercept_only"]:
        row = {
            "row_type": "baseline",
            "target_name": PRIMARY_TARGET,
            "model_name": "intercept_only",
            **headline_row(baseline_payloads["intercept_only"]["cumulative"]),
        }
    return baseline_payloads, row


def _intercept_comparison(group: pd.DataFrame, cumulative: dict[str, Any], *, as_of_utc: str) -> dict[str, Any]:
    """Compare a model against an intercept baseline on the same games."""

    baseline = group.copy()
    baseline["prob_home_win"] = float(group["outcome_home_win"].mean())
    metrics = metric_record(baseline, model_name="intercept_only", window_label="cumulative", as_of_utc=as_of_utc)
    return {
        "log_loss_delta": float(cumulative["log_loss"] - metrics["log_loss"]),
        "brier_delta": float(cumulative["brier"] - metrics["brier"]),
        "accuracy_delta": float(cumulative["accuracy"] - metrics["accuracy"]),
        "n_games": int(cumulative["n_games"]),
    }
