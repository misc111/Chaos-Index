from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.common.market_transforms import vig_free_two_way_probabilities
from src.evaluation.calibration import reliability_table
from src.evaluation.metrics import metric_bundle, per_game_scores
from src.storage.db import Database


PROBABILITY_EPS = 1e-6
MARKET_KEY_MONEYLINE = "h2h"


@dataclass(frozen=True)
class MarketTruthResult:
    per_game: pd.DataFrame
    summary: pd.DataFrame
    calibration: pd.DataFrame


def load_market_truth_moneyline_odds(
    db_path: str,
    *,
    league: str,
    seasons: Iterable[int] | None = None,
    game_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    filters = ["l.league = ?", "COALESCE(l.market_key, '') = 'h2h'", "l.outcome_side IN ('home', 'away')"]
    params: list[Any] = [str(league).upper()]
    resolved_seasons = [int(season) for season in seasons or []]
    if resolved_seasons:
        placeholders = ",".join("?" for _ in resolved_seasons)
        filters.append(f"g.season IN ({placeholders})")
        params.extend(resolved_seasons)
    resolved_game_ids = [int(game_id) for game_id in game_ids or []]
    if resolved_game_ids:
        placeholders = ",".join("?" for _ in resolved_game_ids)
        filters.append(f"g.game_id IN ({placeholders})")
        params.extend(resolved_game_ids)

    db = Database(db_path)
    db.init_schema()
    rows = db.query(
        f"""
        SELECT
          g.game_id,
          g.season,
          g.start_time_utc,
          l.effective_odds_as_of_utc AS odds_as_of_utc,
          l.odds_snapshot_id,
          l.outcome_side,
          l.outcome_price,
          l.bookmaker_key,
          l.bookmaker_title,
          l.market_key
        FROM odds_market_lines_effective_as_of l
        JOIN games g
          ON g.game_id = l.game_id
        WHERE {" AND ".join(filters)}
        ORDER BY g.game_id ASC, l.effective_odds_as_of_utc ASC, l.odds_snapshot_id ASC
        """,
        tuple(params),
    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["odds_as_of_utc"] = pd.to_datetime(frame["odds_as_of_utc"], errors="coerce", utc=True)
    frame["start_time_utc"] = pd.to_datetime(frame["start_time_utc"], errors="coerce", utc=True)
    frame["outcome_price"] = pd.to_numeric(frame["outcome_price"], errors="coerce")
    frame = frame[
        frame["odds_as_of_utc"].notna()
        & frame["start_time_utc"].notna()
        & frame["outcome_price"].notna()
        & (frame["odds_as_of_utc"] <= frame["start_time_utc"])
    ].copy()
    if frame.empty:
        return frame
    frame["odds_as_of_utc"] = frame["odds_as_of_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    frame["start_time_utc"] = frame["start_time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return frame


def _iso_utc(value: Any) -> str | None:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if np.isfinite(numeric) else None


def _american_to_decimal(odds: Any) -> float | None:
    value = _safe_float(odds)
    if value is None or value == 0.0:
        return None
    if value > 0:
        return 1.0 + value / 100.0
    return 1.0 + 100.0 / abs(value)


def _settle_unit_profit(*, side: str | None, odds: Any, home_win: Any) -> float | None:
    if side not in {"home", "away"}:
        return 0.0
    decimal_odds = _american_to_decimal(odds)
    if decimal_odds is None or pd.isna(home_win):
        return None
    home_won = bool(int(home_win))
    won = (side == "home" and home_won) or (side == "away" and not home_won)
    return float(decimal_odds - 1.0) if won else -1.0


def _long_prediction_frame(predictions: pd.DataFrame, model_names: Iterable[str] | None) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    work = predictions.copy()
    if {"model_name", "prob_home_win"} <= set(work.columns):
        work["model_name"] = work["model_name"].astype(str)
        work["prob_home_win"] = pd.to_numeric(work["prob_home_win"], errors="coerce")
        return work

    resolved_models = [str(model) for model in (model_names or []) if str(model).strip()]
    if not resolved_models:
        reserved = {
            "game_id",
            "season",
            "game_date_utc",
            "start_time_utc",
            "as_of_utc",
            "home_team",
            "away_team",
            "home_win",
            "fold",
        }
        resolved_models = [
            column
            for column in work.columns
            if column not in reserved and pd.api.types.is_numeric_dtype(work[column])
        ]
    missing = [model for model in resolved_models if model not in work.columns]
    if missing:
        raise ValueError(f"Prediction frame is missing selected model columns: {missing}")

    id_columns = [
        column
        for column in (
            "game_id",
            "season",
            "game_date_utc",
            "start_time_utc",
            "as_of_utc",
            "home_team",
            "away_team",
            "home_win",
            "fold",
        )
        if column in work.columns
    ]
    long = work.melt(
        id_vars=id_columns,
        value_vars=resolved_models,
        var_name="model_name",
        value_name="prob_home_win",
    )
    long["prob_home_win"] = pd.to_numeric(long["prob_home_win"], errors="coerce")
    return long


def _best_moneyline_snapshots(odds_lines: pd.DataFrame) -> pd.DataFrame:
    if odds_lines.empty:
        return pd.DataFrame()
    required = {"game_id", "outcome_side", "outcome_price"}
    missing = required - set(odds_lines.columns)
    if missing:
        raise ValueError(f"Odds lines are missing required columns: {sorted(missing)}")

    work = odds_lines.copy()
    if "market_key" in work.columns:
        work = work[work["market_key"].fillna("").astype(str) == MARKET_KEY_MONEYLINE].copy()
    if work.empty:
        return pd.DataFrame()

    as_of_source = "odds_as_of_utc" if "odds_as_of_utc" in work.columns else "effective_odds_as_of_utc"
    if as_of_source not in work.columns:
        raise ValueError("Odds lines require odds_as_of_utc or effective_odds_as_of_utc.")
    if "start_time_utc" not in work.columns and "commence_time_utc" in work.columns:
        work["start_time_utc"] = work["commence_time_utc"]
    if "start_time_utc" not in work.columns:
        raise ValueError("Odds lines require start_time_utc or commence_time_utc.")

    work["odds_as_of_ts"] = pd.to_datetime(work[as_of_source], errors="coerce", utc=True)
    work["start_time_ts"] = pd.to_datetime(work["start_time_utc"], errors="coerce", utc=True)
    work["outcome_price"] = pd.to_numeric(work["outcome_price"], errors="coerce")
    work["outcome_side"] = work["outcome_side"].fillna("").astype(str).str.lower()
    work = work[
        work["game_id"].notna()
        & work["odds_as_of_ts"].notna()
        & work["start_time_ts"].notna()
        & work["outcome_price"].notna()
        & work["outcome_side"].isin(["home", "away"])
        & (work["odds_as_of_ts"] <= work["start_time_ts"])
    ].copy()
    if work.empty:
        return pd.DataFrame()

    work["odds_snapshot_id"] = work.get("odds_snapshot_id", pd.Series(index=work.index, dtype=object))
    work["bookmaker_key"] = work.get("bookmaker_key", pd.Series(index=work.index, dtype=object))
    best_side = (
        work.sort_values(
            ["game_id", "odds_as_of_ts", "outcome_side", "outcome_price"],
            ascending=[True, True, True, False],
        )
        .groupby(["game_id", "odds_as_of_ts", "outcome_side"], as_index=False)
        .first()
    )
    prices = (
        best_side.pivot(index=["game_id", "odds_as_of_ts"], columns="outcome_side", values="outcome_price")
        .reset_index()
        .rename(columns={"home": "home_moneyline", "away": "away_moneyline"})
    )
    meta = (
        best_side.groupby(["game_id", "odds_as_of_ts"], as_index=False)
        .agg(
            start_time_ts=("start_time_ts", "first"),
            odds_snapshot_id=("odds_snapshot_id", "first"),
            bookmaker_count=("bookmaker_key", "nunique"),
        )
    )
    snapshots = meta.merge(prices, on=["game_id", "odds_as_of_ts"], how="left")
    snapshots = snapshots[snapshots["home_moneyline"].notna() & snapshots["away_moneyline"].notna()].copy()
    if snapshots.empty:
        return snapshots
    fair_home, fair_away = vig_free_two_way_probabilities(
        snapshots["home_moneyline"],
        snapshots["away_moneyline"],
        input_scale="american_price",
    )
    snapshots["home_vig_free_prob"] = np.asarray(fair_home, dtype=float)
    snapshots["away_vig_free_prob"] = np.asarray(fair_away, dtype=float)
    return snapshots.sort_values(["game_id", "odds_as_of_ts"]).reset_index(drop=True)


def _snapshot_payload(snapshots: pd.DataFrame, *, prefix: str) -> dict[str, Any]:
    if snapshots.empty:
        return {
            f"{prefix}_odds_as_of_utc": None,
            f"{prefix}_odds_snapshot_id": None,
            f"{prefix}_home_moneyline": np.nan,
            f"{prefix}_away_moneyline": np.nan,
            f"{prefix}_home_vig_free_prob": np.nan,
            f"{prefix}_away_vig_free_prob": np.nan,
        }
    row = snapshots.iloc[0]
    return {
        f"{prefix}_odds_as_of_utc": _iso_utc(row["odds_as_of_ts"]),
        f"{prefix}_odds_snapshot_id": row.get("odds_snapshot_id"),
        f"{prefix}_home_moneyline": float(row["home_moneyline"]),
        f"{prefix}_away_moneyline": float(row["away_moneyline"]),
        f"{prefix}_home_vig_free_prob": float(row["home_vig_free_prob"]),
        f"{prefix}_away_vig_free_prob": float(row["away_vig_free_prob"]),
    }


def _market_snapshots_for_predictions(predictions: pd.DataFrame, odds_lines: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    snapshots = _best_moneyline_snapshots(odds_lines)
    rows: list[dict[str, Any]] = []
    prediction_keys = predictions[["game_id", "as_of_utc", "start_time_utc"]].drop_duplicates().copy()
    prediction_keys["prediction_as_of_ts"] = pd.to_datetime(prediction_keys["as_of_utc"], errors="coerce", utc=True)
    prediction_keys["start_time_ts"] = pd.to_datetime(prediction_keys["start_time_utc"], errors="coerce", utc=True)

    for item in prediction_keys.itertuples(index=False):
        game_id = item.game_id
        pred_ts = item.prediction_as_of_ts
        start_ts = item.start_time_ts
        base = {
            "game_id": game_id,
            "as_of_utc": item.as_of_utc,
            "start_time_utc": item.start_time_utc,
            "prediction_as_of_utc": _iso_utc(pred_ts) if pd.notna(pred_ts) else None,
            "game_start_time_utc": _iso_utc(start_ts) if pd.notna(start_ts) else None,
        }
        if pd.isna(pred_ts) or pd.isna(start_ts):
            rows.append(
                base
                | _snapshot_payload(pd.DataFrame(), prefix="opening")
                | _snapshot_payload(pd.DataFrame(), prefix="current")
                | _snapshot_payload(pd.DataFrame(), prefix="closing")
                | {"timestamp_integrity": "missing_prediction_or_start_time"}
            )
            continue
        game_snapshots = snapshots[snapshots["game_id"] == game_id].copy() if not snapshots.empty else pd.DataFrame()
        opening = game_snapshots.sort_values("odds_as_of_ts").head(1)
        current = game_snapshots[game_snapshots["odds_as_of_ts"] <= pred_ts].sort_values("odds_as_of_ts", ascending=False).head(1)
        closing = game_snapshots[game_snapshots["odds_as_of_ts"] <= start_ts].sort_values("odds_as_of_ts", ascending=False).head(1)
        if pred_ts > start_ts:
            current = pd.DataFrame()
            integrity = "prediction_after_start"
        elif current.empty:
            integrity = "missing_current_market_at_prediction_time"
        elif closing.empty:
            integrity = "missing_closing_market"
        else:
            integrity = "ok"
        rows.append(
            base
            | _snapshot_payload(opening, prefix="opening")
            | _snapshot_payload(current, prefix="current")
            | _snapshot_payload(closing, prefix="closing")
            | {"timestamp_integrity": integrity}
        )
    return pd.DataFrame(rows)


def _add_scoring_columns(frame: pd.DataFrame, *, min_edge: float) -> pd.DataFrame:
    out = frame.copy()
    out["prob_home_win"] = pd.to_numeric(out["prob_home_win"], errors="coerce").clip(PROBABILITY_EPS, 1 - PROBABILITY_EPS)
    out["home_win"] = pd.to_numeric(out.get("home_win"), errors="coerce")

    out["edge_home_current"] = out["prob_home_win"] - out["current_home_vig_free_prob"]
    out["edge_away_current"] = (1.0 - out["prob_home_win"]) - out["current_away_vig_free_prob"]
    out["recommended_side"] = np.where(out["edge_home_current"] >= out["edge_away_current"], "home", "away")
    out["recommended_edge"] = np.where(
        out["recommended_side"] == "home",
        out["edge_home_current"],
        out["edge_away_current"],
    )
    has_current = out["current_home_vig_free_prob"].notna() & out["current_away_vig_free_prob"].notna()
    out["bettable_flat"] = has_current & (out["recommended_edge"] >= float(min_edge))
    out.loc[~out["bettable_flat"], "recommended_side"] = "none"
    out["bet_odds"] = np.where(
        out["recommended_side"] == "home",
        out["current_home_moneyline"],
        np.where(out["recommended_side"] == "away", out["current_away_moneyline"], np.nan),
    )
    out["recommended_model_probability"] = np.where(
        out["recommended_side"] == "home",
        out["prob_home_win"],
        np.where(out["recommended_side"] == "away", 1.0 - out["prob_home_win"], np.nan),
    )
    out["recommended_current_market_probability"] = np.where(
        out["recommended_side"] == "home",
        out["current_home_vig_free_prob"],
        np.where(out["recommended_side"] == "away", out["current_away_vig_free_prob"], np.nan),
    )
    out["recommended_closing_market_probability"] = np.where(
        out["recommended_side"] == "home",
        out["closing_home_vig_free_prob"],
        np.where(out["recommended_side"] == "away", out["closing_away_vig_free_prob"], np.nan),
    )
    out["clv_probability"] = out["recommended_closing_market_probability"] - out["recommended_current_market_probability"]
    out["closing_line_price_delta"] = np.where(
        out["recommended_side"] == "home",
        out["current_home_moneyline"] - out["closing_home_moneyline"],
        np.where(out["recommended_side"] == "away", out["current_away_moneyline"] - out["closing_away_moneyline"], np.nan),
    )
    out["flat_bet_profit_units"] = [
        _settle_unit_profit(side=str(row.recommended_side), odds=row.bet_odds, home_win=row.home_win)
        for row in out.itertuples(index=False)
    ]

    scored = out["home_win"].notna()
    if scored.any():
        model_scores = per_game_scores(out.loc[scored, "home_win"].astype(int).to_numpy(), out.loc[scored, "prob_home_win"].to_numpy())
        out.loc[scored, "model_log_loss"] = model_scores["log_loss"].to_numpy()
        out.loc[scored, "model_brier"] = model_scores["brier"].to_numpy()
        for prefix, prob_col in (
            ("current_market", "current_home_vig_free_prob"),
            ("closing_market", "closing_home_vig_free_prob"),
        ):
            valid = scored & out[prob_col].notna()
            if valid.any():
                scores = per_game_scores(out.loc[valid, "home_win"].astype(int).to_numpy(), out.loc[valid, prob_col].to_numpy())
                out.loc[valid, f"{prefix}_log_loss"] = scores["log_loss"].to_numpy()
                out.loc[valid, f"{prefix}_brier"] = scores["brier"].to_numpy()
    return out


def _summary_frame(per_game: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, group in per_game.groupby("model_name", sort=True):
        scored = group[group["home_win"].notna()]
        bets = group[group["bettable_flat"].fillna(False)]
        current_scored = scored[scored["current_home_vig_free_prob"].notna()]
        closing_scored = scored[scored["closing_home_vig_free_prob"].notna()]
        flat_profit = pd.to_numeric(bets["flat_bet_profit_units"], errors="coerce").sum() if not bets.empty else 0.0
        bet_count = int(len(bets))

        row: dict[str, Any] = {
            "model_name": model_name,
            "n_predictions": int(len(group)),
            "n_scored": int(len(scored)),
            "n_current_market": int(group["current_home_vig_free_prob"].notna().sum()),
            "n_closing_market": int(group["closing_home_vig_free_prob"].notna().sum()),
            "current_market_coverage": float(group["current_home_vig_free_prob"].notna().mean()) if len(group) else 0.0,
            "closing_market_coverage": float(group["closing_home_vig_free_prob"].notna().mean()) if len(group) else 0.0,
            "bet_count": bet_count,
            "flat_profit_units": float(flat_profit),
            "flat_roi": float(flat_profit / bet_count) if bet_count else 0.0,
            "mean_edge": _safe_float(pd.to_numeric(group["recommended_edge"], errors="coerce").mean()),
            "mean_abs_edge": _safe_float(pd.to_numeric(group["recommended_edge"], errors="coerce").abs().mean()),
            "mean_clv_probability": _safe_float(pd.to_numeric(bets["clv_probability"], errors="coerce").mean()) if bet_count else None,
            "positive_clv_rate": _safe_float((pd.to_numeric(bets["clv_probability"], errors="coerce") > 0).mean()) if bet_count else None,
            "mean_closing_line_price_delta": _safe_float(pd.to_numeric(bets["closing_line_price_delta"], errors="coerce").mean()) if bet_count else None,
        }
        if not scored.empty:
            model_metrics = metric_bundle(scored["home_win"].astype(int).to_numpy(), scored["prob_home_win"].to_numpy())
            row |= {
                "model_log_loss": float(model_metrics["log_loss"]),
                "model_brier": float(model_metrics["brier"]),
                "model_auc": float(model_metrics["auc"]),
            }
        if not current_scored.empty:
            current_metrics = metric_bundle(
                current_scored["home_win"].astype(int).to_numpy(),
                current_scored["current_home_vig_free_prob"].to_numpy(),
            )
            row |= {
                "current_market_log_loss": float(current_metrics["log_loss"]),
                "current_market_brier": float(current_metrics["brier"]),
                "log_loss_gain_vs_current_market": float(current_metrics["log_loss"] - row.get("model_log_loss", np.nan)),
                "brier_gain_vs_current_market": float(current_metrics["brier"] - row.get("model_brier", np.nan)),
            }
        if not closing_scored.empty:
            closing_metrics = metric_bundle(
                closing_scored["home_win"].astype(int).to_numpy(),
                closing_scored["closing_home_vig_free_prob"].to_numpy(),
            )
            row |= {
                "closing_market_log_loss": float(closing_metrics["log_loss"]),
                "closing_market_brier": float(closing_metrics["brier"]),
                "log_loss_gain_vs_closing_market": float(closing_metrics["log_loss"] - row.get("model_log_loss", np.nan)),
                "brier_gain_vs_closing_market": float(closing_metrics["brier"] - row.get("model_brier", np.nan)),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def _calibration_frame(per_game: pd.DataFrame, *, n_bins: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for model_name, group in per_game.groupby("model_name", sort=True):
        scored = group[group["home_win"].notna()].copy()
        if scored.empty:
            continue
        for source, column in (
            ("model", "prob_home_win"),
            ("current_market", "current_home_vig_free_prob"),
            ("closing_market", "closing_home_vig_free_prob"),
        ):
            valid = scored[scored[column].notna()].copy()
            if valid.empty:
                continue
            table = reliability_table(
                valid["home_win"].astype(int).to_numpy(),
                valid[column].astype(float).to_numpy(),
                n_bins=n_bins,
            )
            if table.empty:
                continue
            table.insert(0, "probability_source", source)
            table.insert(0, "model_name", model_name)
            rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_market_truth(
    predictions: pd.DataFrame,
    odds_lines: pd.DataFrame,
    *,
    model_names: Iterable[str] | None = None,
    min_edge: float = 0.0,
    calibration_bins: int = 10,
) -> MarketTruthResult:
    long_predictions = _long_prediction_frame(predictions, model_names)
    if long_predictions.empty:
        empty = pd.DataFrame()
        return MarketTruthResult(per_game=empty, summary=empty, calibration=empty)

    required_prediction_columns = {"game_id", "as_of_utc", "start_time_utc"}
    missing = required_prediction_columns - set(long_predictions.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing required columns: {sorted(missing)}")

    market_snapshots = _market_snapshots_for_predictions(long_predictions, odds_lines)
    per_game = long_predictions.merge(
        market_snapshots,
        on=["game_id", "as_of_utc", "start_time_utc"],
        how="left",
    )
    per_game = _add_scoring_columns(per_game, min_edge=min_edge)
    return MarketTruthResult(
        per_game=per_game,
        summary=_summary_frame(per_game),
        calibration=_calibration_frame(per_game, n_bins=max(1, int(calibration_bins))),
    )


__all__ = [
    "MARKET_KEY_MONEYLINE",
    "MarketTruthResult",
    "build_market_truth",
    "load_market_truth_moneyline_odds",
]
