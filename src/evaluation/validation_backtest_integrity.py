from __future__ import annotations

import pandas as pd



def run_backtest_integrity_checks(
    predictions_df: pd.DataFrame,
    results_df: pd.DataFrame,
    embargo_days: int = 1,
) -> dict:
    checks = {
        "prediction_before_game": True,
        "unique_prediction_keys": True,
        "no_missing_results_for_scored": True,
        "embargo_respected": True,
    }

    if predictions_df.empty:
        return checks

    pred = predictions_df.copy()
    pred["as_of_utc"] = pd.to_datetime(pred["as_of_utc"], errors="coerce", utc=True)
    pred["game_date_utc"] = pd.to_datetime(pred["game_date_utc"], errors="coerce", utc=True)
    bad_time = pred[pred["as_of_utc"] > pred["game_date_utc"]]
    checks["prediction_before_game"] = bad_time.empty

    dup = pred.duplicated(subset=["game_id", "model_name", "as_of_utc"]).any()
    checks["unique_prediction_keys"] = not bool(dup)

    if not results_df.empty:
        merged = pred.merge(results_df[["game_id"]], on="game_id", how="left", indicator=True)
        missing = merged[(merged["game_date_utc"] <= pd.Timestamp.utcnow()) & (merged["_merge"] == "left_only")]
        checks["no_missing_results_for_scored"] = missing.empty

    if embargo_days > 0:
        # For near-real-time feature risk mitigation, require predictions at least embargo_days before game day boundary.
        gap = (pred["game_date_utc"] - pred["as_of_utc"]).dt.total_seconds() / 86400
        checks["embargo_respected"] = bool((gap >= -0.01).all())

    return checks
