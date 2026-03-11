from __future__ import annotations

import pandas as pd

from src.common.time import utc_now_iso
from src.common.utils import to_json
from src.evaluation.change_detection import detect_change_points
from src.evaluation.metrics import per_game_scores
from src.evaluation.performance_timeseries import compute_performance_aggregates
from src.storage.db import Database
from src.storage.prediction_history import frozen_prediction_source_sql
from src.training.model_catalog import MODEL_ALIASES


def _canonical_model_name(model_name: object) -> str:
    token = str(model_name or "").strip().lower()
    if not token:
        return ""
    return str(MODEL_ALIASES.get(token, token))


def score_predictions(db: Database, windows_days: list[int]) -> dict:
    preds = pd.DataFrame(
        db.query(
            f"""
            SELECT
              p.prediction_id,
              p.game_id,
              p.model_name,
              p.model_run_id,
              p.as_of_utc,
              p.game_date_utc,
              p.prob_home_win,
              r.home_win AS outcome_home_win
            FROM predictions p
            JOIN results r ON r.game_id = p.game_id
            WHERE DATETIME(p.as_of_utc) <= COALESCE(
              DATETIME(r.final_utc),
              DATETIME(r.game_date_utc || 'T23:59:59')
            )
              AND {frozen_prediction_source_sql('p')}
            """
        )
    )
    if preds.empty:
        return {"n_scored": 0}

    preds = preds.dropna(subset=["outcome_home_win", "prob_home_win"]).copy()
    preds["model_name"] = preds["model_name"].map(_canonical_model_name)
    preds = preds[preds["model_name"].astype(str) != ""].copy()
    if preds.empty:
        return {"n_scored": 0}
    preds["as_of_sort"] = pd.to_datetime(preds["as_of_utc"], utc=True, errors="coerce")
    preds = preds.sort_values(["game_id", "model_name", "as_of_sort", "prediction_id"]).drop_duplicates(
        subset=["game_id", "model_name"],
        keep="last",
    )
    scored = preds.reset_index(drop=True).copy()
    s = per_game_scores(scored["outcome_home_win"].astype(int).to_numpy(), scored["prob_home_win"].astype(float).to_numpy())
    scored["log_loss"] = s["log_loss"].to_numpy()
    scored["brier"] = s["brier"].to_numpy()
    scored["accuracy"] = s["accuracy"].to_numpy()
    scored["scored_at_utc"] = utc_now_iso()

    db.execute("DELETE FROM model_scores")

    insert_rows = [
        (
            int(r.game_id),
            str(r.model_name),
            r.model_run_id,
            str(r.as_of_utc),
            str(r.game_date_utc),
            float(r.prob_home_win),
            int(r.outcome_home_win),
            float(r.log_loss),
            float(r.brier),
            int(r.accuracy),
            str(r.scored_at_utc),
        )
        for r in scored.itertuples(index=False)
    ]
    db.executemany(
        """
        INSERT OR REPLACE INTO model_scores(
            game_id, model_name, model_run_id, as_of_utc, game_date_utc,
            prob_home_win, outcome_home_win, log_loss, brier, accuracy, scored_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        insert_rows,
    )

    # Aggregates.
    all_scores = pd.DataFrame(db.query("SELECT * FROM model_scores"))
    agg = compute_performance_aggregates(all_scores, as_of_utc=utc_now_iso(), windows_days=windows_days)
    if not agg.empty:
        agg_rows = [
            (
                str(r.as_of_utc),
                str(r.model_name),
                str(r.window_label),
                r.start_date,
                r.end_date,
                int(r.n_games),
                float(r.log_loss),
                float(r.brier),
                float(r.accuracy),
                float(r.auc),
                float(r.ece),
                float(r.mce),
                float(r.calibration_alpha),
                float(r.calibration_beta),
                utc_now_iso(),
            )
            for r in agg.itertuples(index=False)
        ]
        db.executemany(
            """
            INSERT OR REPLACE INTO performance_aggregates(
                as_of_utc, model_name, window_label, start_date, end_date, n_games,
                log_loss, brier, accuracy, auc, ece, mce, calibration_alpha, calibration_beta, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            agg_rows,
        )

    # Change detection.
    cps = detect_change_points(all_scores, metric_col="log_loss")
    db.execute("DELETE FROM change_points")
    if not cps.empty:
        cp_rows = [
            (
                utc_now_iso(),
                str(r.model_name),
                str(r.metric_name),
                str(r.method),
                float(r.statistic),
                float(r.threshold),
                int(r.detected),
                to_json({"index": int(r.index), "date": str(r.date), "value": float(r.value)}),
            )
            for r in cps.itertuples(index=False)
        ]
        db.executemany(
            """
            INSERT INTO change_points(
                as_of_utc, model_name, metric_name, method, statistic, threshold, detected, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            cp_rows,
        )

    return {
        "n_scored": int(len(scored)),
        "n_aggregates": int(len(agg)) if not agg.empty else 0,
        "n_change_points": int(len(cps)) if not cps.empty else 0,
    }
