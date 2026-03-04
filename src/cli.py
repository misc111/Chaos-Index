from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.common.config import AppConfig, load_config
from src.common.logging import get_logger, setup_logging
from src.common.time import utc_now_iso
from src.common.utils import ensure_dir, to_json
from src.data_sources.base import HttpClient, SourceFetchResult
from src.data_sources.nhl.games import fetch_games
from src.data_sources.nhl.goalies import fetch_goalie_game_stats
from src.data_sources.nhl.injuries import fetch_injuries_proxy
from src.data_sources.nhl.odds import fetch_public_odds_optional
from src.data_sources.nhl.players import fetch_players
from src.data_sources.nhl.results import build_results_from_games
from src.data_sources.nhl.schedule import fetch_upcoming_schedule
from src.data_sources.nhl.teams import fetch_teams
from src.data_sources.nhl.xg import fetch_xg_optional
from src.evaluation.brier_decomposition import brier_decompose
from src.evaluation.calibration import calibration_alpha_beta, ece_mce
from src.evaluation.diagnostics_glm import save_glm_diagnostics
from src.evaluation.diagnostics_ml import permutation_importance_report
from src.evaluation.validation_backtest_integrity import run_backtest_integrity_checks
from src.evaluation.validation_fragility import missingness_stress_test, perturbation_sensitivity
from src.evaluation.validation_influence import influence_diagnostics
from src.evaluation.validation_significance import blockwise_nested_lrt
from src.evaluation.validation_stability import break_test_trade_deadline, coefficient_paths, vif_table
from src.features.build_features import build_features_from_interim
from src.storage.db import Database
from src.storage.tracker import RunTracker
from src.training.backtest import run_walk_forward_backtest
from src.training.prequential import score_predictions
from src.training.train import train_and_predict

logger = get_logger(__name__)


INTERIM_FILES = {
    "games": "games.parquet",
    "schedule": "schedule.parquet",
    "teams": "teams.parquet",
    "players": "players.parquet",
    "goalies": "goalies.parquet",
    "injuries": "injuries.parquet",
    "odds": "odds.parquet",
    "xg": "xg.parquet",
}



def _client(cfg: AppConfig) -> HttpClient:
    return HttpClient(
        raw_dir=cfg.paths.raw_dir,
        timeout_seconds=cfg.data.timeout_seconds,
        max_retries=cfg.data.max_retries,
        backoff_seconds=cfg.data.backoff_seconds,
        offline_mode=cfg.data.offline_mode,
    )



def _save_interim(df: pd.DataFrame, interim_dir: str, name: str) -> str:
    path = Path(interim_dir) / INTERIM_FILES[name]
    ensure_dir(path.parent)
    try:
        df.to_parquet(path, index=False)
        return str(path)
    except Exception:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return str(csv_path)



def _insert_snapshot(db: Database, res: SourceFetchResult) -> None:
    db.execute(
        """
        INSERT OR REPLACE INTO raw_snapshots(snapshot_id, source, extracted_at_utc, raw_path, metadata_json, freshness_utc, row_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            res.snapshot_id,
            res.source,
            res.extracted_at_utc,
            res.raw_path,
            to_json(res.metadata),
            res.extracted_at_utc,
            int(len(res.dataframe)),
        ),
    )



def _upsert_games(db: Database, games_df: pd.DataFrame) -> None:
    if games_df.empty:
        return
    rows = [
        (
            int(r.game_id),
            int(r.season) if pd.notna(r.season) else None,
            r.game_date_utc,
            r.start_time_utc,
            r.game_state,
            r.home_team,
            r.away_team,
            int(r.home_team_id) if pd.notna(r.home_team_id) else None,
            int(r.away_team_id) if pd.notna(r.away_team_id) else None,
            r.venue,
            int(r.is_neutral_site) if pd.notna(r.is_neutral_site) else 0,
            int(r.home_score) if pd.notna(r.home_score) else None,
            int(r.away_score) if pd.notna(r.away_score) else None,
            int(r.went_ot) if pd.notna(r.went_ot) else 0,
            int(r.went_so) if pd.notna(r.went_so) else 0,
            int(r.home_win) if pd.notna(r.home_win) else None,
            int(r.status_final) if pd.notna(r.status_final) else 0,
            str(r.as_of_utc),
        )
        for r in games_df.itertuples(index=False)
    ]
    db.executemany(
        """
        INSERT OR REPLACE INTO games(
          game_id, season, game_date_utc, start_time_utc, game_state,
          home_team, away_team, home_team_id, away_team_id, venue,
          is_neutral_site, home_score, away_score, went_ot, went_so,
          home_win, status_final, as_of_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )



def _upsert_results(db: Database, results_df: pd.DataFrame) -> None:
    if results_df.empty:
        return
    rows = [
        (
            int(r.game_id),
            int(r.season) if pd.notna(r.season) else None,
            r.game_date_utc,
            r.final_utc,
            r.home_team,
            r.away_team,
            int(r.home_score) if pd.notna(r.home_score) else None,
            int(r.away_score) if pd.notna(r.away_score) else None,
            int(r.home_win) if pd.notna(r.home_win) else None,
            r.ingested_at_utc,
        )
        for r in results_df.itertuples(index=False)
    ]
    db.executemany(
        """
        INSERT OR REPLACE INTO results(
          game_id, season, game_date_utc, final_utc, home_team, away_team,
          home_score, away_score, home_win, ingested_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )



def _latest_snapshot_id(db: Database) -> str | None:
    rows = db.query("SELECT snapshot_id FROM raw_snapshots ORDER BY extracted_at_utc DESC LIMIT 1")
    return rows[0]["snapshot_id"] if rows else None



def cmd_init_db(cfg: AppConfig) -> None:
    db = Database(cfg.paths.db_path)
    db.init_schema()
    logger.info("Initialized DB schema at %s", cfg.paths.db_path)



def cmd_fetch(cfg: AppConfig) -> None:
    db = Database(cfg.paths.db_path)
    db.init_schema()
    client = _client(cfg)

    end_date = datetime.now(timezone.utc) + timedelta(days=cfg.data.upcoming_days)
    start_date = datetime.now(timezone.utc) - timedelta(days=cfg.data.history_days)

    games_res = fetch_games(client, start_date=start_date, end_date=end_date)
    _save_interim(games_res.dataframe, cfg.paths.interim_dir, "games")
    _insert_snapshot(db, games_res)
    _upsert_games(db, games_res.dataframe)

    schedule_res = fetch_upcoming_schedule(client, days_ahead=cfg.data.upcoming_days)
    _save_interim(schedule_res.dataframe, cfg.paths.interim_dir, "schedule")
    _insert_snapshot(db, schedule_res)

    teams_res = fetch_teams(client)
    _save_interim(teams_res.dataframe, cfg.paths.interim_dir, "teams")
    _insert_snapshot(db, teams_res)

    team_abbrevs = sorted(set(teams_res.dataframe.get("team_abbrev", pd.Series(dtype=str)).dropna().astype(str).tolist()))
    season_guess = str(int(games_res.dataframe["season"].dropna().max())) if not games_res.dataframe.empty else "20252026"

    players_res = fetch_players(client, team_abbrevs=team_abbrevs, season=season_guess)
    _save_interim(players_res.dataframe, cfg.paths.interim_dir, "players")
    _insert_snapshot(db, players_res)

    final_ids = games_res.dataframe[games_res.dataframe["status_final"] == 1]["game_id"].astype(int).tolist()
    goalies_res = fetch_goalie_game_stats(client, game_ids=final_ids, max_games=350)
    _save_interim(goalies_res.dataframe, cfg.paths.interim_dir, "goalies")
    _insert_snapshot(db, goalies_res)

    injuries_res = fetch_injuries_proxy(client, teams=team_abbrevs)
    _save_interim(injuries_res.dataframe, cfg.paths.interim_dir, "injuries")
    _insert_snapshot(db, injuries_res)

    odds_res = fetch_public_odds_optional(client)
    _save_interim(odds_res.dataframe, cfg.paths.interim_dir, "odds")
    _insert_snapshot(db, odds_res)

    xg_res = fetch_xg_optional(client)
    _save_interim(xg_res.dataframe, cfg.paths.interim_dir, "xg")
    _insert_snapshot(db, xg_res)

    results_df = build_results_from_games(games_res.dataframe)
    _upsert_results(db, results_df)

    logger.info(
        "Fetch complete | games=%d final=%d upcoming=%d players=%d goalie_rows=%d",
        len(games_res.dataframe),
        int(games_res.dataframe["status_final"].sum()) if not games_res.dataframe.empty else 0,
        len(schedule_res.dataframe),
        len(players_res.dataframe),
        len(goalies_res.dataframe),
    )



def cmd_features(cfg: AppConfig) -> None:
    db = Database(cfg.paths.db_path)
    db.init_schema()

    res = build_features_from_interim(cfg.paths.interim_dir, cfg.paths.processed_dir)
    db.execute(
        """
        INSERT OR REPLACE INTO feature_sets(feature_set_version, created_at_utc, snapshot_id, feature_columns_json, metadata_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            res.feature_set_version,
            utc_now_iso(),
            _latest_snapshot_id(db),
            to_json(res.feature_columns),
            to_json(res.metadata),
        ),
    )
    logger.info("Features built | rows=%d features=%d version=%s", len(res.dataframe), len(res.feature_columns), res.feature_set_version)



def _persist_predictions(
    db: Database,
    forecasts: pd.DataFrame,
    per_model_probs: pd.DataFrame,
    model_run_id: str,
    feature_set_version: str,
) -> None:
    snapshot_id = _latest_snapshot_id(db)

    pred_rows = []
    forecast_rows = []

    per_model_map = per_model_probs.set_index("game_id").to_dict(orient="index")

    for r in forecasts.itertuples(index=False):
        game_id = int(r.game_id)
        model_probs = per_model_map.get(game_id, {})
        as_of = str(r.as_of_utc)

        for model_name, p in model_probs.items():
            if model_name == "game_id":
                continue
            prob = float(p)
            winner = r.home_team if prob >= 0.5 else r.away_team
            pred_rows.append(
                (
                    game_id,
                    as_of,
                    model_name,
                    f"{model_run_id}__{model_name}",
                    feature_set_version,
                    snapshot_id,
                    r.game_date_utc,
                    r.home_team,
                    r.away_team,
                    prob,
                    winner,
                    None,
                    None,
                    r.uncertainty_flags_json,
                    to_json({"source": "train_upcoming"}),
                )
            )

        ensemble_prob = float(r.ensemble_prob_home_win)
        ensemble_winner = r.home_team if ensemble_prob >= 0.5 else r.away_team
        pred_rows.append(
            (
                game_id,
                as_of,
                "ensemble",
                f"{model_run_id}__ensemble",
                feature_set_version,
                snapshot_id,
                r.game_date_utc,
                r.home_team,
                r.away_team,
                ensemble_prob,
                ensemble_winner,
                r.bayes_ci_low,
                r.bayes_ci_high,
                r.uncertainty_flags_json,
                to_json({"source": "train_upcoming"}),
            )
        )

        forecast_rows.append(
            (
                game_id,
                as_of,
                r.game_date_utc,
                r.home_team,
                r.away_team,
                ensemble_prob,
                ensemble_winner,
                r.per_model_probs_json,
                r.spread_min,
                r.spread_median,
                r.spread_max,
                r.spread_mean,
                r.spread_sd,
                r.spread_iqr,
                r.bayes_ci_low,
                r.bayes_ci_high,
                r.uncertainty_flags_json,
                snapshot_id,
                feature_set_version,
                model_run_id,
            )
        )

    db.executemany(
        """
        INSERT OR REPLACE INTO predictions(
          game_id, as_of_utc, model_name, model_run_id, feature_set_version, snapshot_id,
          game_date_utc, home_team, away_team, prob_home_win, pred_winner, prob_low, prob_high,
          uncertainty_flags_json, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        pred_rows,
    )

    db.executemany(
        """
        INSERT OR REPLACE INTO upcoming_game_forecasts(
          game_id, as_of_utc, game_date_utc, home_team, away_team,
          ensemble_prob_home_win, predicted_winner, per_model_probs_json,
          spread_min, spread_median, spread_max, spread_mean, spread_sd, spread_iqr,
          bayes_ci_low, bayes_ci_high, uncertainty_flags_json, snapshot_id,
          feature_set_version, model_run_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        forecast_rows,
    )



def _run_validation_outputs(result: dict, cfg: AppConfig) -> None:
    models = result["models"]
    train_df = result["train_df"].copy()
    feature_cols = result["feature_columns"]
    out_val = ensure_dir(Path(cfg.paths.artifacts_dir) / "validation")

    split = int(len(train_df) * 0.8)
    tr = train_df.iloc[:split].copy()
    va = train_df.iloc[split:].copy()

    # GLM diagnostics.
    glm = models.get("glm_logit")
    if glm is not None and not va.empty:
        va = va.copy()
        va["glm_prob"] = glm.predict_proba(va)
        save_glm_diagnostics(
            df=va,
            p_col="glm_prob",
            y_col="home_win",
            feature_cols=feature_cols,
            coefs=glm.model.coef_[0],
            out_dir=str(Path(cfg.paths.artifacts_dir) / "plots"),
            prefix="glm",
        )

    # Permutation importance.
    for model_name in ["gbdt", "rf"]:
        m = models.get(model_name)
        if m is not None and len(va) > 25:
            permutation_importance_report(
                m.model,
                va[feature_cols],
                va["home_win"].astype(int).to_numpy(),
                out_dir=str(Path(cfg.paths.artifacts_dir) / "validation"),
                model_name=model_name,
            )

    # Significance blocks.
    feature_blocks = {
        "goalie_block": [c for c in feature_cols if "goalie" in c],
        "xg_block": [c for c in feature_cols if "xg" in c],
        "special_teams_block": [c for c in feature_cols if "special" in c or "penalty" in c or "pp_" in c],
        "travel_block": [c for c in feature_cols if "travel" in c or "rest" in c or "tz_" in c],
        "lineup_block": [c for c in feature_cols if "lineup" in c or "roster" in c or "man_games" in c],
        "rink_block": [c for c in feature_cols if "rink" in c],
    }
    sig = blockwise_nested_lrt(tr, va, feature_blocks=feature_blocks, all_features=feature_cols)
    sig.to_csv(out_val / "validation_significance.csv", index=False)

    # Stability and multicollinearity.
    coef_path = coefficient_paths(train_df, features=feature_cols)
    coef_path.to_csv(out_val / "validation_coef_paths.csv", index=False)
    vif = vif_table(train_df, features=feature_cols)
    vif.to_csv(out_val / "validation_vif.csv", index=False)
    break_test = break_test_trade_deadline(train_df, features=feature_cols)
    (out_val / "validation_break_test.json").write_text(json.dumps(break_test, indent=2, sort_keys=True))

    # Influence.
    infl_df, infl_summary = influence_diagnostics(train_df, features=feature_cols, top_k=10)
    infl_df.to_csv(out_val / "validation_influence_top.csv", index=False)
    (out_val / "validation_influence_summary.json").write_text(json.dumps(infl_summary, indent=2, sort_keys=True))

    # Fragility.
    if glm is not None:
        stress = missingness_stress_test(glm, va if not va.empty else train_df, feature_cols=feature_cols)
        stress.to_csv(out_val / "validation_fragility_missingness.csv", index=False)
        pert = perturbation_sensitivity(glm, va if not va.empty else train_df, feature_cols=feature_cols)
        (out_val / "validation_fragility_perturbation.json").write_text(json.dumps(pert, indent=2, sort_keys=True))

    # Calibration robustness outputs.
    if glm is not None and not va.empty:
        p = glm.predict_proba(va)
        y = va["home_win"].astype(int).to_numpy()
        cal = calibration_alpha_beta(y, p) | ece_mce(y, p)
        cal |= brier_decompose(y, p)
        (out_val / "validation_calibration_robustness.json").write_text(json.dumps(cal, indent=2, sort_keys=True))



def cmd_train(cfg: AppConfig) -> None:
    db = Database(cfg.paths.db_path)
    db.init_schema()

    feat_path = Path(cfg.paths.processed_dir) / "features.parquet"
    if not feat_path.exists():
        feat_path = Path(cfg.paths.processed_dir) / "features.csv"
    if not feat_path.exists():
        raise FileNotFoundError("features.parquet not found. Run features first.")

    if feat_path.suffix == ".parquet":
        features_df = pd.read_parquet(feat_path)
    else:
        features_df = pd.read_csv(feat_path)
    feature_set_rows = db.query("SELECT feature_set_version FROM feature_sets ORDER BY created_at_utc DESC LIMIT 1")
    feature_set_version = feature_set_rows[0]["feature_set_version"] if feature_set_rows else "unknown_feature_set"

    tracker = RunTracker(cfg.paths.artifacts_dir)
    run_id = tracker.start_run("train", {"feature_set_version": feature_set_version})
    result = train_and_predict(
        features_df=features_df,
        feature_set_version=feature_set_version,
        artifacts_dir=cfg.paths.artifacts_dir,
        bayes_cfg=cfg.bayes.model_dump(),
    )
    tracker.log_metrics(run_id, {"n_upcoming": int(len(result["forecasts"])), "stack_ready": int(result["stack_ready"])})
    tracker.log_artifact(run_id, "train_metrics", result["train_metrics"])

    _persist_predictions(
        db,
        forecasts=result["forecasts"],
        per_model_probs=result["upcoming_model_probs"],
        model_run_id=result["model_run_id"],
        feature_set_version=feature_set_version,
    )

    # Persist model run metadata rows.
    run_rows = []
    for model_name in [c for c in result["upcoming_model_probs"].columns if c != "game_id"] + ["ensemble"]:
        run_rows.append(
            (
                f"{result['model_run_id']}__{model_name}",
                model_name,
                "daily_train",
                utc_now_iso(),
                _latest_snapshot_id(db),
                feature_set_version,
                to_json({"weights": result["weights"]}),
                to_json(result["train_metrics"].get(model_name, {})),
                result["model_dir"],
                result["model_run_id"],
            )
        )
    db.executemany(
        """
        INSERT OR REPLACE INTO model_runs(
          model_run_id, model_name, run_type, created_at_utc, snapshot_id,
          feature_set_version, params_json, metrics_json, artifact_path, model_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        run_rows,
    )

    _run_validation_outputs(result, cfg)

    tracker.end_run(run_id)
    logger.info("Train complete | model_run_id=%s upcoming=%d", result["model_run_id"], len(result["forecasts"]))



def cmd_backtest(cfg: AppConfig) -> None:
    db = Database(cfg.paths.db_path)
    db.init_schema()

    feat_path = Path(cfg.paths.processed_dir) / "features.parquet"
    if not feat_path.exists():
        feat_path = Path(cfg.paths.processed_dir) / "features.csv"
    if not feat_path.exists():
        raise FileNotFoundError("features.parquet not found. Run features first.")

    if feat_path.suffix == ".parquet":
        features_df = pd.read_parquet(feat_path)
    else:
        features_df = pd.read_csv(feat_path)
    bt = run_walk_forward_backtest(features_df, artifacts_dir=cfg.paths.artifacts_dir, bayes_cfg=cfg.bayes.model_dump(), n_splits=cfg.modeling.cv_splits)

    oof = bt["oof_predictions"]
    if oof.empty:
        logger.warning("Backtest produced no folds.")
        return

    feature_set_rows = db.query("SELECT feature_set_version FROM feature_sets ORDER BY created_at_utc DESC LIMIT 1")
    feature_set_version = feature_set_rows[0]["feature_set_version"] if feature_set_rows else "unknown_feature_set"
    model_run_id = f"backtest_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    snapshot_id = _latest_snapshot_id(db)

    pred_rows = []
    for r in oof.itertuples(index=False):
        game_date = pd.to_datetime(r.game_date_utc)
        as_of = (game_date - pd.Timedelta(days=1)).replace(hour=0, minute=0, second=0).isoformat()
        for model_name in [c for c in oof.columns if c not in {"fold", "home_win", "game_id", "game_date_utc"}]:
            p = float(getattr(r, model_name))
            pred_rows.append(
                (
                    int(r.game_id),
                    as_of,
                    model_name,
                    f"{model_run_id}__{model_name}",
                    feature_set_version,
                    snapshot_id,
                    str(r.game_date_utc),
                    None,
                    None,
                    p,
                    None,
                    None,
                    None,
                    to_json({"backtest_fold": int(r.fold)}),
                    to_json({"source": "walk_forward_backtest"}),
                )
            )

    db.executemany(
        """
        INSERT OR REPLACE INTO predictions(
          game_id, as_of_utc, model_name, model_run_id, feature_set_version, snapshot_id,
          game_date_utc, home_team, away_team, prob_home_win, pred_winner, prob_low, prob_high,
          uncertainty_flags_json, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        pred_rows,
    )

    # Score prequentially after inserting backtest predictions.
    score_info = score_predictions(db, windows_days=cfg.modeling.rolling_windows_days)

    # Integrity checks.
    pred_df = pd.DataFrame(db.query("SELECT * FROM predictions"))
    res_df = pd.DataFrame(db.query("SELECT * FROM results"))
    integrity = run_backtest_integrity_checks(pred_df, res_df, embargo_days=cfg.runtime.embargo_days)
    out_path = Path(cfg.paths.artifacts_dir) / "validation" / "backtest_integrity.json"
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(integrity, indent=2, sort_keys=True))

    logger.info("Backtest complete | oof_rows=%d scored=%d", len(oof), score_info.get("n_scored", 0))



def cmd_run_daily(cfg: AppConfig) -> None:
    cmd_fetch(cfg)
    cmd_features(cfg)
    if cfg.runtime.retrain_daily:
        cmd_train(cfg)

    db = Database(cfg.paths.db_path)
    score_info = score_predictions(db, windows_days=cfg.modeling.rolling_windows_days)

    # Export leaderboard/performance artifacts for dashboard convenience.
    perf = pd.DataFrame(db.query("SELECT * FROM performance_aggregates ORDER BY as_of_utc DESC"))
    if not perf.empty:
        out = Path(cfg.paths.artifacts_dir) / "reports" / "performance_aggregates_latest.csv"
        ensure_dir(out.parent)
        perf.to_csv(out, index=False)

    logger.info("Daily run complete | scored=%s", score_info)



def cmd_smoke(cfg: AppConfig) -> None:
    # Limited smoke with shortened history for speed.
    old_hist = cfg.data.history_days
    old_upc = cfg.data.upcoming_days
    cfg.data.history_days = min(60, old_hist)
    cfg.data.upcoming_days = min(7, old_upc)

    cmd_init_db(cfg)
    cmd_fetch(cfg)
    cmd_features(cfg)
    cmd_train(cfg)
    score_info = score_predictions(Database(cfg.paths.db_path), windows_days=cfg.modeling.rolling_windows_days)

    logger.info("Smoke scoring info: %s", score_info)
    logger.info("Smoke query checks:")
    from src.query.answer import answer_question

    db = Database(cfg.paths.db_path)
    for q in [
        "What's the chance the Leafs win their next game?",
        "Which model has performed best the last 60 days?",
    ]:
        ans, payload = answer_question(db, q)
        logger.info("Q: %s", q)
        logger.info("A: %s", ans)
        logger.info("Payload keys: %s", list(payload.keys()))

    cfg.data.history_days = old_hist
    cfg.data.upcoming_days = old_upc



def main() -> None:
    parser = argparse.ArgumentParser(description="NHL probabilistic forecasting pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd in ["init-db", "fetch", "features", "train", "backtest", "run-daily", "smoke"]:
        p = sub.add_parser(cmd)
        p.add_argument("--config", default="configs/nhl.yaml")

    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging("INFO")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="X has feature names")

    if args.command == "init-db":
        cmd_init_db(cfg)
    elif args.command == "fetch":
        cmd_fetch(cfg)
    elif args.command == "features":
        cmd_features(cfg)
    elif args.command == "train":
        cmd_train(cfg)
    elif args.command == "backtest":
        cmd_backtest(cfg)
    elif args.command == "run-daily":
        cmd_run_daily(cfg)
    elif args.command == "smoke":
        cmd_smoke(cfg)


if __name__ == "__main__":
    main()
