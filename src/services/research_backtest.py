from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import AppConfig
from src.common.logging import get_logger
from src.common.research import resolve_research_paths
from src.common.utils import ensure_dir, to_json
from src.evaluation.calibration import calibration_alpha_beta, ece_mce
from src.evaluation.metrics import metric_bundle
from src.evaluation.research_betting import score_betting_performance
from src.evaluation.validation_backtest_integrity import run_backtest_integrity_checks
from src.features.build_features import build_features_from_interim
from src.features.leakage_checks import run_leakage_checks
from src.research.model_comparison import (
    CANDIDATE_MODEL_NAMES,
    FEATURE_POOL_FULL_SCREENED,
    FEATURE_POOL_PRODUCTION_MODEL_MAP,
    CandidateFeatureSets,
    CandidateSpec,
    _candidate_specs,
    _select_feature_sets,
)
from src.services.ingest import save_interim
from src.services.train import load_features_dataframe
from src.storage.db import Database
from src.training.cv import expanding_window_date_splits
from src.training.feature_selection import select_feature_columns
from src.training.model_feature_research import load_model_feature_map

logger = get_logger(__name__)

FEATURE_POOL_RESEARCH_BROAD = "research_broad"


@dataclass(frozen=True)
class ResearchBacktestResult:
    league: str
    report_path: Path
    scorecard_path: Path
    fold_metrics_path: Path
    promotion_path: Path
    best_candidate_model: str


def _params_json(params: dict[str, Any]) -> str:
    if not params:
        return "{}"
    return "; ".join(f"{key}={value}" for key, value in sorted(params.items()))


def _parse_candidate_models(candidate_models: list[str] | None) -> set[str] | None:
    if not candidate_models:
        return None
    normalized = {str(model).strip() for model in candidate_models if str(model).strip()}
    bad = sorted(normalized - CANDIDATE_MODEL_NAMES)
    if bad:
        raise ValueError(f"Unknown candidate model names: {bad}. Valid={sorted(CANDIDATE_MODEL_NAMES)}")
    return normalized or None


def _selected_seasons(db: Database, history_seasons: int) -> list[int]:
    rows = db.query("SELECT DISTINCT season FROM games WHERE season IS NOT NULL ORDER BY season DESC")
    seasons = [int(row["season"]) for row in rows if row.get("season") is not None]
    return seasons[: max(1, int(history_seasons))]


def _export_research_interim(cfg: AppConfig, *, history_seasons: int) -> tuple[list[int], Path]:
    db = Database(cfg.paths.db_path)
    seasons = _selected_seasons(db, history_seasons)
    if not seasons:
        raise RuntimeError("No seasons were available in the games table. Run import-history or fetch data first.")

    placeholders = ",".join("?" for _ in seasons)
    games = pd.DataFrame(
        db.query(
            f"""
            SELECT game_id, season, game_date_utc, start_time_utc, game_state,
                   home_team, away_team, home_team_id, away_team_id, venue,
                   is_neutral_site, home_score, away_score, went_ot, went_so,
                   home_win, status_final, as_of_utc
            FROM games
            WHERE season IN ({placeholders})
            ORDER BY start_time_utc ASC, game_id ASC
            """,
            tuple(seasons),
        )
    )
    if games.empty:
        raise RuntimeError("Selected research seasons produced no games rows")

    paths = resolve_research_paths(cfg)
    ensure_dir(paths.interim_dir)
    save_interim(games, str(paths.interim_dir), "games")
    save_interim(pd.DataFrame(), str(paths.interim_dir), "goalies")
    save_interim(pd.DataFrame(), str(paths.interim_dir), "players")
    save_interim(pd.DataFrame(), str(paths.interim_dir), "injuries")
    save_interim(pd.DataFrame(), str(paths.interim_dir), "odds")
    save_interim(pd.DataFrame(), str(paths.interim_dir), "schedule")
    save_interim(pd.DataFrame(), str(paths.interim_dir), "teams")
    save_interim(pd.DataFrame(), str(paths.interim_dir), "xg")
    return seasons, paths.interim_dir


def _build_research_features(cfg: AppConfig, *, history_seasons: int) -> tuple[pd.DataFrame, list[int]]:
    seasons, interim_dir = _export_research_interim(cfg, history_seasons=history_seasons)
    paths = resolve_research_paths(cfg)
    ensure_dir(paths.processed_dir)
    build_features_from_interim(str(interim_dir), str(paths.processed_dir), league=cfg.data.league)
    features_df = load_features_dataframe(str(paths.processed_dir))
    if features_df.empty:
        raise RuntimeError("Research feature build produced no rows")
    return features_df, seasons


def _latest_pregame_moneylines(db: Database, *, league: str, seasons: list[int]) -> pd.DataFrame:
    if not seasons:
        return pd.DataFrame()
    placeholders = ",".join("?" for _ in seasons)
    rows = db.query(
        f"""
        SELECT
          g.game_id,
          g.season,
          g.start_time_utc,
          s.as_of_utc AS odds_as_of_utc,
          l.odds_snapshot_id,
          l.outcome_side,
          l.outcome_price,
          l.bookmaker_key,
          l.bookmaker_title,
          l.market_key
        FROM odds_market_lines l
        JOIN odds_snapshots s
          ON s.odds_snapshot_id = l.odds_snapshot_id
        JOIN games g
          ON g.game_id = l.game_id
        WHERE l.league = ?
          AND g.season IN ({placeholders})
          AND COALESCE(l.market_key, '') = 'h2h'
          AND l.outcome_side IN ('home', 'away')
        ORDER BY g.game_id ASC, s.as_of_utc ASC
        """,
        tuple([league, *seasons]),
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

    latest_snapshot_per_game = frame.groupby("game_id")["odds_as_of_utc"].transform("max")
    frame = frame[frame["odds_as_of_utc"] == latest_snapshot_per_game].copy()
    best_prices = (
        frame.sort_values(["game_id", "outcome_side", "outcome_price"], ascending=[True, True, False])
        .groupby(["game_id", "outcome_side"], as_index=False)
        .first()
    )
    pivot = best_prices.pivot(index="game_id", columns="outcome_side", values="outcome_price").reset_index()
    meta = (
        frame.sort_values(["game_id", "odds_as_of_utc", "odds_snapshot_id"], ascending=[True, False, False])
        .groupby("game_id", as_index=False)
        .first()[["game_id", "season", "start_time_utc", "odds_as_of_utc", "odds_snapshot_id"]]
    )
    merged = meta.merge(pivot, on="game_id", how="left").rename(columns={"home": "home_moneyline", "away": "away_moneyline"})
    return merged


def _resolve_raw_features(
    features_df: pd.DataFrame,
    *,
    cfg: AppConfig,
    feature_pool: str,
    feature_map_model: str,
) -> tuple[list[str], str]:
    raw_features = select_feature_columns(features_df)
    token = str(feature_pool or FEATURE_POOL_RESEARCH_BROAD).strip().lower()
    if token not in {FEATURE_POOL_RESEARCH_BROAD, FEATURE_POOL_FULL_SCREENED, FEATURE_POOL_PRODUCTION_MODEL_MAP}:
        raise ValueError(
            f"Unknown feature_pool='{feature_pool}'. "
            f"Valid={[FEATURE_POOL_RESEARCH_BROAD, FEATURE_POOL_FULL_SCREENED, FEATURE_POOL_PRODUCTION_MODEL_MAP]}"
        )
    if token == FEATURE_POOL_PRODUCTION_MODEL_MAP:
        model_feature_map = load_model_feature_map(cfg.data.league)
        requested = [str(column) for column in model_feature_map.get(str(feature_map_model), []) if str(column).strip()]
        if not requested:
            raise ValueError(f"Production model feature map for league={cfg.data.league} does not contain '{feature_map_model}'")
        missing = [feature for feature in requested if feature not in raw_features]
        if missing:
            raise ValueError(f"Production feature map '{feature_map_model}' includes columns not available in the research dataset: {missing}")
        return requested, f"production model map `{feature_map_model}`"
    if token == FEATURE_POOL_FULL_SCREENED:
        return raw_features, "full screened numeric feature pool after leakage bans"
    return raw_features, "research-broad numeric feature pool after leakage bans"


def _tune_candidate_on_inner_cv(
    spec: CandidateSpec,
    *,
    fit_df: pd.DataFrame,
    feature_sets: CandidateFeatureSets,
    cfg: AppConfig,
) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    resolved_min_train_days = _resolve_adaptive_min_train_days(
        fit_df,
        validation_days=cfg.research.inner_valid_days,
        embargo_days=cfg.research.embargo_days,
        requested_min_train_days=max(120, cfg.research.inner_valid_days * 2),
    )
    if resolved_min_train_days is None:
        return None, pd.DataFrame()
    splits = expanding_window_date_splits(
        fit_df,
        n_splits=cfg.research.inner_folds,
        validation_days=cfg.research.inner_valid_days,
        embargo_days=cfg.research.embargo_days,
        min_train_days=resolved_min_train_days,
    )
    rows: list[dict[str, Any]] = []
    if not splits:
        return None, pd.DataFrame()
    for params in spec.param_grid:
        params_text = _params_json(params)
        for fold_number, (train_idx, valid_idx, bounds) in enumerate(splits, start=1):
            train_df = fit_df.loc[train_idx].copy().sort_values("start_time_utc")
            valid_df = fit_df.loc[valid_idx].copy().sort_values("start_time_utc")
            if train_df.empty or valid_df.empty or valid_df["home_win"].nunique() < 2:
                continue
            try:
                model = spec.builder(feature_sets, params)
                model.fit(train_df)
                predictions = model.predict_proba(valid_df)
                metrics = metric_bundle(valid_df["home_win"].astype(int).to_numpy(), predictions)
                rows.append(
                    {
                        "model_name": spec.model_name,
                        "display_name": spec.display_name,
                        "params": params_text,
                        "fold": fold_number,
                        "log_loss": float(metrics["log_loss"]),
                        "brier": float(metrics["brier"]),
                        "accuracy": float(metrics["accuracy"]),
                        "auc": float(metrics["auc"]),
                        **bounds,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "model_name": spec.model_name,
                        "display_name": spec.display_name,
                        "params": params_text,
                        "fold": fold_number,
                        "log_loss": np.nan,
                        "brier": np.nan,
                        "accuracy": np.nan,
                        "auc": np.nan,
                        "error": f"{type(exc).__name__}: {exc}",
                        **bounds,
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return None, frame
    ok = frame[frame["log_loss"].notna()].copy()
    if ok.empty:
        return None, frame
    summary = (
        ok.groupby("params", as_index=False)
        .agg(
            mean_log_loss=("log_loss", "mean"),
            mean_brier=("brier", "mean"),
            mean_auc=("auc", "mean"),
        )
        .sort_values(["mean_log_loss", "mean_brier", "mean_auc", "params"], ascending=[True, True, False, True])
        .reset_index(drop=True)
    )
    best_params_text = str(summary.iloc[0]["params"])
    best_params = next((params for params in spec.param_grid if _params_json(params) == best_params_text), None)
    return best_params, frame


def _outer_prediction_as_of(bounds: dict[str, str]) -> str:
    train_end = str(bounds["train_end"])
    return f"{train_end}T23:59:59+00:00"


def _predict_outer_fold(
    spec: CandidateSpec,
    *,
    params: dict[str, Any],
    feature_sets: CandidateFeatureSets,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> np.ndarray:
    model = spec.builder(feature_sets, params)
    model.fit(train_df)
    return np.asarray(model.predict_proba(valid_df), dtype=float)


def _predictive_row(*, fold: int, model_name: str, strategy: str, y_true: np.ndarray, p_pred: np.ndarray) -> dict[str, Any]:
    metrics = metric_bundle(y_true, p_pred)
    calibration = calibration_alpha_beta(y_true, p_pred) | ece_mce(y_true, p_pred)
    return {
        "fold": fold,
        "model_name": model_name,
        "strategy": strategy,
        "log_loss": float(metrics["log_loss"]),
        "brier": float(metrics["brier"]),
        "accuracy": float(metrics["accuracy"]),
        "auc": float(metrics["auc"]),
        "ece": float(calibration.get("ece") or 0.0),
        "mce": float(calibration.get("mce") or 0.0),
        "calibration_alpha": float(calibration.get("calibration_alpha") or 0.0),
        "calibration_beta": float(calibration.get("calibration_beta") or 0.0),
    }


def _choose_best_candidate(scorecard: pd.DataFrame, *, baseline_model: str) -> str:
    candidates = scorecard[scorecard["model_name"] != baseline_model].copy()
    if candidates.empty:
        return baseline_model
    ranked = candidates.sort_values(
        ["mean_ending_bankroll", "profitable_folds", "mean_log_loss", "mean_brier"],
        ascending=[False, False, True, True],
    )
    return str(ranked.iloc[0]["model_name"])


def _promotion_summary(scorecard: pd.DataFrame, *, best_model: str, baseline_model: str) -> dict[str, Any]:
    best_rows = scorecard[scorecard["model_name"] == best_model].copy()
    if best_rows.empty:
        return {"eligible": False, "reason": "best_model_not_present"}
    chosen = best_rows.sort_values(["mean_ending_bankroll", "mean_log_loss"], ascending=[False, True]).iloc[0]
    baseline_rows = scorecard[
        (scorecard["model_name"] == baseline_model) & (scorecard["strategy"] == chosen["strategy"])
    ].copy()
    if baseline_rows.empty:
        return {"eligible": False, "reason": "baseline_row_missing", "best_model": best_model}
    baseline = baseline_rows.iloc[0]
    checks = {
        "mean_ending_bankroll": float(chosen["mean_ending_bankroll"]) > float(baseline["mean_ending_bankroll"]),
        "median_roi": float(chosen["median_roi"]) > float(baseline["median_roi"]),
        "outer_fold_profit_wins": int(chosen["profit_winning_folds"]) >= 6,
        "mean_log_loss": float(chosen["mean_log_loss"]) <= float(baseline["mean_log_loss"]),
        "mean_brier": float(chosen["mean_brier"]) <= float(baseline["mean_brier"]),
        "ece_guardrail": float(chosen["mean_ece"]) <= float(baseline["mean_ece"]) + 0.01,
        "integrity_checks": bool(chosen["all_integrity_checks"]),
    }
    return {
        "eligible": bool(all(checks.values())),
        "best_model": best_model,
        "strategy": str(chosen["strategy"]),
        "baseline_model": baseline_model,
        "checks": checks,
        "best_candidate_row": chosen.to_dict(),
        "baseline_row": baseline.to_dict(),
    }


def _write_report(
    *,
    cfg: AppConfig,
    report_path: Path,
    seasons: list[int],
    feature_pool_note: str,
    scorecard: pd.DataFrame,
    promotion: dict[str, Any],
    best_model: str,
) -> None:
    lines = [
        f"# {cfg.data.league} Research Backtest",
        "",
        "Summary",
        f"- Seasons included: {seasons}",
        f"- Feature pool: {feature_pool_note}",
        f"- Outer folds / valid days / inner folds / holdout days: {cfg.research.outer_folds} / {cfg.research.outer_valid_days} / {cfg.research.inner_folds} / {cfg.research.final_holdout_days}",
        f"- Best candidate model: {best_model}",
        f"- Promotion eligible: {promotion.get('eligible', False)}",
        "",
        "Scorecard",
        "```text",
        scorecard[
            [
                "model_name",
                "strategy",
                "mean_ending_bankroll",
                "median_roi",
                "profit_winning_folds",
                "profitable_folds",
                "mean_log_loss",
                "mean_brier",
                "mean_ece",
                "all_integrity_checks",
            ]
        ].to_string(index=False),
        "```",
        "",
        "Promotion Gate",
        "```json",
        to_json(promotion),
        "```",
    ]
    report_path.write_text("\n".join(lines) + "\n")


def _resolve_adaptive_min_train_days(
    df: pd.DataFrame,
    *,
    validation_days: int,
    embargo_days: int,
    requested_min_train_days: int,
    date_col: str = "start_time_utc",
) -> int | None:
    work = df[df["home_win"].notna()].copy()
    if work.empty or date_col not in work.columns:
        return None

    work[date_col] = pd.to_datetime(work[date_col], errors="coerce", utc=True)
    work = work[work[date_col].notna()].sort_values(date_col)
    if work.empty:
        return None

    unique_days = work[date_col].dt.normalize().nunique()
    validation_days = max(1, int(validation_days))
    embargo_days = max(0, int(embargo_days))
    requested = max(1, int(requested_min_train_days))
    max_supported_train_days = int(unique_days) - validation_days - embargo_days - 1
    minimum_usable_train_days = max(30, validation_days)
    if max_supported_train_days < minimum_usable_train_days:
        return None
    return min(requested, max_supported_train_days)


def run_research_backtest(
    cfg: AppConfig,
    *,
    report_slug: str | None = None,
    candidate_models: list[str] | None = None,
    feature_pool: str | None = None,
    feature_map_model: str = "glm_ridge",
    history_seasons: int | None = None,
) -> ResearchBacktestResult:
    history_seasons_value = max(1, int(history_seasons or cfg.research.history_seasons))
    features_df, seasons = _build_research_features(cfg, history_seasons=history_seasons_value)
    leakage_issues = run_leakage_checks(features_df, feature_columns=select_feature_columns(features_df))
    if leakage_issues:
        raise RuntimeError(f"Leakage checks failed for the research dataset: {leakage_issues}")

    raw_features, feature_pool_note = _resolve_raw_features(
        features_df,
        cfg=cfg,
        feature_pool=str(feature_pool or cfg.research.feature_pool),
        feature_map_model=feature_map_model,
    )
    candidate_model_set = _parse_candidate_models(candidate_models)

    historical_df = features_df[features_df["home_win"].notna()].copy().sort_values("start_time_utc").reset_index(drop=True)
    historical_df["start_time_utc"] = pd.to_datetime(historical_df["start_time_utc"], errors="coerce", utc=True)
    historical_df = historical_df[historical_df["start_time_utc"].notna()].copy().reset_index(drop=True)
    max_date = historical_df["start_time_utc"].dt.normalize().max()
    holdout_start = max_date - pd.Timedelta(days=max(1, cfg.research.final_holdout_days) - 1)
    research_df = historical_df[historical_df["start_time_utc"].dt.normalize() < holdout_start].copy().reset_index(drop=True)
    holdout_df = historical_df[historical_df["start_time_utc"].dt.normalize() >= holdout_start].copy().reset_index(drop=True)
    if research_df.empty or holdout_df.empty:
        raise RuntimeError("Research dataset needs both outer-fold rows and a final holdout window")

    resolved_outer_min_train_days = _resolve_adaptive_min_train_days(
        research_df,
        validation_days=cfg.research.outer_valid_days,
        embargo_days=cfg.research.embargo_days,
        requested_min_train_days=max(180, cfg.research.outer_valid_days * 2),
    )
    if resolved_outer_min_train_days is None:
        raise RuntimeError(
            "Research dataset is too short for the requested outer validation window. "
            f"Available_days={research_df['start_time_utc'].dt.normalize().nunique()} "
            f"validation_days={cfg.research.outer_valid_days} embargo_days={cfg.research.embargo_days}."
        )

    outer_splits = expanding_window_date_splits(
        research_df,
        n_splits=cfg.research.outer_folds,
        validation_days=cfg.research.outer_valid_days,
        embargo_days=cfg.research.embargo_days,
        min_train_days=resolved_outer_min_train_days,
    )
    if not outer_splits:
        raise RuntimeError("No outer research CV splits were available")

    odds_df = _latest_pregame_moneylines(Database(cfg.paths.db_path), league=cfg.data.league, seasons=seasons)
    if odds_df.empty:
        raise RuntimeError("Research backtest requires historical moneyline odds in odds_snapshots/odds_market_lines")
    odds_overlap_count = int(research_df["game_id"].isin(set(odds_df["game_id"].tolist())).sum())
    if odds_overlap_count == 0:
        research_start = str(research_df["start_time_utc"].min())
        research_end = str(research_df["start_time_utc"].max())
        odds_start = str(odds_df["start_time_utc"].min())
        odds_end = str(odds_df["start_time_utc"].max())
        raise RuntimeError(
            "Historical moneyline odds do not overlap the outer research window. "
            f"Research window={research_start}..{research_end}; odds window={odds_start}..{odds_end}. "
            "Backtest bankroll results would be misleading because every game would be treated as missing odds."
        )

    candidate_specs = _candidate_specs(
        _select_feature_sets(research_df, raw_features),
        selected_models=candidate_model_set,
    )
    spec_map = {spec.model_name: spec for spec in candidate_specs}
    baseline_model = feature_map_model if feature_map_model in spec_map else "glm_ridge"
    if baseline_model not in spec_map:
        raise RuntimeError(f"Baseline model '{baseline_model}' was not present in the selected candidate set")

    outer_prediction_frames: list[pd.DataFrame] = []
    inner_cv_frames: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []

    for fold_number, (train_idx, valid_idx, bounds) in enumerate(outer_splits, start=1):
        train_df = research_df.loc[train_idx].copy().sort_values("start_time_utc")
        valid_df = research_df.loc[valid_idx].copy().sort_values("start_time_utc")
        if train_df.empty or valid_df.empty or valid_df["home_win"].nunique() < 2:
            continue
        feature_sets = _select_feature_sets(train_df, raw_features)
        prediction_frame = valid_df[
            ["game_id", "season", "game_date_utc", "start_time_utc", "home_team", "away_team", "home_win"]
        ].copy()
        prediction_frame["as_of_utc"] = _outer_prediction_as_of(bounds)

        model_names_for_fold: list[str] = []
        for model_name, spec in spec_map.items():
            best_params, inner_frame = _tune_candidate_on_inner_cv(spec, fit_df=train_df, feature_sets=feature_sets, cfg=cfg)
            if not inner_frame.empty:
                inner_frame["outer_fold"] = fold_number
                inner_cv_frames.append(inner_frame)
            if best_params is None:
                continue
            predictions = _predict_outer_fold(spec, params=best_params, feature_sets=feature_sets, train_df=train_df, valid_df=valid_df)
            prediction_frame[model_name] = predictions
            model_names_for_fold.append(model_name)

        if not model_names_for_fold:
            continue

        prediction_frame = prediction_frame.merge(odds_df, on="game_id", how="left")
        prediction_frame["fold"] = fold_number
        outer_prediction_frames.append(prediction_frame)

        betting_summary, _ = score_betting_performance(prediction_frame, model_names=model_names_for_fold)
        integrity_predictions = pd.concat(
            [
                prediction_frame[["game_id", "as_of_utc", "game_date_utc"]]
                .assign(model_name=model_name, prob_home_win=prediction_frame[model_name].to_numpy(dtype=float))
                for model_name in model_names_for_fold
            ],
            ignore_index=True,
        )
        integrity = run_backtest_integrity_checks(
            integrity_predictions,
            valid_df[["game_id", "home_win"]].rename(columns={"home_win": "outcome_home_win"}),
            embargo_days=cfg.research.embargo_days,
        )
        for _, betting_row in betting_summary.iterrows():
            model_name = str(betting_row["model_name"])
            predictive_row = _predictive_row(
                fold=fold_number,
                model_name=model_name,
                strategy=str(betting_row["strategy"]),
                y_true=valid_df["home_win"].astype(int).to_numpy(),
                p_pred=prediction_frame[model_name].to_numpy(dtype=float),
            )
            fold_metric_rows.append(
                {
                    "fold": fold_number,
                    "model_name": model_name,
                    "strategy": str(betting_row["strategy"]),
                    "train_end": bounds["train_end"],
                    "valid_start": bounds["valid_start"],
                    "valid_end": bounds["valid_end"],
                    **predictive_row,
                    **betting_row.to_dict(),
                }
            )
            integrity_rows.append(
                {
                    "fold": fold_number,
                    "model_name": model_name,
                    "strategy": str(betting_row["strategy"]),
                    "prediction_before_game": bool(integrity["prediction_before_game"]),
                    "unique_prediction_keys": bool(integrity["unique_prediction_keys"]),
                    "no_missing_results_for_scored": bool(integrity["no_missing_results_for_scored"]),
                    "embargo_respected": bool(integrity["embargo_respected"]),
                }
            )

    if not fold_metric_rows:
        raise RuntimeError("Research backtest did not produce any fold metrics")

    fold_metrics = pd.DataFrame(fold_metric_rows)
    integrity_df = pd.DataFrame(integrity_rows)
    scorecard = (
        fold_metrics.groupby(["model_name", "strategy"], as_index=False)
        .agg(
            mean_ending_bankroll=("ending_bankroll", "mean"),
            mean_net_profit=("net_profit", "mean"),
            median_roi=("roi", "median"),
            mean_roi=("roi", "mean"),
            mean_turnover=("turnover", "mean"),
            mean_max_drawdown=("max_drawdown", "mean"),
            profitable_folds=("ending_bankroll", lambda values: int(sum(value > 5000 for value in values))),
            profit_winning_folds=("net_profit", lambda values: int(sum(value > 0 for value in values))),
            mean_log_loss=("log_loss", "mean"),
            mean_brier=("brier", "mean"),
            mean_ece=("ece", "mean"),
            mean_auc=("auc", "mean"),
            bet_count=("bet_count", "sum"),
        )
        .sort_values(["mean_ending_bankroll", "mean_log_loss"], ascending=[False, True])
        .reset_index(drop=True)
    )
    integrity_summary = (
        integrity_df.groupby(["model_name", "strategy"], as_index=False)
        .agg(
            prediction_before_game=("prediction_before_game", "all"),
            unique_prediction_keys=("unique_prediction_keys", "all"),
            no_missing_results_for_scored=("no_missing_results_for_scored", "all"),
            embargo_respected=("embargo_respected", "all"),
        )
    )
    integrity_summary["all_integrity_checks"] = integrity_summary[
        [
            "prediction_before_game",
            "unique_prediction_keys",
            "no_missing_results_for_scored",
            "embargo_respected",
        ]
    ].all(axis=1)
    scorecard = scorecard.merge(integrity_summary, on=["model_name", "strategy"], how="left")

    best_model = _choose_best_candidate(scorecard, baseline_model=baseline_model)
    promotion = _promotion_summary(scorecard, best_model=best_model, baseline_model=baseline_model)

    paths = resolve_research_paths(cfg)
    report_slug_token = report_slug or datetime.now(timezone.utc).strftime("%Y-%m-%d_research_backtest")
    run_dir = ensure_dir(paths.artifacts_dir / report_slug_token)
    scorecard_path = run_dir / "candidate_scorecard.csv"
    fold_metrics_path = run_dir / "outer_fold_metrics.csv"
    promotion_path = run_dir / "promotion_summary.json"
    report_path = run_dir / "research_backtest_report.md"
    scorecard.to_csv(scorecard_path, index=False)
    fold_metrics.to_csv(fold_metrics_path, index=False)
    promotion_path.write_text(to_json(promotion) + "\n")
    if inner_cv_frames:
        pd.concat(inner_cv_frames, ignore_index=True).to_csv(run_dir / "inner_cv_metrics.csv", index=False)
    if outer_prediction_frames:
        pd.concat(outer_prediction_frames, ignore_index=True).to_csv(run_dir / "outer_fold_predictions.csv", index=False)
    holdout_df.to_csv(run_dir / "final_holdout_window.csv", index=False)
    _write_report(
        cfg=cfg,
        report_path=report_path,
        seasons=seasons,
        feature_pool_note=feature_pool_note,
        scorecard=scorecard,
        promotion=promotion,
        best_model=best_model,
    )
    logger.info(
        "Research backtest complete | league=%s best_model=%s scorecard=%s report=%s",
        cfg.data.league,
        best_model,
        scorecard_path,
        report_path,
    )
    return ResearchBacktestResult(
        league=str(cfg.data.league).upper(),
        report_path=report_path,
        scorecard_path=scorecard_path,
        fold_metrics_path=fold_metrics_path,
        promotion_path=promotion_path,
        best_candidate_model=best_model,
    )
