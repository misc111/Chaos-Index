from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.bayes.fit_offline import run_bayes_offline_fit
from src.common.time import utc_now_iso
from src.common.utils import ensure_dir, stable_hash
from src.evaluation.metrics import metric_bundle
from src.features.leakage_checks import run_leakage_checks
from src.models.bayes_state_space_goals import BayesGoalsModel
from src.models.ensemble_stack import StackingEnsemble
from src.models.ensemble_weighted import compute_weights, spread_stats, weighted_ensemble
from src.models.gbdt import GBDTModel
from src.models.glm_goals import GoalsPoissonModel
from src.models.glm_logit import GLMLogitModel
from src.models.nn import NNModel
from src.models.rf import RFModel
from src.models.two_stage import TwoStageModel
from src.simulation.game_simulator import GameSimulator
from src.training.cv import time_series_splits
from src.training.tune import quick_tune_glm

ALL_MODEL_NAMES = [
    "elo_baseline",
    "dynamic_rating",
    "glm_logit",
    "gbdt",
    "rf",
    "two_stage",
    "goals_poisson",
    "simulation_first",
    "bayes_bt_state_space",
    "bayes_goals",
    "nn_mlp",
]

MODEL_ALIASES = {
    "elo": "elo_baseline",
    "dyn": "dynamic_rating",
    "dynamic": "dynamic_rating",
    "glm": "glm_logit",
    "logit": "glm_logit",
    "gbm": "gbdt",
    "forest": "rf",
    "goals": "goals_poisson",
    "sim": "simulation_first",
    "simulation": "simulation_first",
    "bayes_bt": "bayes_bt_state_space",
    "bayes_goals_model": "bayes_goals",
    "nn": "nn_mlp",
}


RESERVED_NON_FEATURES = {
    "game_id",
    "season",
    "game_date_utc",
    "start_time_utc",
    "home_team",
    "away_team",
    "venue",
    "status_final",
    "home_win",
    "as_of_utc",
    "home_score",
    "away_score",
}

ProgressCallback = Callable[[dict[str, Any]], None]


def _emit_progress(progress_callback: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if progress_callback is None:
        return
    event = {"ts_utc": utc_now_iso(), **payload}
    try:
        progress_callback(event)
    except Exception:
        # Progress reporting should never block model training.
        return


def normalize_selected_models(selected_models: list[str] | None) -> list[str]:
    if not selected_models:
        return list(ALL_MODEL_NAMES)

    out: list[str] = []
    bad: list[str] = []
    seen = set()
    for raw in selected_models:
        token = str(raw).strip().lower()
        if not token:
            continue
        if token in {"all", "*"}:
            return list(ALL_MODEL_NAMES)
        canonical = MODEL_ALIASES.get(token, token)
        if canonical not in ALL_MODEL_NAMES:
            bad.append(raw)
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)

    if bad:
        raise ValueError(f"Unknown model names: {sorted(set(bad))}. Valid={ALL_MODEL_NAMES}")
    if not out:
        raise ValueError(f"No valid models selected. Valid={ALL_MODEL_NAMES}")
    return out



def select_feature_columns(df: pd.DataFrame) -> list[str]:
    banned_exact = {
        "home_goals_for",
        "away_goals_for",
        "home_goals_against",
        "away_goals_against",
        "home_goal_diff",
        "away_goal_diff",
        "home_points_for",
        "away_points_for",
        "home_points_against",
        "away_points_against",
        "home_point_margin",
        "away_point_margin",
        "home_home_score",
        "home_away_score",
        "away_home_score",
        "away_away_score",
        "home_home_win",
        "away_home_win",
        "home_status_final",
        "away_status_final",
    }
    lag_markers = ("ewm_", "r5_", "r14_")
    direct_event_tokens = (
        "goals_for",
        "goals_against",
        "points_for",
        "points_against",
        "shots_for",
        "shots_against",
        "field_goal_attempts_for",
        "field_goal_attempts_against",
        "penalties_taken",
        "penalties_drawn",
        "fouls_committed",
        "fouls_drawn",
        "pp_goals",
        "free_throws_made",
        "starter_save_pct",
        "goalie_quality_raw",
        "team_save_pct_proxy",
        "xg_share_proxy",
        "penalty_diff_proxy",
        "pace_proxy",
        "scoring_efficiency_proxy",
        "possession_proxy",
    )

    cols = []
    for c in df.columns:
        if c in RESERVED_NON_FEATURES:
            continue
        if c.startswith("target_"):
            continue
        if c in banned_exact:
            continue
        if re.search(r"(^|_)home_score($|_)|(^|_)away_score($|_)|(^|_)status_final($|_)", c):
            continue
        if "home_win" in c and "win_rate" not in c:
            continue
        if ("goals_for" in c or "goals_against" in c or "points_for" in c or "points_against" in c) and not any(
            m in c for m in lag_markers
        ):
            continue
        if any(tok in c for tok in direct_event_tokens) and not any(m in c for m in lag_markers):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols



def bayes_feature_subset(feature_cols: list[str]) -> list[str]:
    keep = []
    keywords = [
        "diff_",
        "travel",
        "rest",
        "goalie",
        "special",
        "discipline",
        "availability",
        "rink",
        "arena",
        "elo",
        "dyn",
        "lineup",
    ]
    for c in feature_cols:
        if any(k in c for k in keywords):
            keep.append(c)
    if len(keep) < 12:
        keep = feature_cols[: min(40, len(feature_cols))]
    return keep



def glm_feature_subset(feature_cols: list[str]) -> list[str]:
    keep_exact = {
        "travel_diff",
        "rest_diff",
        "rink_goal_effect",
        "rink_shot_effect",
        "arena_margin_effect",
        "arena_shot_volume_effect",
    }
    keep_prefix = ("diff_", "special_", "discipline_", "goalie_", "availability_", "elo_", "dyn_")

    out = [
        c
        for c in feature_cols
        if (c.startswith(keep_prefix) or c in keep_exact) and not c.endswith("_goalie_id")
    ]

    # Fallback when upstream feature engineering changes and curated set is too small.
    if len(out) < 12:
        banned = {"home_season", "away_season", "home_is_home", "away_is_home"}
        out = [c for c in feature_cols if c not in banned and not c.endswith("_goalie_id")]
    return out


def _resolve_model_feature_columns(
    all_feature_cols: list[str],
    *,
    model_name: str,
    model_feature_columns: dict[str, list[str]] | None,
    fallback_columns: list[str],
) -> list[str]:
    if not model_feature_columns:
        return list(fallback_columns)

    requested = model_feature_columns.get(model_name, [])
    if not requested:
        return list(fallback_columns)

    missing = [c for c in requested if c not in all_feature_cols]
    if missing:
        raise ValueError(f"model_feature_columns[{model_name}] includes missing columns: {missing}")
    return list(requested)



def _fit_suite(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    artifacts_dir: str,
    bayes_cfg: dict,
    selected_models: list[str],
    progress_callback: ProgressCallback | None = None,
    allow_nn: bool = True,
    glm_feature_cols: list[str] | None = None,
    glm_c: float = 1.0,
    model_feature_columns: dict[str, list[str]] | None = None,
):
    models: dict[str, object] = {}
    selected = set(selected_models)
    used_feature_map: dict[str, list[str]] = {}

    glm_cols = _resolve_model_feature_columns(
        feature_cols,
        model_name="glm_logit",
        model_feature_columns=model_feature_columns,
        fallback_columns=glm_feature_cols if glm_feature_cols else feature_cols,
    )
    if "glm_logit" in selected:
        _emit_progress(
            progress_callback,
            {"kind": "model", "model": "glm_logit", "stage": "fit", "status": "started", "message": "Fitting glm_logit"},
        )
        glm = GLMLogitModel(c=float(glm_c))
        glm.fit(train_df, glm_cols)
        models[glm.model_name] = glm
        used_feature_map[glm.model_name] = glm_cols
        _emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "glm_logit",
                "stage": "fit",
                "status": "completed",
                "message": "Completed glm_logit fit",
            },
        )

    gbdt = None
    if "gbdt" in selected:
        _emit_progress(
            progress_callback,
            {"kind": "model", "model": "gbdt", "stage": "fit", "status": "started", "message": "Fitting gbdt"},
        )
        gbdt_cols = _resolve_model_feature_columns(
            feature_cols,
            model_name="gbdt",
            model_feature_columns=model_feature_columns,
            fallback_columns=feature_cols,
        )
        gbdt = GBDTModel()
        gbdt.fit(train_df, gbdt_cols)
        models[gbdt.model_name] = gbdt
        used_feature_map[gbdt.model_name] = gbdt_cols
        _emit_progress(
            progress_callback,
            {"kind": "model", "model": "gbdt", "stage": "fit", "status": "completed", "message": "Completed gbdt fit"},
        )

    if "rf" in selected:
        _emit_progress(
            progress_callback,
            {"kind": "model", "model": "rf", "stage": "fit", "status": "started", "message": "Fitting rf"},
        )
        rf_cols = _resolve_model_feature_columns(
            feature_cols,
            model_name="rf",
            model_feature_columns=model_feature_columns,
            fallback_columns=feature_cols,
        )
        rf = RFModel()
        rf.fit(train_df, rf_cols)
        models[rf.model_name] = rf
        used_feature_map[rf.model_name] = rf_cols
        _emit_progress(
            progress_callback,
            {"kind": "model", "model": "rf", "stage": "fit", "status": "completed", "message": "Completed rf fit"},
        )

    if "two_stage" in selected:
        _emit_progress(
            progress_callback,
            {"kind": "model", "model": "two_stage", "stage": "fit", "status": "started", "message": "Fitting two_stage"},
        )
        two_stage_cols = _resolve_model_feature_columns(
            feature_cols,
            model_name="two_stage",
            model_feature_columns=model_feature_columns,
            fallback_columns=feature_cols,
        )
        two_stage = TwoStageModel()
        two_stage.fit(train_df, two_stage_cols)
        models[two_stage.model_name] = two_stage
        used_feature_map[two_stage.model_name] = two_stage_cols
        _emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "two_stage",
                "stage": "fit",
                "status": "completed",
                "message": "Completed two_stage fit",
            },
        )

    if "goals_poisson" in selected:
        _emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "goals_poisson",
                "stage": "fit",
                "status": "started",
                "message": "Fitting goals_poisson",
            },
        )
        goals = GoalsPoissonModel()
        goals.fit(train_df)
        models[goals.model_name] = goals
        _emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "goals_poisson",
                "stage": "fit",
                "status": "completed",
                "message": "Completed goals_poisson fit",
            },
        )

    if "bayes_goals" in selected:
        _emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "bayes_goals",
                "stage": "fit",
                "status": "started",
                "message": "Fitting bayes_goals",
            },
        )
        bayes_goals = BayesGoalsModel()
        bayes_goals.fit(train_df)
        models[bayes_goals.model_name] = bayes_goals
        _emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "bayes_goals",
                "stage": "fit",
                "status": "completed",
                "message": "Completed bayes_goals fit",
            },
        )

    bcols: list[str] = []
    bayes_diag: dict = {}
    if "bayes_bt_state_space" in selected:
        _emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "bayes_bt_state_space",
                "stage": "fit",
                "status": "started",
                "message": "Fitting bayes_bt_state_space",
            },
        )
        bcols = _resolve_model_feature_columns(
            feature_cols,
            model_name="bayes_bt_state_space",
            model_feature_columns=model_feature_columns,
            fallback_columns=bayes_feature_subset(feature_cols),
        )
        bayes_model, bayes_diag = run_bayes_offline_fit(
            features_df=train_df,
            feature_columns=bcols,
            artifacts_dir=artifacts_dir,
            process_variance=bayes_cfg.get("process_variance", 0.08),
            prior_variance=bayes_cfg.get("prior_variance", 1.5),
            draws=bayes_cfg.get("posterior_draws", 500),
        )
        models[bayes_model.model_name] = bayes_model
        used_feature_map[bayes_model.model_name] = bcols
        _emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "bayes_bt_state_space",
                "stage": "fit",
                "status": "completed",
                "message": "Completed bayes_bt_state_space fit",
            },
        )

    nn_included = False
    if "nn_mlp" in selected and allow_nn and len(train_df) >= 350:
        _emit_progress(
            progress_callback,
            {"kind": "model", "model": "nn_mlp", "stage": "fit_gate", "status": "started", "message": "Evaluating nn_mlp gate"},
        )
        # Gate NN by quick holdout improvement versus GBDT.
        split_ix = int(len(train_df) * 0.85)
        tr = train_df.iloc[:split_ix]
        va = train_df.iloc[split_ix:]
        if not va.empty and va["home_win"].nunique() > 1:
            _emit_progress(
                progress_callback,
                {"kind": "model", "model": "nn_mlp", "stage": "fit", "status": "started", "message": "Fitting nn_mlp"},
            )
            nn_cols = _resolve_model_feature_columns(
                feature_cols,
                model_name="nn_mlp",
                model_feature_columns=model_feature_columns,
                fallback_columns=feature_cols,
            )
            nn = NNModel()
            nn.fit(tr, nn_cols)
            include_nn = True
            if gbdt is not None:
                nn_p = nn.predict_proba(va)
                gbdt_p = gbdt.predict_proba(va)
                m_nn = metric_bundle(va["home_win"].to_numpy(), nn_p)
                m_g = metric_bundle(va["home_win"].to_numpy(), gbdt_p)
                include_nn = m_nn["log_loss"] + 0.001 < m_g["log_loss"]
            if include_nn:
                models[nn.model_name] = nn
                nn_included = True
                used_feature_map[nn.model_name] = nn_cols
                _emit_progress(
                    progress_callback,
                    {
                        "kind": "model",
                        "model": "nn_mlp",
                        "stage": "fit",
                        "status": "completed",
                        "message": "Completed nn_mlp fit",
                    },
                )
            else:
                _emit_progress(
                    progress_callback,
                    {
                        "kind": "model",
                        "model": "nn_mlp",
                        "stage": "fit_gate",
                        "status": "skipped",
                        "message": "Skipped nn_mlp because holdout did not beat gbdt",
                    },
                )
        else:
            _emit_progress(
                progress_callback,
                {
                    "kind": "model",
                    "model": "nn_mlp",
                    "stage": "fit_gate",
                    "status": "skipped",
                    "message": "Skipped nn_mlp because validation split was not eligible",
                },
            )
    elif "nn_mlp" in selected:
        _emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "nn_mlp",
                "stage": "fit_gate",
                "status": "skipped",
                "message": "Skipped nn_mlp because training set is too small",
            },
        )

    return models, bcols, bayes_diag, nn_included, used_feature_map



def _predict_suite(
    models: dict[str, object],
    df: pd.DataFrame,
    feature_cols: list[str],
    selected_models: list[str],
    progress_callback: ProgressCallback | None = None,
    phase: str = "predict",
) -> tuple[pd.DataFrame, dict]:
    out = pd.DataFrame({"game_id": df["game_id"].values})
    extras: dict = {}
    selected = set(selected_models)

    # Direct feature baselines.
    if "elo_baseline" in selected:
        _emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "elo_baseline",
                "stage": phase,
                "status": "started",
                "message": f"Running {phase} for elo_baseline",
            },
        )
        out["elo_baseline"] = np.clip(df.get("elo_home_prob", 0.5).to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        _emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "elo_baseline",
                "stage": phase,
                "status": "completed",
                "message": f"Completed {phase} for elo_baseline",
            },
        )
    if "dynamic_rating" in selected:
        _emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "dynamic_rating",
                "stage": phase,
                "status": "started",
                "message": f"Running {phase} for dynamic_rating",
            },
        )
        out["dynamic_rating"] = np.clip(df.get("dyn_home_prob", 0.5).to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        _emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "dynamic_rating",
                "stage": phase,
                "status": "completed",
                "message": f"Completed {phase} for dynamic_rating",
            },
        )

    for name, m in models.items():
        _emit_progress(
            progress_callback,
            {"kind": "model", "model": name, "stage": phase, "status": "started", "message": f"Running {phase} for {name}"},
        )
        if name in {"goals_poisson"}:
            out[name] = m.predict_proba(df)
        elif name == "bayes_goals":
            mean, low, high = m.predict_proba(df)
            out[name] = mean
            extras["bayes_goals_low"] = low
            extras["bayes_goals_high"] = high
        elif name == "bayes_bt_state_space":
            summary = m.predict_summary(df)
            out[name] = summary.mean
            extras["bayes_low"] = summary.low
            extras["bayes_high"] = summary.high
            extras["bayes_pred_var"] = summary.pred_var
        else:
            out[name] = m.predict_proba(df)
        _emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": name,
                "stage": phase,
                "status": "completed",
                "message": f"Completed {phase} for {name}",
            },
        )

    if "simulation_first" in selected:
        _emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "simulation_first",
                "stage": phase,
                "status": "started",
                "message": f"Running {phase} for simulation_first",
            },
        )
        sim = GameSimulator(seed=42)
        sim_df = sim.simulate_dataframe(df, n_sims=3500)
        out = out.merge(sim_df[["game_id", "sim_prob_home_win"]], on="game_id", how="left")
        out = out.rename(columns={"sim_prob_home_win": "simulation_first"})
        _emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "simulation_first",
                "stage": phase,
                "status": "completed",
                "message": f"Completed {phase} for simulation_first",
            },
        )

    return out, extras



def _oof_predictions(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    glm_feature_cols: list[str],
    artifacts_dir: str,
    bayes_cfg: dict,
    selected_models: list[str],
    progress_callback: ProgressCallback | None = None,
    model_feature_columns: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    splits = time_series_splits(train_df, n_splits=5, min_train_size=min(220, max(80, len(train_df) // 2)))
    rows = []
    selected = set(selected_models)
    _emit_progress(
        progress_callback,
        {
            "kind": "pipeline",
            "stage": "oof_validation",
            "status": "started",
            "message": "Starting time-series OOF validation",
            "fold_total": len(splits),
        },
    )

    for fold_number, (tr_idx, va_idx) in enumerate(splits, start=1):
        _emit_progress(
            progress_callback,
            {
                "kind": "pipeline",
                "stage": "oof_fold",
                "status": "started",
                "message": f"Starting OOF fold {fold_number}/{len(splits)}",
                "fold": fold_number,
                "fold_total": len(splits),
            },
        )
        tr = train_df.loc[tr_idx].copy().sort_values("start_time_utc")
        va = train_df.loc[va_idx].copy().sort_values("start_time_utc")
        if tr.empty or va.empty:
            _emit_progress(
                progress_callback,
                {
                    "kind": "pipeline",
                    "stage": "oof_fold",
                    "status": "skipped",
                    "message": f"Skipped OOF fold {fold_number}/{len(splits)} because split was empty",
                    "fold": fold_number,
                    "fold_total": len(splits),
                },
            )
            continue
        fold_glm_c = 1.0
        fold_glm_cols = _resolve_model_feature_columns(
            feature_cols,
            model_name="glm_logit",
            model_feature_columns=model_feature_columns,
            fallback_columns=glm_feature_cols,
        )
        if "glm_logit" in selected:
            tune = quick_tune_glm(
                tr,
                fold_glm_cols,
                n_splits=3,
                min_train_size=min(140, max(70, len(tr) // 2)),
            )
            fold_glm_c = float(tune.get("best_c", 1.0))
        models, _, _, _, _ = _fit_suite(
            tr,
            feature_cols,
            artifacts_dir=artifacts_dir,
            bayes_cfg=bayes_cfg,
            selected_models=selected_models,
            progress_callback=progress_callback,
            allow_nn=False,
            glm_feature_cols=fold_glm_cols,
            glm_c=fold_glm_c,
            model_feature_columns=model_feature_columns,
        )
        pred, _ = _predict_suite(
            models,
            va,
            feature_cols,
            selected_models=selected_models,
            progress_callback=progress_callback,
            phase="oof_predict",
        )
        pred["home_win"] = va["home_win"].to_numpy()
        pred["game_date_utc"] = va["game_date_utc"].to_numpy()
        rows.append(pred)
        _emit_progress(
            progress_callback,
            {
                "kind": "pipeline",
                "stage": "oof_fold",
                "status": "completed",
                "message": f"Completed OOF fold {fold_number}/{len(splits)}",
                "fold": fold_number,
                "fold_total": len(splits),
            },
        )

    if not rows:
        _emit_progress(
            progress_callback,
            {
                "kind": "pipeline",
                "stage": "oof_validation",
                "status": "completed",
                "message": "OOF validation completed with no folds",
                "fold_total": len(splits),
            },
        )
        return pd.DataFrame()
    _emit_progress(
        progress_callback,
        {
            "kind": "pipeline",
            "stage": "oof_validation",
            "status": "completed",
            "message": "Completed time-series OOF validation",
            "fold_total": len(splits),
        },
    )
    return pd.concat(rows, ignore_index=True)



def train_and_predict(
    features_df: pd.DataFrame,
    feature_set_version: str,
    artifacts_dir: str,
    bayes_cfg: dict,
    selected_models: list[str] | None = None,
    progress_callback: ProgressCallback | None = None,
    selected_feature_columns: list[str] | None = None,
    selected_model_feature_columns: dict[str, list[str]] | None = None,
) -> dict:
    _emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "prepare_data", "status": "started", "message": "Preparing training datasets"},
    )
    df = features_df.sort_values("start_time_utc").copy()
    train_df = df[df["home_win"].notna()].copy()
    upcoming_df = df[df["home_win"].isna()].copy()
    models_selected = normalize_selected_models(selected_models)
    _emit_progress(
        progress_callback,
        {
            "kind": "pipeline",
            "stage": "prepare_data",
            "status": "completed",
            "message": "Prepared training and upcoming datasets",
            "n_train": int(len(train_df)),
            "n_upcoming": int(len(upcoming_df)),
            "model_total": len(models_selected),
            "selected_models": models_selected,
        },
    )

    _emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "feature_selection", "status": "started", "message": "Selecting feature columns"},
    )
    if selected_feature_columns is None:
        feature_cols = select_feature_columns(df)
    else:
        missing_cols = [c for c in selected_feature_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"selected_feature_columns includes missing columns: {missing_cols}")
        non_numeric = [c for c in selected_feature_columns if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise ValueError(f"selected_feature_columns includes non-numeric columns: {non_numeric}")
        feature_cols = list(selected_feature_columns)
    glm_cols = _resolve_model_feature_columns(
        feature_cols,
        model_name="glm_logit",
        model_feature_columns=selected_model_feature_columns,
        fallback_columns=glm_feature_subset(feature_cols),
    )
    _emit_progress(
        progress_callback,
        {
            "kind": "pipeline",
            "stage": "feature_selection",
            "status": "completed",
            "message": "Selected feature columns",
            "feature_count": len(feature_cols),
            "glm_feature_count": len(glm_cols),
        },
    )

    _emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "leakage_checks", "status": "started", "message": "Running leakage checks"},
    )
    issues = run_leakage_checks(df, feature_columns=feature_cols)
    if issues:
        _emit_progress(
            progress_callback,
            {
                "kind": "pipeline",
                "stage": "leakage_checks",
                "status": "failed",
                "message": f"Leakage checks failed: {issues}",
            },
        )
        raise RuntimeError(f"Leakage checks failed: {issues}")
    _emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "leakage_checks", "status": "completed", "message": "Leakage checks passed"},
    )
    model_run_prefix = stable_hash({"feature_set_version": feature_set_version, "n_train": len(train_df), "ts": utc_now_iso()})

    model_dir = ensure_dir(Path(artifacts_dir) / "models" / model_run_prefix)
    glm_tune: dict = {"best_c": 1.0, "results": [], "fold_metrics": []}
    glm_best_c = 1.0
    if "glm_logit" in models_selected:
        _emit_progress(
            progress_callback,
            {"kind": "pipeline", "stage": "glm_tuning", "status": "started", "message": "Running GLM hyperparameter tuning"},
        )
        glm_tune = quick_tune_glm(train_df, glm_cols, n_splits=4, min_train_size=min(220, max(100, len(train_df) // 2)))
        glm_best_c = float(glm_tune.get("best_c", 1.0))
        _emit_progress(
            progress_callback,
            {
                "kind": "pipeline",
                "stage": "glm_tuning",
                "status": "completed",
                "message": "Completed GLM hyperparameter tuning",
                "glm_best_c": glm_best_c,
            },
        )
    _emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "fit_models", "status": "started", "message": "Fitting selected models"},
    )
    models, bayes_cols, bayes_diag, nn_included, used_feature_map = _fit_suite(
        train_df,
        feature_cols,
        artifacts_dir=artifacts_dir,
        bayes_cfg=bayes_cfg,
        selected_models=models_selected,
        progress_callback=progress_callback,
        allow_nn=True,
        glm_feature_cols=glm_cols,
        glm_c=glm_best_c,
        model_feature_columns=selected_model_feature_columns,
    )
    _emit_progress(
        progress_callback,
        {
            "kind": "pipeline",
            "stage": "fit_models",
            "status": "completed",
            "message": "Completed model fitting",
            "fitted_model_count": len(models),
        },
    )

    # Save models.
    for name, m in models.items():
        if hasattr(m, "save"):
            _emit_progress(
                progress_callback,
                {"kind": "model", "model": name, "stage": "save", "status": "started", "message": f"Saving {name} artifact"},
            )
            ext = "json" if name == "bayes_bt_state_space" else "joblib"
            m.save(model_dir / f"{name}.{ext}")
            _emit_progress(
                progress_callback,
                {
                    "kind": "model",
                    "model": name,
                    "stage": "save",
                    "status": "completed",
                    "message": f"Saved {name} artifact",
                },
            )

    oof = _oof_predictions(
        train_df,
        feature_cols,
        glm_cols,
        artifacts_dir=artifacts_dir,
        bayes_cfg=bayes_cfg,
        selected_models=models_selected,
        progress_callback=progress_callback,
        model_feature_columns=selected_model_feature_columns,
    )

    # Fit stacker.
    _emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "stacking", "status": "started", "message": "Preparing stacking ensemble"},
    )
    stacker = StackingEnsemble()
    stack_base_cols = [
        c
        for c in [
            "elo_baseline",
            "dynamic_rating",
            "glm_logit",
            "gbdt",
            "rf",
            "two_stage",
            "goals_poisson",
            "simulation_first",
            "bayes_bt_state_space",
            "bayes_goals",
            "nn_mlp",
        ]
        if c in oof.columns
    ]

    if not oof.empty and len(stack_base_cols) >= 3:
        stacker.fit(oof.dropna(subset=["home_win"]), base_columns=stack_base_cols, target_col="home_win")
        stack_ready = True
    else:
        stack_ready = False
    _emit_progress(
        progress_callback,
        {
            "kind": "pipeline",
            "stage": "stacking",
            "status": "completed",
            "message": "Stacking ensemble stage completed",
            "stack_ready": bool(stack_ready),
            "stack_base_count": len(stack_base_cols),
        },
    )

    oof_metrics = []
    if not oof.empty:
        y = oof["home_win"].astype(int).to_numpy()
        for col in [c for c in oof.columns if c not in {"game_id", "home_win", "game_date_utc"}]:
            m = metric_bundle(y, oof[col].to_numpy())
            oof_metrics.append(
                {
                    "model_name": col,
                    "log_loss": m["log_loss"],
                    "brier": m["brier"],
                    "ece": abs(m["accuracy"] - y.mean()),
                    "calibration_beta": 1.0,
                }
            )

    weights = compute_weights(pd.DataFrame(oof_metrics))

    # Predict upcoming and in-sample for diagnostics.
    _emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "predict_upcoming", "status": "started", "message": "Generating upcoming predictions"},
    )
    upcoming_preds, upcoming_extras = _predict_suite(
        models,
        upcoming_df,
        feature_cols,
        selected_models=models_selected,
        progress_callback=progress_callback,
        phase="predict_upcoming",
    )
    _emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "predict_upcoming", "status": "completed", "message": "Completed upcoming predictions"},
    )
    _emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "predict_train", "status": "started", "message": "Generating in-sample diagnostics"},
    )
    train_preds, _ = _predict_suite(
        models,
        train_df,
        feature_cols,
        selected_models=models_selected,
        progress_callback=progress_callback,
        phase="predict_train",
    )
    _emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "predict_train", "status": "completed", "message": "Completed in-sample diagnostics"},
    )

    model_cols = [c for c in upcoming_preds.columns if c != "game_id"]
    if not model_cols:
        raise RuntimeError(f"No model predictions were produced. selected_models={models_selected}")
    if not weights:
        weights = {c: 1.0 for c in model_cols}
    if stack_ready:
        stack_prob = stacker.predict_proba(upcoming_preds)
    else:
        stack_prob = weighted_ensemble(upcoming_preds, weights)

    weight_prob = weighted_ensemble(upcoming_preds, weights)
    ensemble_prob = np.clip(0.6 * stack_prob + 0.4 * weight_prob, 1e-6, 1 - 1e-6)
    _emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "ensemble", "status": "completed", "message": "Built ensemble probabilities"},
    )

    spread = spread_stats(upcoming_preds, model_cols)

    forecasts = upcoming_df[["game_id", "game_date_utc", "home_team", "away_team", "as_of_utc"]].copy()
    forecasts["ensemble_prob_home_win"] = ensemble_prob
    forecasts["predicted_winner"] = np.where(ensemble_prob >= 0.5, forecasts["home_team"], forecasts["away_team"])
    forecasts = pd.concat([forecasts.reset_index(drop=True), spread.reset_index(drop=True)], axis=1)
    forecasts["bayes_ci_low"] = upcoming_extras.get("bayes_low", np.full(len(forecasts), np.nan))
    forecasts["bayes_ci_high"] = upcoming_extras.get("bayes_high", np.full(len(forecasts), np.nan))

    flags = []
    nba_style_flags = "home_availability_uncertainty" in upcoming_df.columns or "fallback_shot_profile_proxy_used" in upcoming_df.columns
    for _, r in upcoming_df.iterrows():
        if nba_style_flags:
            game_flags = {
                "availability_uncertainty": bool(
                    (r.get("home_availability_uncertainty", 1) + r.get("away_availability_uncertainty", 1)) > 0
                ),
                "shot_profile_proxy_used": bool(r.get("fallback_shot_profile_proxy_used", 1) == 1),
                "availability_proxy_used": bool(r.get("fallback_availability_proxy_used", 1) == 1),
            }
        else:
            game_flags = {
                "starter_unknown": bool((r.get("home_goalie_uncertainty_feature", 1) + r.get("away_goalie_uncertainty_feature", 1)) > 0),
                "xg_unavailable": bool(r.get("fallback_xg_proxy_used", 1) == 1),
                "lineup_uncertainty": bool((r.get("home_lineup_uncertainty", 1) + r.get("away_lineup_uncertainty", 1)) > 0),
            }
        flags.append(json.dumps(game_flags, sort_keys=True))
    forecasts["uncertainty_flags_json"] = flags

    per_model_rows = []
    for _, fr in forecasts.iterrows():
        gid = fr["game_id"]
        p_row = upcoming_preds[upcoming_preds["game_id"] == gid].iloc[0]
        per_model = {c: float(p_row[c]) for c in model_cols}
        per_model_rows.append(json.dumps(per_model, sort_keys=True))
    forecasts["per_model_probs_json"] = per_model_rows

    # Save artifacts.
    _emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "save_artifacts", "status": "started", "message": "Saving training artifacts"},
    )
    try:
        forecasts.to_parquet(model_dir / "upcoming_forecasts.parquet", index=False)
    except Exception:
        forecasts.to_csv(model_dir / "upcoming_forecasts.csv", index=False)
    try:
        upcoming_preds.to_parquet(model_dir / "upcoming_model_probs.parquet", index=False)
    except Exception:
        upcoming_preds.to_csv(model_dir / "upcoming_model_probs.csv", index=False)
    if not oof.empty:
        try:
            oof.to_parquet(model_dir / "oof_predictions.parquet", index=False)
        except Exception:
            oof.to_csv(model_dir / "oof_predictions.csv", index=False)
    _emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "save_artifacts", "status": "completed", "message": "Saved training artifacts"},
    )

    run_payload = {
        "model_run_id": f"run_{model_run_prefix}",
        "feature_set_version": feature_set_version,
        "selected_models": models_selected,
        "feature_columns": feature_cols,
        "glm_feature_columns": glm_cols,
        "model_feature_columns": used_feature_map,
        "glm_tuning": glm_tune,
        "glm_best_c": glm_best_c,
        "bayes_feature_columns": bayes_cols,
        "stack_base_columns": stack_base_cols,
        "weights": weights,
        "nn_included": nn_included,
        "bayes_diagnostics": bayes_diag,
        "model_dir": str(model_dir),
    }
    (model_dir / "run_payload.json").write_text(json.dumps(run_payload, indent=2, sort_keys=True))

    # training metrics on train set for traceability
    train_metrics = {}
    if not train_preds.empty:
        y = train_df["home_win"].astype(int).to_numpy()
        for col in [c for c in train_preds.columns if c != "game_id"]:
            train_metrics[col] = metric_bundle(y, train_preds[col].to_numpy())

    _emit_progress(
        progress_callback,
        {
            "kind": "pipeline",
            "stage": "train_complete",
            "status": "completed",
            "message": "Model training completed",
            "model_run_id": run_payload["model_run_id"],
        },
    )

    return {
        "model_run_id": run_payload["model_run_id"],
        "feature_columns": feature_cols,
        "weights": weights,
        "oof_metrics": oof_metrics,
        "forecasts": forecasts,
        "upcoming_model_probs": upcoming_preds,
        "train_metrics": train_metrics,
        "stack_ready": stack_ready,
        "model_dir": str(model_dir),
        "run_payload": run_payload,
        "models": models,
        "train_df": train_df,
        "upcoming_df": upcoming_df,
    }
