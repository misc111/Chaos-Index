from __future__ import annotations

import json
from pathlib import Path
import re

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
        "shots_for",
        "shots_against",
        "penalties_taken",
        "penalties_drawn",
        "pp_goals",
        "starter_save_pct",
        "goalie_quality_raw",
        "team_save_pct_proxy",
        "xg_share_proxy",
        "penalty_diff_proxy",
        "pace_proxy",
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
        if ("goals_for" in c or "goals_against" in c) and not any(m in c for m in lag_markers):
            continue
        if any(tok in c for tok in direct_event_tokens) and not any(m in c for m in lag_markers):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols



def bayes_feature_subset(feature_cols: list[str]) -> list[str]:
    keep = []
    keywords = ["diff_", "travel", "rest", "goalie", "special", "rink", "elo", "dyn", "lineup"]
    for c in feature_cols:
        if any(k in c for k in keywords):
            keep.append(c)
    if len(keep) < 12:
        keep = feature_cols[: min(40, len(feature_cols))]
    return keep



def glm_feature_subset(feature_cols: list[str]) -> list[str]:
    keep_exact = {"travel_diff", "rest_diff", "rink_goal_effect", "rink_shot_effect"}
    keep_prefix = ("diff_", "special_", "goalie_", "elo_", "dyn_")

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



def _fit_suite(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    artifacts_dir: str,
    bayes_cfg: dict,
    selected_models: list[str],
    allow_nn: bool = True,
    glm_feature_cols: list[str] | None = None,
    glm_c: float = 1.0,
):
    models: dict[str, object] = {}
    selected = set(selected_models)

    glm_cols = glm_feature_cols if glm_feature_cols else feature_cols
    if "glm_logit" in selected:
        glm = GLMLogitModel(c=float(glm_c))
        glm.fit(train_df, glm_cols)
        models[glm.model_name] = glm

    gbdt = None
    if "gbdt" in selected:
        gbdt = GBDTModel()
        gbdt.fit(train_df, feature_cols)
        models[gbdt.model_name] = gbdt

    if "rf" in selected:
        rf = RFModel()
        rf.fit(train_df, feature_cols)
        models[rf.model_name] = rf

    if "two_stage" in selected:
        two_stage = TwoStageModel()
        two_stage.fit(train_df, feature_cols)
        models[two_stage.model_name] = two_stage

    if "goals_poisson" in selected:
        goals = GoalsPoissonModel()
        goals.fit(train_df)
        models[goals.model_name] = goals

    if "bayes_goals" in selected:
        bayes_goals = BayesGoalsModel()
        bayes_goals.fit(train_df)
        models[bayes_goals.model_name] = bayes_goals

    bcols: list[str] = []
    bayes_diag: dict = {}
    if "bayes_bt_state_space" in selected:
        bcols = bayes_feature_subset(feature_cols)
        bayes_model, bayes_diag = run_bayes_offline_fit(
            features_df=train_df,
            feature_columns=bcols,
            artifacts_dir=artifacts_dir,
            process_variance=bayes_cfg.get("process_variance", 0.08),
            prior_variance=bayes_cfg.get("prior_variance", 1.5),
            draws=bayes_cfg.get("posterior_draws", 500),
        )
        models[bayes_model.model_name] = bayes_model

    nn_included = False
    if "nn_mlp" in selected and allow_nn and len(train_df) >= 350:
        # Gate NN by quick holdout improvement versus GBDT.
        split_ix = int(len(train_df) * 0.85)
        tr = train_df.iloc[:split_ix]
        va = train_df.iloc[split_ix:]
        if not va.empty and va["home_win"].nunique() > 1:
            nn = NNModel()
            nn.fit(tr, feature_cols)
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

    return models, bcols, bayes_diag, nn_included



def _predict_suite(
    models: dict[str, object],
    df: pd.DataFrame,
    feature_cols: list[str],
    selected_models: list[str],
) -> tuple[pd.DataFrame, dict]:
    out = pd.DataFrame({"game_id": df["game_id"].values})
    extras: dict = {}
    selected = set(selected_models)

    # Direct feature baselines.
    if "elo_baseline" in selected:
        out["elo_baseline"] = np.clip(df.get("elo_home_prob", 0.5).to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    if "dynamic_rating" in selected:
        out["dynamic_rating"] = np.clip(df.get("dyn_home_prob", 0.5).to_numpy(dtype=float), 1e-6, 1 - 1e-6)

    for name, m in models.items():
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

    if "simulation_first" in selected:
        sim = GameSimulator(seed=42)
        sim_df = sim.simulate_dataframe(df, n_sims=3500)
        out = out.merge(sim_df[["game_id", "sim_prob_home_win"]], on="game_id", how="left")
        out = out.rename(columns={"sim_prob_home_win": "simulation_first"})

    return out, extras



def _oof_predictions(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    glm_feature_cols: list[str],
    artifacts_dir: str,
    bayes_cfg: dict,
    selected_models: list[str],
) -> pd.DataFrame:
    splits = time_series_splits(train_df, n_splits=5, min_train_size=min(220, max(80, len(train_df) // 2)))
    rows = []
    selected = set(selected_models)

    for tr_idx, va_idx in splits:
        tr = train_df.loc[tr_idx].copy().sort_values("start_time_utc")
        va = train_df.loc[va_idx].copy().sort_values("start_time_utc")
        if tr.empty or va.empty:
            continue
        fold_glm_c = 1.0
        if "glm_logit" in selected:
            tune = quick_tune_glm(
                tr,
                glm_feature_cols,
                n_splits=3,
                min_train_size=min(140, max(70, len(tr) // 2)),
            )
            fold_glm_c = float(tune.get("best_c", 1.0))
        models, _, _, _ = _fit_suite(
            tr,
            feature_cols,
            artifacts_dir=artifacts_dir,
            bayes_cfg=bayes_cfg,
            selected_models=selected_models,
            allow_nn=False,
            glm_feature_cols=glm_feature_cols,
            glm_c=fold_glm_c,
        )
        pred, _ = _predict_suite(models, va, feature_cols, selected_models=selected_models)
        pred["home_win"] = va["home_win"].to_numpy()
        pred["game_date_utc"] = va["game_date_utc"].to_numpy()
        rows.append(pred)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)



def train_and_predict(
    features_df: pd.DataFrame,
    feature_set_version: str,
    artifacts_dir: str,
    bayes_cfg: dict,
    selected_models: list[str] | None = None,
) -> dict:
    df = features_df.sort_values("start_time_utc").copy()
    train_df = df[df["home_win"].notna()].copy()
    upcoming_df = df[df["home_win"].isna()].copy()
    models_selected = normalize_selected_models(selected_models)

    feature_cols = select_feature_columns(df)
    glm_cols = glm_feature_subset(feature_cols)
    issues = run_leakage_checks(df, feature_columns=feature_cols)
    if issues:
        raise RuntimeError(f"Leakage checks failed: {issues}")
    model_run_prefix = stable_hash({"feature_set_version": feature_set_version, "n_train": len(train_df), "ts": utc_now_iso()})

    model_dir = ensure_dir(Path(artifacts_dir) / "models" / model_run_prefix)
    glm_tune: dict = {"best_c": 1.0, "results": [], "fold_metrics": []}
    glm_best_c = 1.0
    if "glm_logit" in models_selected:
        glm_tune = quick_tune_glm(train_df, glm_cols, n_splits=4, min_train_size=min(220, max(100, len(train_df) // 2)))
        glm_best_c = float(glm_tune.get("best_c", 1.0))
    models, bayes_cols, bayes_diag, nn_included = _fit_suite(
        train_df,
        feature_cols,
        artifacts_dir=artifacts_dir,
        bayes_cfg=bayes_cfg,
        selected_models=models_selected,
        allow_nn=True,
        glm_feature_cols=glm_cols,
        glm_c=glm_best_c,
    )

    # Save models.
    for name, m in models.items():
        if hasattr(m, "save"):
            ext = "json" if name == "bayes_bt_state_space" else "joblib"
            m.save(model_dir / f"{name}.{ext}")

    oof = _oof_predictions(
        train_df,
        feature_cols,
        glm_cols,
        artifacts_dir=artifacts_dir,
        bayes_cfg=bayes_cfg,
        selected_models=models_selected,
    )

    # Fit stacker.
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
    upcoming_preds, upcoming_extras = _predict_suite(models, upcoming_df, feature_cols, selected_models=models_selected)
    train_preds, _ = _predict_suite(models, train_df, feature_cols, selected_models=models_selected)

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

    spread = spread_stats(upcoming_preds, model_cols)

    forecasts = upcoming_df[["game_id", "game_date_utc", "home_team", "away_team", "as_of_utc"]].copy()
    forecasts["ensemble_prob_home_win"] = ensemble_prob
    forecasts["predicted_winner"] = np.where(ensemble_prob >= 0.5, forecasts["home_team"], forecasts["away_team"])
    forecasts = pd.concat([forecasts.reset_index(drop=True), spread.reset_index(drop=True)], axis=1)
    forecasts["bayes_ci_low"] = upcoming_extras.get("bayes_low", np.full(len(forecasts), np.nan))
    forecasts["bayes_ci_high"] = upcoming_extras.get("bayes_high", np.full(len(forecasts), np.nan))

    flags = []
    for _, r in upcoming_df.iterrows():
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

    run_payload = {
        "model_run_id": f"run_{model_run_prefix}",
        "feature_set_version": feature_set_version,
        "selected_models": models_selected,
        "feature_columns": feature_cols,
        "glm_feature_columns": glm_cols,
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
