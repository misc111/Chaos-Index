"""Model fitting runner separated from training orchestration."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.bayes.fit_offline import run_bayes_offline_fit
from src.evaluation.metrics import metric_bundle
from src.models.bayes_state_space_goals import BayesGoalsModel
from src.models.gbdt import GBDTModel
from src.models.glm_goals import GoalsPoissonModel
from src.models.glm_ridge import GLMRidgeModel
from src.models.nn import NNModel
from src.models.rf import RFModel
from src.models.two_stage import TwoStageModel
from src.training.feature_selection import bayes_feature_subset, resolve_model_feature_columns
from src.training.progress import ProgressCallback, emit_progress


def fit_model_suite(
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
    metric_bundle_fn=metric_bundle,
):
    models: dict[str, object] = {}
    selected = set(selected_models)
    used_feature_map: dict[str, list[str]] = {}

    glm_cols = resolve_model_feature_columns(
        feature_cols,
        model_name="glm_ridge",
        model_feature_columns=model_feature_columns,
        fallback_columns=glm_feature_cols if glm_feature_cols else feature_cols,
    )
    if "glm_ridge" in selected:
        emit_progress(
            progress_callback,
            {"kind": "model", "model": "glm_ridge", "stage": "fit", "status": "started", "message": "Fitting glm_ridge"},
        )
        glm = GLMRidgeModel(c=float(glm_c))
        glm.fit(train_df, glm_cols)
        models[glm.model_name] = glm
        used_feature_map[glm.model_name] = glm_cols
        emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "glm_ridge",
                "stage": "fit",
                "status": "completed",
                "message": "Completed glm_ridge fit",
            },
        )

    gbdt = None
    if "gbdt" in selected:
        emit_progress(
            progress_callback,
            {"kind": "model", "model": "gbdt", "stage": "fit", "status": "started", "message": "Fitting gbdt"},
        )
        gbdt_cols = resolve_model_feature_columns(
            feature_cols,
            model_name="gbdt",
            model_feature_columns=model_feature_columns,
            fallback_columns=feature_cols,
        )
        gbdt = GBDTModel()
        gbdt.fit(train_df, gbdt_cols)
        models[gbdt.model_name] = gbdt
        used_feature_map[gbdt.model_name] = gbdt_cols
        emit_progress(
            progress_callback,
            {"kind": "model", "model": "gbdt", "stage": "fit", "status": "completed", "message": "Completed gbdt fit"},
        )

    if "rf" in selected:
        emit_progress(
            progress_callback,
            {"kind": "model", "model": "rf", "stage": "fit", "status": "started", "message": "Fitting rf"},
        )
        rf_cols = resolve_model_feature_columns(
            feature_cols,
            model_name="rf",
            model_feature_columns=model_feature_columns,
            fallback_columns=feature_cols,
        )
        rf = RFModel()
        rf.fit(train_df, rf_cols)
        models[rf.model_name] = rf
        used_feature_map[rf.model_name] = rf_cols
        emit_progress(
            progress_callback,
            {"kind": "model", "model": "rf", "stage": "fit", "status": "completed", "message": "Completed rf fit"},
        )

    if "two_stage" in selected:
        emit_progress(
            progress_callback,
            {"kind": "model", "model": "two_stage", "stage": "fit", "status": "started", "message": "Fitting two_stage"},
        )
        two_stage_cols = resolve_model_feature_columns(
            feature_cols,
            model_name="two_stage",
            model_feature_columns=model_feature_columns,
            fallback_columns=feature_cols,
        )
        two_stage = TwoStageModel()
        two_stage.fit(train_df, two_stage_cols)
        models[two_stage.model_name] = two_stage
        used_feature_map[two_stage.model_name] = two_stage_cols
        emit_progress(
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
        emit_progress(
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
        emit_progress(
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
        emit_progress(
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
        emit_progress(
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
    bayes_diag: dict[str, Any] = {}
    if "bayes_bt_state_space" in selected:
        emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "bayes_bt_state_space",
                "stage": "fit",
                "status": "started",
                "message": "Fitting bayes_bt_state_space",
            },
        )
        bcols = resolve_model_feature_columns(
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
        emit_progress(
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
        emit_progress(
            progress_callback,
            {"kind": "model", "model": "nn_mlp", "stage": "fit_gate", "status": "started", "message": "Evaluating nn_mlp gate"},
        )
        split_ix = int(len(train_df) * 0.85)
        tr = train_df.iloc[:split_ix]
        va = train_df.iloc[split_ix:]
        if not va.empty and va["home_win"].nunique() > 1:
            emit_progress(
                progress_callback,
                {"kind": "model", "model": "nn_mlp", "stage": "fit", "status": "started", "message": "Fitting nn_mlp"},
            )
            nn_cols = resolve_model_feature_columns(
                feature_cols,
                model_name="nn_mlp",
                model_feature_columns=model_feature_columns,
                fallback_columns=feature_cols,
            )
            gate_nn = NNModel()
            gate_nn.fit(tr, nn_cols)
            include_nn = True
            if gbdt is not None:
                gate_gbdt_cols = resolve_model_feature_columns(
                    feature_cols,
                    model_name="gbdt",
                    model_feature_columns=model_feature_columns,
                    fallback_columns=feature_cols,
                )
                gate_gbdt = GBDTModel()
                gate_gbdt.fit(tr, gate_gbdt_cols)
                nn_p = gate_nn.predict_proba(va)
                gbdt_p = gate_gbdt.predict_proba(va)
                m_nn = metric_bundle_fn(va["home_win"].to_numpy(), nn_p)
                m_g = metric_bundle_fn(va["home_win"].to_numpy(), gbdt_p)
                include_nn = m_nn["log_loss"] + 0.001 < m_g["log_loss"]
            if include_nn:
                nn = NNModel()
                nn.fit(train_df, nn_cols)
                models[nn.model_name] = nn
                nn_included = True
                used_feature_map[nn.model_name] = nn_cols
                emit_progress(
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
                emit_progress(
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
            emit_progress(
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
        emit_progress(
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
