"""Validation context assembly for the MLB diagnostics pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import AppConfig
from src.common.utils import ensure_dir
from src.evaluation.validation_contract import ValidationModelMetadata, resolve_validation_model_metadata
from src.evaluation.validation_models import (
    _fit_validation_models,
    _resolve_primary_validation_model,
    _selected_validation_models,
    _validation_feature_columns,
)
from src.evaluation.validation_splits import (
    ValidationSplitPlan,
    concat_validation_frames as _concat_frames,
    resolve_validation_split as _validation_split,
)

def _canonical_league(league: str | None) -> str:
    token = str(league or "").strip().upper()
    if token == "MLB":
        return token
    raise ValueError(f"Unsupported league '{league}'. Expected only: MLB.")


@dataclass(slots=True)
class ValidationContext:
    cfg: AppConfig
    run_payload: dict[str, Any]
    models: dict[str, object]
    train_df: pd.DataFrame
    feature_cols: list[str]
    out_dir: Path
    plots_dir: Path
    league: str
    tr: pd.DataFrame
    va: pd.DataFrame
    te: pd.DataFrame
    split_plan: "ValidationSplitPlan"
    fit_df: pd.DataFrame
    glm: object | None
    diagnostic_feature_cols: list[str]
    glm_validation_metadata: ValidationModelMetadata
    primary_model_resolution: dict[str, Any]

    @classmethod
    def from_result(cls, result: dict[str, Any], cfg: AppConfig) -> "ValidationContext":
        train_df = result["train_df"].copy()
        feature_cols = list(result["feature_columns"])
        run_payload = result.get("run_payload", {})
        selected_models = _selected_validation_models(result, run_payload)
        split_plan, tr, va, te = _validation_split(train_df, cfg)
        fit_df = _concat_frames(tr, va)
        models, model_fit_status = _fit_validation_models(
            fit_df,
            feature_cols=feature_cols,
            selected_models=selected_models,
            run_payload=run_payload,
        )
        league = _canonical_league(cfg.data.league)
        out_dir = ensure_dir(Path(cfg.paths.artifacts_dir) / "validation" / league.lower())
        plots_dir = ensure_dir(Path(cfg.paths.artifacts_dir) / "plots" / league.lower() / "glm" / "performance")
        glm_name, glm, primary_model_resolution = _resolve_primary_validation_model(
            selected_models=selected_models,
            models=models,
            model_fit_status=model_fit_status,
            run_payload=run_payload,
        )
        diagnostic_feature_cols = list(
            getattr(glm, "feature_columns", [])
            or _validation_feature_columns(feature_cols, run_payload, glm_name or "glm_vanilla")
            or feature_cols[: min(40, len(feature_cols))]
        )
        glm_validation_metadata = resolve_validation_model_metadata(
            run_payload={**run_payload, "glm_primary_model": glm_name or run_payload.get("glm_primary_model")},
            model=glm,
            model_name=glm_name,
        )

        return cls(
            cfg=cfg,
            run_payload=run_payload,
            models=models,
            train_df=train_df,
            feature_cols=feature_cols,
            out_dir=out_dir,
            plots_dir=plots_dir,
            league=league,
            tr=tr,
            va=va,
            te=te,
            split_plan=split_plan,
            fit_df=fit_df,
            glm=glm,
            diagnostic_feature_cols=diagnostic_feature_cols,
            glm_validation_metadata=glm_validation_metadata,
            primary_model_resolution=primary_model_resolution,
        )

def _holdout_df(ctx: ValidationContext) -> pd.DataFrame:
    return ctx.te if not ctx.te.empty else ctx.va


def _date_bounds(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if df.empty:
        return None, None
    for col in ("start_time_utc", "game_date_utc"):
        if col in df.columns:
            values = df[col].dropna().astype(str)
            if not values.empty:
                return str(values.min()), str(values.max())
    return None, None


canonical_league = _canonical_league
holdout_df = _holdout_df
date_bounds = _date_bounds
