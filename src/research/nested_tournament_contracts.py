"""Shared contracts and low-level helpers for nested MLB tournaments.

The nested tournament pipeline has several responsibilities: candidate
construction, validation-lane evaluation, governance gates, reporting, and CLI
orchestration.  This module keeps the small immutable contracts and pure helpers
that all of those layers depend on, avoiding import cycles between the more
specific modules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.utils import ensure_dir
from src.research.candidate_models import BaseCandidateModel
from src.research.mlb_targets import TargetDefinition

MARKET_BASELINE_FAMILIES = ("market_baseline",)
CORE_LINEAR_FAMILIES = ("glm_vanilla", "glm_ridge", "glm_lasso", "glm_elastic_net")
EXTENSION_FAMILIES = ("glmm_logit", "dglm_margin", "gam_spline")
DEFAULT_NESTED_MODELS = MARKET_BASELINE_FAMILIES + CORE_LINEAR_FAMILIES + EXTENSION_FAMILIES
DEFAULT_STRUCTURED_SPEC_PATH = "configs/research/mlb_tournament_structured_glm.yaml"


@dataclass(frozen=True, slots=True)
class NestedTournamentResult:
    run_id: str
    artifact_root: Path
    summary_path: Path
    target_coverage_path: Path
    variant_leaderboard_path: Path
    family_champions_path: Path
    inter_family_leaderboard_path: Path
    current_best_models_path: Path


@dataclass(frozen=True, slots=True)
class FeatureVariant:
    variant_key: str
    display_name: str
    source: str
    features: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NestedCandidateSpec:
    target: TargetDefinition
    model_name: str
    display_name: str
    variant_key: str
    variant_display_name: str
    feature_source: str
    features: tuple[str, ...]
    param_grid: tuple[dict[str, Any], ...]
    builder: Callable[[dict[str, Any]], BaseCandidateModel]

    @property
    def candidate_key(self) -> str:
        return f"{self.target.target_name}::{self.model_name}::{self.variant_key}"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        numeric = float(value)
    except Exception:
        return None
    return numeric if np.isfinite(numeric) else None


def _safe_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    if isinstance(value, tuple):
        return [_json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if pd.isna(value) if not isinstance(value, (str, bytes, bool)) else False:
        return None
    return value


def _split_csv(raw_value: str | None, default: tuple[str, ...]) -> list[str]:
    raw = str(raw_value or "").strip()
    if not raw or raw.lower() in {"all", "*"}:
        return list(default)
    return [token.strip() for token in raw.split(",") if token.strip()]


def _time_ordered_split(
    df: pd.DataFrame,
    *,
    target_col: str,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df[df[target_col].notna()].copy().sort_values("start_time_utc").reset_index(drop=True)
    n_obs = len(work)
    if n_obs < 12:
        return work.iloc[:0].copy(), work.iloc[:0].copy(), work.iloc[:0].copy()
    train_end = int(round(train_fraction * n_obs))
    valid_end = int(round((train_fraction + validation_fraction) * n_obs))
    train_end = min(max(train_end, 1), max(n_obs - 2, 1))
    valid_end = min(max(valid_end, train_end + 1), n_obs - 1)
    return work.iloc[:train_end].copy(), work.iloc[train_end:valid_end].copy(), work.iloc[valid_end:].copy()
