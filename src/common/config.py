from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "nhl_forecast"
    timezone: str = "America/Chicago"


class PathsConfig(BaseModel):
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"
    db_path: str = "data/processed/nhl_forecast.db"


class DataConfig(BaseModel):
    league: str = "NHL"
    season_start: str
    season_end: str
    history_days: int = 220
    upcoming_days: int = 14
    offline_mode: bool = False
    timeout_seconds: int = 30
    max_retries: int = 3
    backoff_seconds: float = 1.5


class ModelingConfig(BaseModel):
    random_seed: int = 42
    cv_splits: int = 5
    rolling_windows_days: list[int] = Field(default_factory=lambda: [7, 30, 60, 90])
    calibration_bins: int = 10


class BayesConfig(BaseModel):
    process_variance: float = 0.08
    prior_variance: float = 1.5
    observation_scale: float = 1.0
    posterior_draws: int = 400


class RuntimeConfig(BaseModel):
    retrain_daily: bool = True
    embargo_days: int = 1


class FeaturePolicyConfig(BaseModel):
    # production: block feature entry/exit unless explicitly approved via CLI flag.
    # research: do not block, but track newly discovered features as candidates.
    mode: Literal["production", "research"] = "production"
    registry_path: str = "configs/feature_registry_{league}.yaml"


class AppConfig(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    data: DataConfig
    modeling: ModelingConfig
    bayes: BayesConfig
    runtime: RuntimeConfig
    feature_policy: FeaturePolicyConfig = Field(default_factory=FeaturePolicyConfig)


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    data = yaml.safe_load(path.read_text()) or {}

    extends = data.get("extends")
    if extends:
        parent_path = Path(extends)
        if not parent_path.is_absolute():
            candidate = (path.parent / parent_path).resolve()
            if candidate.exists():
                parent_path = candidate
            else:
                parent_path = Path(extends).resolve()
        parent_data = yaml.safe_load(parent_path.read_text()) or {}
        data = _deep_update(parent_data, {k: v for k, v in data.items() if k != "extends"})

    return AppConfig.model_validate(data)
