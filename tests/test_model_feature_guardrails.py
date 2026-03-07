import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.training.model_feature_research import (
    load_model_feature_map,
    research_model_feature_map,
    save_model_feature_map,
)


def _guardrails_template(tmp_path) -> str:
    return str(tmp_path / "model_feature_guardrails_{league}.yaml")


def _map_template(tmp_path) -> str:
    return str(tmp_path / "model_feature_map_{league}.yaml")


def _write_guardrails(tmp_path, league: str, blocked_features: dict[str, dict]) -> None:
    payload = {
        "version": 1,
        "league": league,
        "updated_at_utc": "2026-03-06T15:10:00+00:00",
        "models": {
            "glm_ridge": {
                "blocked_features": blocked_features,
                "watchlist_features": {},
                "watchlist_pairs": [],
            }
        },
    }
    path = tmp_path / f"model_feature_guardrails_{league.lower()}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _synthetic_nba_glm_frame(n: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(314)
    skill = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    start = pd.date_range("2025-10-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "game_id": np.arange(1, n + 1),
            "start_time_utc": start.astype(str),
            "game_date_utc": start.date.astype(str),
            "home_win": (0.95 * skill + 0.25 * noise > 0).astype(int),
            "diff_form_point_margin": 1.1 * skill + rng.normal(0, 0.1, n),
            "rest_diff": 0.2 * skill + rng.normal(0, 0.2, n),
            "discipline_free_throw_pressure_diff": 0.8 * skill + rng.normal(0, 0.15, n),
            "discipline_foul_margin_diff": 0.7 * skill + rng.normal(0, 0.15, n),
            "elo_home_prob": 0.55 + 0.10 * np.tanh(skill),
            "dyn_home_prob": 0.55 + 0.12 * np.tanh(skill),
            "arena_margin_effect": 0.4 * skill + rng.normal(0, 0.2, n),
            "diff_shot_volume_share": 0.05 * np.tanh(skill) + rng.normal(0, 0.01, n),
        }
    )


def test_load_model_feature_map_rejects_blocked_features(tmp_path) -> None:
    _write_guardrails(tmp_path, "NBA", {"dyn_home_prob": {"decision": "blocked"}})
    path = tmp_path / "model_feature_map_nba.yaml"
    path.write_text(
        """
version: 1
league: NBA
updated_at_utc: '2026-03-06T15:10:00+00:00'
models:
  glm_ridge:
    active_features:
    - elo_home_prob
    - dyn_home_prob
    feature_count: 2
"""
    )

    with pytest.raises(RuntimeError, match="dyn_home_prob"):
        load_model_feature_map(
            "NBA",
            path_template=_map_template(tmp_path),
            guardrails_path_template=_guardrails_template(tmp_path),
        )


def test_save_model_feature_map_filters_blocked_features(tmp_path) -> None:
    _write_guardrails(tmp_path, "NBA", {"dyn_home_prob": {"decision": "blocked"}})

    path = save_model_feature_map(
        "NBA",
        {"glm_ridge": ["elo_home_prob", "dyn_home_prob", "rest_diff"]},
        path_template=_map_template(tmp_path),
        guardrails_path_template=_guardrails_template(tmp_path),
    )

    raw = yaml.safe_load(path.read_text())
    assert raw["models"]["glm_ridge"]["active_features"] == ["elo_home_prob", "rest_diff"]
    assert raw["models"]["glm_ridge"]["feature_count"] == 2


def test_research_model_feature_map_records_guardrail_exclusions(tmp_path) -> None:
    _write_guardrails(tmp_path, "NBA", {"dyn_home_prob": {"decision": "blocked"}})
    df = _synthetic_nba_glm_frame()

    result = research_model_feature_map(
        df,
        league="NBA",
        artifacts_dir=str(tmp_path / "artifacts"),
        feature_columns=[c for c in df.columns if c not in {"game_id", "start_time_utc", "game_date_utc", "home_win"}],
        selected_models=["glm_ridge"],
        approve_changes=True,
        path_template=_map_template(tmp_path),
        guardrails_path_template=_guardrails_template(tmp_path),
    )

    saved = load_model_feature_map(
        "NBA",
        path_template=_map_template(tmp_path),
        guardrails_path_template=_guardrails_template(tmp_path),
    )
    report_path = next((tmp_path / "artifacts" / "research").glob("nba_model_feature_research_*.json"))
    report = json.loads(report_path.read_text())

    assert "dyn_home_prob" not in result.approved_model_features["glm_ridge"]
    assert "dyn_home_prob" not in saved["glm_ridge"]
    assert report["guardrail_exclusions"]["glm_ridge"] == ["dyn_home_prob"]
    assert report["model_feature_guardrails_path"].endswith("model_feature_guardrails_nba.yaml")
