from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.feature_contract import (
    resolve_training_feature_contract,
    validate_selected_feature_columns,
)
from src.training.feature_selection import select_feature_columns


def _training_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1, 2, 3],
            "start_time_utc": pd.date_range("2026-04-01", periods=3, freq="D").astype(str),
            "home_win": [1.0, 0.0, np.nan],
            "diff_starting_pitcher_quality": [0.5, -0.1, 0.2],
            "rest_diff": [1.0, 0.0, -1.0],
            "travel_diff": [0.2, -0.3, 0.1],
            "park_run_effect": [1.02, 0.98, 1.01],
            "market_offset_logit": [0.1, -0.2, 0.05],
            "prior_model_offset_logit": [0.0, -0.1, 0.2],
            "home_runs": [5, 3, np.nan],
            "notes": ["pregame", "pregame", "pregame"],
        }
    )


def test_training_feature_contract_preserves_explicit_order_and_model_maps() -> None:
    df = _training_feature_frame()
    requested_features = [
        "diff_starting_pitcher_quality",
        "rest_diff",
        "travel_diff",
        "park_run_effect",
        "market_offset_logit",
        "prior_model_offset_logit",
    ]

    contract = resolve_training_feature_contract(
        df,
        selected_models=["glm_lasso", "glm_lasso_prior_credibility", "glm_vanilla"],
        selected_feature_columns=requested_features,
        selected_model_feature_columns={
            "glm_lasso": ["rest_diff", "travel_diff"],
            "glm_lasso_prior_credibility": ["diff_starting_pitcher_quality", "rest_diff"],
            "glm_vanilla": ["park_run_effect", "rest_diff"],
        },
    )

    assert contract.feature_columns == requested_features
    assert contract.penalized_glm_feature_columns == {"glm_lasso": ["rest_diff", "travel_diff"]}
    assert contract.lasso_credibility_feature_columns == {
        "glm_lasso_prior_credibility": ["diff_starting_pitcher_quality", "rest_diff"]
    }
    assert contract.glm_feature_columns == ["rest_diff", "travel_diff"]


def test_training_feature_contract_matches_default_selection_rules() -> None:
    df = _training_feature_frame()

    contract = resolve_training_feature_contract(df, selected_models=["glm_vanilla"])

    assert contract.feature_columns == select_feature_columns(df)
    assert "home_runs" not in contract.feature_columns
    assert "notes" not in contract.feature_columns
    assert contract.glm_feature_columns


def test_validate_selected_feature_columns_fails_before_training_side_effects() -> None:
    df = _training_feature_frame()

    with pytest.raises(ValueError, match="selected_feature_columns includes missing columns: \\['missing_feature'\\]"):
        validate_selected_feature_columns(df, ["missing_feature"])

    with pytest.raises(ValueError, match="selected_feature_columns includes non-numeric columns: \\['notes'\\]"):
        validate_selected_feature_columns(df, ["notes"])
