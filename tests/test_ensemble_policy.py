import numpy as np
import pandas as pd
import pytest

from src.models.ensemble_stack import StackingEnsemble
from src.training.ensemble_builder import build_ensemble_outputs, fit_stacker
from src.training.ensemble_policy import demoted_ensemble_models, ensemble_component_columns


def test_ensemble_component_columns_demotes_nba_simulation_first():
    cols = ["elo_baseline", "simulation_first", "rf"]

    assert demoted_ensemble_models(league="NBA") == ["simulation_first"]
    assert ensemble_component_columns(cols, league="NBA") == ["elo_baseline", "rf"]
    assert ensemble_component_columns(cols, league="NHL") == cols


def test_ensemble_component_columns_falls_back_when_everything_is_demoted():
    assert ensemble_component_columns(["simulation_first"], league="NBA") == ["simulation_first"]


def test_fit_stacker_excludes_nba_simulation_first():
    oof = pd.DataFrame(
        {
            "home_win": [0, 1, 0, 1, 0, 1, 0, 1],
            "elo_baseline": [0.32, 0.68, 0.35, 0.66, 0.4, 0.6, 0.38, 0.7],
            "glm_ridge": [0.3, 0.7, 0.34, 0.64, 0.41, 0.61, 0.36, 0.72],
            "rf": [0.33, 0.69, 0.37, 0.67, 0.42, 0.58, 0.39, 0.71],
            "simulation_first": [0.49, 0.51, 0.5, 0.52, 0.48, 0.5, 0.51, 0.49],
        }
    )

    stacker, stack_ready, stack_base_cols = fit_stacker(oof, league="NBA")

    assert stack_ready is True
    assert stack_base_cols == ["elo_baseline", "glm_ridge", "rf"]
    assert stacker.base_columns == ["elo_baseline", "glm_ridge", "rf"]


def test_build_ensemble_outputs_excludes_nba_simulation_first_from_weights_and_spread():
    upcoming_preds = pd.DataFrame(
        {
            "game_id": [1, 2],
            "elo_baseline": [0.65, 0.35],
            "glm_ridge": [0.62, 0.38],
            "rf": [0.6, 0.4],
            "simulation_first": [0.51, 0.49],
        }
    )
    oof_metrics = [
        {"model_name": "elo_baseline", "log_loss": 0.62, "brier": 0.21, "ece": 0.03, "calibration_beta": 1.0},
        {"model_name": "glm_ridge", "log_loss": 0.63, "brier": 0.215, "ece": 0.035, "calibration_beta": 1.0},
        {"model_name": "rf", "log_loss": 0.64, "brier": 0.22, "ece": 0.04, "calibration_beta": 1.0},
        {"model_name": "simulation_first", "log_loss": 0.7, "brier": 0.25, "ece": 0.02, "calibration_beta": 1.0},
    ]

    _, nba_weights, nba_spread = build_ensemble_outputs(
        upcoming_preds,
        oof_metrics,
        StackingEnsemble(),
        False,
        league="NBA",
    )
    _, nhl_weights, nhl_spread = build_ensemble_outputs(
        upcoming_preds,
        oof_metrics,
        StackingEnsemble(),
        False,
        league="NHL",
    )

    assert set(nba_weights) == {"elo_baseline", "glm_ridge", "rf"}
    assert "simulation_first" not in nba_weights
    assert "simulation_first" in nhl_weights
    assert nba_spread.loc[0, "spread_mean"] == pytest.approx(np.mean([0.65, 0.62, 0.6]))
    assert nhl_spread.loc[0, "spread_mean"] == pytest.approx(np.mean([0.65, 0.62, 0.6, 0.51]))
