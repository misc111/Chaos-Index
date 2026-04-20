import json

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.evaluation.validation_pipeline import build_validation_tasks, run_validation_pipeline
from src.evaluation.validation_stability import break_test_trade_deadline


def test_validation_pipeline_reports_honest_mlb_penalized_significance_and_contract(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.data.league = "MLB"

    rng = np.random.default_rng(2026)
    n = 260
    start = pd.date_range("2025-03-25", periods=n, freq="D")
    market_offset_logit = rng.normal(0.0, 0.45, n)
    diff_starter_era = rng.normal(0.0, 1.0, n)
    starting_pitcher_hand_matchup = rng.integers(0, 2, n)
    diff_bullpen_quality = rng.normal(0.0, 1.0, n)
    diff_lineup_talent = rng.normal(0.0, 1.0, n)
    park_weather_run_effect = rng.normal(0.0, 1.0, n)
    travel_diff = rng.normal(0.0, 1.0, n)
    rest_diff = rng.normal(0.0, 1.0, n)
    elo_home_prob = np.clip(0.5 + rng.normal(0.0, 0.08, n), 0.2, 0.8)
    logits = (
        0.85 * market_offset_logit
        - 0.55 * diff_starter_era
        + 0.45 * diff_bullpen_quality
        + 0.35 * diff_lineup_talent
        + 0.20 * park_weather_run_effect
        - 0.18 * travel_diff
        + 1.10 * (elo_home_prob - 0.5)
    )
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob)

    feature_cols = [
        "market_offset_logit",
        "diff_starter_era",
        "starting_pitcher_hand_matchup",
        "diff_bullpen_quality",
        "diff_lineup_talent",
        "park_weather_run_effect",
        "travel_diff",
        "rest_diff",
        "elo_home_prob",
    ]
    train_df = pd.DataFrame(
        {
            "start_time_utc": start.astype(str),
            "game_date_utc": start.date.astype(str),
            "home_win": y,
            "market_offset_logit": market_offset_logit,
            "diff_starter_era": diff_starter_era,
            "starting_pitcher_hand_matchup": starting_pitcher_hand_matchup,
            "diff_bullpen_quality": diff_bullpen_quality,
            "diff_lineup_talent": diff_lineup_talent,
            "park_weather_run_effect": park_weather_run_effect,
            "travel_diff": travel_diff,
            "rest_diff": rest_diff,
            "elo_home_prob": elo_home_prob,
        }
    )

    result = {
        "models": {},
        "train_df": train_df,
        "feature_columns": feature_cols,
        "run_payload": {
            "model_run_id": "run_mlb_validation_honesty",
            "feature_set_version": "fset_mlb_validation_honesty",
            "selected_models": ["glm_lasso"],
            "glm_primary_model": "glm_lasso",
            "model_feature_columns": {"glm_lasso": feature_cols},
            "glm_tuning_by_model": {"glm_lasso": {"best_c": 0.7}},
            "credibility_metadata_by_model": {
                "glm_lasso": {
                    "complement_kind": "market",
                    "complement_label": "Vig-free market logit",
                    "complement_column": "market_offset_logit",
                    "offset_scale": "logit",
                    "driver_features": ["diff_starter_era", "diff_bullpen_quality"],
                    "complement_summary": {"source": "vig_free_market"},
                }
            },
        },
    }

    tasks = [task for task in build_validation_tasks() if task.name in {"significance", "fragility"}]
    run_validation_pipeline(result, cfg, tasks=tasks)

    validation_root = tmp_path / "artifacts" / "validation" / "mlb"
    significance = pd.read_csv(validation_root / "diagnostics" / "significance" / "validation_significance.csv")
    assert {
        "market_credibility_block",
        "starting_pitcher_block",
        "bullpen_block",
        "lineup_block",
        "park_weather_block",
        "schedule_travel_block",
        "rating_block",
    }.issubset(set(significance["block"]))
    assert significance["p_value"].isna().all()
    assert significance["inference_applicability"].eq("not_applicable").all()
    assert significance.loc[significance["block"] == "market_credibility_block", "block_role"].eq("complement").all()

    fragility = pd.read_csv(validation_root / "diagnostics" / "fragility" / "validation_fragility_missingness.csv")
    assert fragility["scenario"].tolist() == [
        "market_complement_removed",
        "unknown_starter_context",
        "bullpen_context_removed",
        "lineup_context_removed",
        "park_weather_removed",
    ]
    market_row = fragility.set_index("scenario").loc["market_complement_removed"]
    assert market_row["scenario_status"] == "ok"
    assert market_row["fill_strategy"] == "zero"

    contract_payload = json.loads((validation_root / "validation_outputs_contract.json").read_text())
    records = {record["task_name"]: record for record in contract_payload["validation_outputs"]}
    assert records["significance"]["applicability"] == "partial"
    assert records["significance"]["summary"]["penalty"]["penalty_family"] == "lasso"
    assert records["significance"]["summary"]["credibility"]["complement_column"] == "market_offset_logit"
    assert records["fragility"]["summary"]["scenario_count"] == 5


def test_break_test_trade_deadline_uses_mlb_calendar_cut():
    rng = np.random.default_rng(11)
    n = 180
    dates = pd.date_range("2025-05-01", periods=n, freq="D")
    signal = rng.normal(0.0, 1.0, n)
    logits = 0.9 * signal
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob)
    df = pd.DataFrame(
        {
            "game_date_utc": dates.astype(str),
            "home_win": y,
            "signal": signal,
        }
    )

    report = break_test_trade_deadline(df, features=["signal"], league="MLB", model_name="glm_ridge", c=1.0)

    assert report["deadline"] == "2025-07-31"
    assert report["n_pre"] > 0
    assert report["n_post"] > 0
