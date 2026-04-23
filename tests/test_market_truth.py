from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.market_truth import build_market_truth


def _odds_row(
    *,
    game_id: int,
    start_time_utc: str,
    odds_as_of_utc: str,
    side: str,
    price: int,
    snapshot_id: str,
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "start_time_utc": start_time_utc,
        "odds_as_of_utc": odds_as_of_utc,
        "odds_snapshot_id": snapshot_id,
        "market_key": "h2h",
        "outcome_side": side,
        "outcome_price": price,
        "bookmaker_key": "book-a",
    }


def test_market_truth_uses_prediction_time_for_current_odds_and_game_start_for_close() -> None:
    predictions = pd.DataFrame(
        [
            {
                "game_id": 10,
                "start_time_utc": "2026-04-10T23:05:00Z",
                "as_of_utc": "2026-04-10T18:00:00Z",
                "home_team": "CHC",
                "away_team": "LAD",
                "home_win": 1,
                "glm_ridge": 0.58,
            }
        ]
    )
    odds = pd.DataFrame(
        [
            _odds_row(
                game_id=10,
                start_time_utc="2026-04-10T23:05:00Z",
                odds_as_of_utc="2026-04-10T12:00:00Z",
                side="home",
                price=-110,
                snapshot_id="open",
            ),
            _odds_row(
                game_id=10,
                start_time_utc="2026-04-10T23:05:00Z",
                odds_as_of_utc="2026-04-10T12:00:00Z",
                side="away",
                price=100,
                snapshot_id="open",
            ),
            _odds_row(
                game_id=10,
                start_time_utc="2026-04-10T23:05:00Z",
                odds_as_of_utc="2026-04-10T19:30:00Z",
                side="home",
                price=-145,
                snapshot_id="after-prediction",
            ),
            _odds_row(
                game_id=10,
                start_time_utc="2026-04-10T23:05:00Z",
                odds_as_of_utc="2026-04-10T19:30:00Z",
                side="away",
                price=125,
                snapshot_id="after-prediction",
            ),
            _odds_row(
                game_id=10,
                start_time_utc="2026-04-10T23:05:00Z",
                odds_as_of_utc="2026-04-10T22:59:00Z",
                side="home",
                price=-160,
                snapshot_id="close",
            ),
            _odds_row(
                game_id=10,
                start_time_utc="2026-04-10T23:05:00Z",
                odds_as_of_utc="2026-04-10T22:59:00Z",
                side="away",
                price=140,
                snapshot_id="close",
            ),
            _odds_row(
                game_id=10,
                start_time_utc="2026-04-10T23:05:00Z",
                odds_as_of_utc="2026-04-10T23:08:00Z",
                side="home",
                price=-200,
                snapshot_id="post-start",
            ),
            _odds_row(
                game_id=10,
                start_time_utc="2026-04-10T23:05:00Z",
                odds_as_of_utc="2026-04-10T23:08:00Z",
                side="away",
                price=180,
                snapshot_id="post-start",
            ),
        ]
    )

    result = build_market_truth(predictions, odds, model_names=["glm_ridge"])
    row = result.per_game.iloc[0]

    assert row["opening_odds_as_of_utc"] == "2026-04-10T12:00:00Z"
    assert row["current_odds_as_of_utc"] == "2026-04-10T12:00:00Z"
    assert row["closing_odds_as_of_utc"] == "2026-04-10T22:59:00Z"
    assert row["current_home_moneyline"] == -110
    assert row["closing_home_moneyline"] == -160
    assert row["timestamp_integrity"] == "ok"


def test_market_truth_scores_edge_clv_roi_and_market_baselines() -> None:
    predictions = pd.DataFrame(
        [
            {
                "game_id": 1,
                "start_time_utc": "2026-04-11T23:05:00Z",
                "as_of_utc": "2026-04-11T16:00:00Z",
                "home_team": "CHC",
                "away_team": "LAD",
                "home_win": 1,
                "glm_ridge": 0.60,
            },
            {
                "game_id": 2,
                "start_time_utc": "2026-04-12T23:05:00Z",
                "as_of_utc": "2026-04-12T16:00:00Z",
                "home_team": "NYM",
                "away_team": "ATL",
                "home_win": 0,
                "glm_ridge": 0.42,
            },
        ]
    )
    odds = pd.DataFrame(
        [
            _odds_row(game_id=1, start_time_utc="2026-04-11T23:05:00Z", odds_as_of_utc="2026-04-11T16:00:00Z", side="home", price=-110, snapshot_id="g1-current"),
            _odds_row(game_id=1, start_time_utc="2026-04-11T23:05:00Z", odds_as_of_utc="2026-04-11T16:00:00Z", side="away", price=100, snapshot_id="g1-current"),
            _odds_row(game_id=1, start_time_utc="2026-04-11T23:05:00Z", odds_as_of_utc="2026-04-11T22:55:00Z", side="home", price=-150, snapshot_id="g1-close"),
            _odds_row(game_id=1, start_time_utc="2026-04-11T23:05:00Z", odds_as_of_utc="2026-04-11T22:55:00Z", side="away", price=135, snapshot_id="g1-close"),
            _odds_row(game_id=2, start_time_utc="2026-04-12T23:05:00Z", odds_as_of_utc="2026-04-12T16:00:00Z", side="home", price=120, snapshot_id="g2-current"),
            _odds_row(game_id=2, start_time_utc="2026-04-12T23:05:00Z", odds_as_of_utc="2026-04-12T16:00:00Z", side="away", price=-130, snapshot_id="g2-current"),
            _odds_row(game_id=2, start_time_utc="2026-04-12T23:05:00Z", odds_as_of_utc="2026-04-12T22:55:00Z", side="home", price=145, snapshot_id="g2-close"),
            _odds_row(game_id=2, start_time_utc="2026-04-12T23:05:00Z", odds_as_of_utc="2026-04-12T22:55:00Z", side="away", price=-165, snapshot_id="g2-close"),
        ]
    )

    result = build_market_truth(predictions, odds, model_names=["glm_ridge"], min_edge=0.01, calibration_bins=4)

    assert set(result.per_game["recommended_side"]) == {"home", "away"}
    assert result.per_game["flat_bet_profit_units"].sum() > 0
    assert result.per_game["clv_probability"].mean() > 0

    summary = result.summary.set_index("model_name").loc["glm_ridge"]
    assert summary["bet_count"] == 2
    assert summary["flat_roi"] > 0
    assert summary["positive_clv_rate"] == pytest.approx(1.0)
    assert summary["model_log_loss"] < summary["current_market_log_loss"]
    assert summary["model_brier"] < summary["current_market_brier"]

    assert {"model", "current_market", "closing_market"} <= set(result.calibration["probability_source"])
