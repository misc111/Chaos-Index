import json

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.research.candidate_models import PenalizedLogitCandidate, VanillaGLMBinomialCandidate
from src.research.model_comparison import CandidateSpec
from src.services.history_import import import_historical_data
from src.services.research_backtest import (
    _choose_best_candidate,
    _latest_pregame_moneylines,
    _promotion_summary,
    _resolve_adaptive_min_train_days,
    run_research_backtest,
)
from src.storage.db import Database
from src.storage.schema import EFFECTIVE_ODDS_MARKET_LINES_VIEW_NAME


def test_resolve_adaptive_min_train_days_caps_to_available_history():
    frame = pd.DataFrame(
        {
            "home_win": [0, 1] * 70,
            "start_time_utc": pd.date_range("2025-10-01", periods=140, freq="D").strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )

    resolved = _resolve_adaptive_min_train_days(
        frame,
        validation_days=60,
        embargo_days=1,
        requested_min_train_days=180,
    )

    assert resolved == 78


def test_resolve_adaptive_min_train_days_returns_none_when_history_is_too_short():
    frame = pd.DataFrame(
        {
            "home_win": [0, 1] * 40,
            "start_time_utc": pd.date_range("2025-10-01", periods=80, freq="D").strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )

    resolved = _resolve_adaptive_min_train_days(
        frame,
        validation_days=60,
        embargo_days=1,
        requested_min_train_days=180,
    )

    assert resolved is None


def test_latest_pregame_moneylines_accepts_historical_bundle_rows_with_late_snapshot_time(tmp_path):
    source_dir = tmp_path / "historical" / "nba"
    source_dir.mkdir(parents=True, exist_ok=True)

    games = pd.DataFrame(
        [
            {
                "game_id": 401810863,
                "season": 20252026,
                "game_date_utc": "2026-03-19",
                "start_time_utc": "2026-03-19T23:00:00Z",
                "home_team": "CHA",
                "away_team": "ORL",
                "home_score": 101,
                "away_score": 112,
                "home_win": 0,
                "status_final": 1,
                "as_of_utc": "2026-03-20T05:00:00Z",
            }
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "odds_event_id": 401810863,
                "commence_time_utc": "2026-03-19T23:00:00Z",
                "home_team": "CHA",
                "away_team": "ORL",
                "bookmaker_key": "book-a",
                "bookmaker_title": "Book A",
                "market_key": "h2h",
                "outcome_name": "CHA",
                "outcome_side": "home",
                "outcome_price": 455,
            },
            {
                "odds_event_id": 401810863,
                "commence_time_utc": "2026-03-19T23:00:00Z",
                "home_team": "CHA",
                "away_team": "ORL",
                "bookmaker_key": "book-a",
                "bookmaker_title": "Book A",
                "market_key": "h2h",
                "outcome_name": "ORL",
                "outcome_side": "away",
                "outcome_price": -625,
            },
        ]
    )

    games.to_csv(source_dir / "games.csv", index=False)
    odds.to_csv(source_dir / "odds.csv", index=False)
    (source_dir / "manifest.json").write_text(
        json.dumps(
            {
                "league": "NBA",
                "games": [
                    {
                        "path": "games.csv",
                        "source": "nba_historical_games",
                        "extracted_at_utc": "2026-03-24T14:05:01Z",
                    }
                ],
                "odds_snapshots": [
                    {
                        "path": "odds.csv",
                        "source": "nba_historical_odds",
                        "as_of_utc": "2026-03-24T14:05:01Z",
                        "metadata": {"import_mode": "historical_bundle"},
                    }
                ],
            }
        )
    )

    cfg = load_config("configs/nba.yaml")
    cfg.paths.db_path = str(tmp_path / "processed" / "nba_forecast.db")
    cfg.paths.interim_dir = str(tmp_path / "interim" / "nba")
    cfg.paths.processed_dir = str(tmp_path / "processed" / "nba")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.research.source_dir = str(source_dir.parent)

    import_historical_data(cfg, history_seasons=1)

    db = Database(cfg.paths.db_path)
    rows = _latest_pregame_moneylines(db, league="NBA", seasons=[20252026])
    assert not rows.empty
    record = rows.iloc[0].to_dict()
    assert record["game_id"] == 401810863
    assert record["home_moneyline"] == 455
    assert record["away_moneyline"] == -625
    assert str(pd.Timestamp(record["odds_as_of_utc"])) == "2026-03-19 23:00:00+00:00"

    effective_rows = db.query(
        f"""
        SELECT effective_odds_as_of_utc, commence_time_utc
        FROM {EFFECTIVE_ODDS_MARKET_LINES_VIEW_NAME}
        """
    )
    assert len(effective_rows) == 2
    assert all(row["effective_odds_as_of_utc"] == row["commence_time_utc"] for row in effective_rows)


def test_research_backtest_writes_dual_scorecard_bundle(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    n = 260
    dates = pd.date_range("2025-10-01", periods=n, freq="D")
    teams = np.array(["ATL", "BOS", "CHI", "DAL", "DEN", "GS", "LAL", "MIA"])
    signal = rng.normal(0.0, 1.0, n)
    home_score = 108 + np.round(8 * signal).astype(int)
    away_score = 104 + np.round(-6 * signal + rng.normal(0.0, 2.0, n)).astype(int)
    home_win = (home_score > away_score).astype(int)

    games = pd.DataFrame(
        {
            "game_id": np.arange(1, n + 1),
            "season": [20252026] * n,
            "game_date_utc": dates.date.astype(str),
            "start_time_utc": (dates + pd.Timedelta(hours=23)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "home_team": teams[np.arange(n) % len(teams)],
            "away_team": teams[(np.arange(n) + 2) % len(teams)],
            "home_score": home_score,
            "away_score": away_score,
            "home_win": home_win,
            "status_final": 1,
            "as_of_utc": (dates + pd.Timedelta(hours=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    odds_rows = []
    for row in games.itertuples(index=False):
        centered = float(signal[row.game_id - 1])
        home_price = int(-120 - round(centered * 18))
        away_price = int(105 + round(centered * 18))
        odds_rows.extend(
            [
                {
                    "odds_event_id": f"evt-{row.game_id}",
                    "commence_time_utc": row.start_time_utc,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "bookmaker_key": "book-a",
                    "bookmaker_title": "Book A",
                    "market_key": "h2h",
                    "outcome_name": row.home_team,
                    "outcome_side": "home",
                    "outcome_price": home_price,
                },
                {
                    "odds_event_id": f"evt-{row.game_id}",
                    "commence_time_utc": row.start_time_utc,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "bookmaker_key": "book-a",
                    "bookmaker_title": "Book A",
                    "market_key": "h2h",
                    "outcome_name": row.away_team,
                    "outcome_side": "away",
                    "outcome_price": away_price,
                },
            ]
        )
    odds = pd.DataFrame(odds_rows)

    source_dir = tmp_path / "historical" / "nba"
    source_dir.mkdir(parents=True, exist_ok=True)
    games.to_csv(source_dir / "games.csv", index=False)
    odds.to_csv(source_dir / "odds.csv", index=False)
    (source_dir / "manifest.json").write_text(
        json.dumps(
            {
                "league": "NBA",
                "games": [
                    {
                        "path": "games.csv",
                        "source": "nba_historical_games",
                        "extracted_at_utc": "2025-09-30T00:00:00Z",
                    }
                ],
                "odds_snapshots": [
                    {
                        "path": "odds.csv",
                        "source": "nba_historical_odds",
                        "as_of_utc": "2025-09-30T00:00:00Z",
                    }
                ],
            }
        )
    )

    cfg = load_config("configs/nba.yaml")
    cfg.paths.db_path = str(tmp_path / "processed" / "nba_forecast.db")
    cfg.paths.interim_dir = str(tmp_path / "interim" / "nba")
    cfg.paths.processed_dir = str(tmp_path / "processed" / "nba")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.research.source_dir = str(source_dir.parent)
    cfg.research.history_seasons = 1
    cfg.research.outer_folds = 3
    cfg.research.outer_valid_days = 20
    cfg.research.inner_folds = 2
    cfg.research.inner_valid_days = 15
    cfg.research.final_holdout_days = 15

    import_historical_data(cfg, history_seasons=1)

    def small_specs(feature_sets, *, selected_models=None):
        specs = [
            CandidateSpec(
                model_name="glm_ridge",
                display_name="GLM Ridge",
                param_grid=[{"c": 0.5}],
                builder=lambda fs, params: PenalizedLogitCandidate(
                    model_name="glm_ridge",
                    display_name="GLM Ridge",
                    features=fs.screened_features,
                    penalty="l2",
                    c=float(params["c"]),
                    solver="lbfgs",
                ),
            ),
            CandidateSpec(
                model_name="glm_vanilla",
                display_name="Vanilla GLM",
                param_grid=[{}],
                builder=lambda fs, params: VanillaGLMBinomialCandidate(features=fs.screened_features),
            ),
        ]
        if not selected_models:
            return specs
        return [spec for spec in specs if spec.model_name in selected_models]

    monkeypatch.setattr("src.services.research_backtest._candidate_specs", small_specs)

    result = run_research_backtest(
        cfg,
        report_slug="unit_research_backtest",
        candidate_models=["glm_ridge", "glm_vanilla"],
        feature_pool="research_broad",
        feature_map_model="glm_ridge",
        history_seasons=1,
    )

    assert result.report_path.exists()
    assert result.scorecard_path.exists()
    assert result.fold_metrics_path.exists()
    assert result.promotion_path.exists()
    scorecard = pd.read_csv(result.scorecard_path)
    assert {"glm_ridge", "glm_vanilla"} <= set(scorecard["model_name"].tolist())


def test_choose_best_candidate_prefers_profit_over_better_log_loss():
    scorecard = pd.DataFrame(
        [
            {
                "model_name": "glm_ridge",
                "strategy": "riskAdjusted",
                "mean_ending_bankroll": 5120.0,
                "mean_net_profit": 120.0,
                "mean_roi": 0.021,
                "profitable_folds": 2,
                "mean_max_drawdown": 300.0,
                "mean_log_loss": 0.59,
                "mean_brier": 0.20,
            },
            {
                "model_name": "gam_spline",
                "strategy": "riskAdjusted",
                "mean_ending_bankroll": 5400.0,
                "mean_net_profit": 400.0,
                "mean_roi": 0.031,
                "profitable_folds": 3,
                "mean_max_drawdown": 325.0,
                "mean_log_loss": 0.64,
                "mean_brier": 0.22,
            },
        ]
    )

    assert _choose_best_candidate(scorecard, baseline_model="glm_ridge") == "gam_spline"


def test_promotion_summary_requires_profit_and_drawdown_guardrail():
    scorecard = pd.DataFrame(
        [
            {
                "model_name": "gam_spline",
                "strategy": "riskAdjusted",
                "mean_ending_bankroll": 5450.0,
                "mean_net_profit": 450.0,
                "median_roi": 0.033,
                "profitable_folds": 3,
                "profit_winning_folds": 3,
                "mean_max_drawdown": 500.0,
                "mean_log_loss": 0.64,
                "mean_brier": 0.22,
                "mean_ece": 0.05,
                "all_integrity_checks": True,
            },
            {
                "model_name": "glm_ridge",
                "strategy": "riskAdjusted",
                "mean_ending_bankroll": 5120.0,
                "mean_net_profit": 120.0,
                "median_roi": 0.021,
                "profitable_folds": 2,
                "profit_winning_folds": 2,
                "mean_max_drawdown": 300.0,
                "mean_log_loss": 0.59,
                "mean_brier": 0.20,
                "mean_ece": 0.045,
                "all_integrity_checks": True,
            },
        ]
    )

    summary = _promotion_summary(scorecard, best_model="gam_spline", baseline_model="glm_ridge")

    assert summary["checks"]["mean_ending_bankroll"] is True
    assert summary["checks"]["mean_net_profit"] is True
    assert summary["checks"]["drawdown_guardrail"] is True


def test_research_backtest_supports_structured_glm_research_spec(tmp_path, monkeypatch):
    rng = np.random.default_rng(17)
    n = 240
    dates = pd.date_range("2025-10-01", periods=n, freq="D")
    teams = np.array(["ATL", "BOS", "CHI", "DAL", "DEN", "GS", "LAL", "MIA"])
    signal = rng.normal(0.0, 1.0, n)
    home_score = 109 + np.round(7 * signal).astype(int)
    away_score = 104 + np.round(-5 * signal + rng.normal(0.0, 2.0, n)).astype(int)
    home_win = (home_score > away_score).astype(int)

    games = pd.DataFrame(
        {
            "game_id": np.arange(1, n + 1),
            "season": [20252026] * n,
            "game_date_utc": dates.date.astype(str),
            "start_time_utc": (dates + pd.Timedelta(hours=23)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "home_team": teams[np.arange(n) % len(teams)],
            "away_team": teams[(np.arange(n) + 2) % len(teams)],
            "home_score": home_score,
            "away_score": away_score,
            "home_win": home_win,
            "status_final": 1,
            "as_of_utc": (dates + pd.Timedelta(hours=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    odds_rows = []
    for row in games.itertuples(index=False):
        centered = float(signal[row.game_id - 1])
        home_price = int(-118 - round(centered * 16))
        away_price = int(102 + round(centered * 16))
        odds_rows.extend(
            [
                {
                    "odds_event_id": f"evt-{row.game_id}",
                    "commence_time_utc": row.start_time_utc,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "bookmaker_key": "book-a",
                    "bookmaker_title": "Book A",
                    "market_key": "h2h",
                    "outcome_name": row.home_team,
                    "outcome_side": "home",
                    "outcome_price": home_price,
                },
                {
                    "odds_event_id": f"evt-{row.game_id}",
                    "commence_time_utc": row.start_time_utc,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "bookmaker_key": "book-a",
                    "bookmaker_title": "Book A",
                    "market_key": "h2h",
                    "outcome_name": row.away_team,
                    "outcome_side": "away",
                    "outcome_price": away_price,
                },
            ]
        )
    odds = pd.DataFrame(odds_rows)

    source_dir = tmp_path / "historical" / "nba"
    source_dir.mkdir(parents=True, exist_ok=True)
    games.to_csv(source_dir / "games.csv", index=False)
    odds.to_csv(source_dir / "odds.csv", index=False)
    (source_dir / "manifest.json").write_text(
        json.dumps(
            {
                "league": "NBA",
                "games": [
                    {
                        "path": "games.csv",
                        "source": "nba_historical_games",
                        "extracted_at_utc": "2025-09-30T00:00:00Z",
                    }
                ],
                "odds_snapshots": [
                    {
                        "path": "odds.csv",
                        "source": "nba_historical_odds",
                        "as_of_utc": "2025-09-30T00:00:00Z",
                    }
                ],
            }
        )
    )

    cfg = load_config("configs/nba.yaml")
    cfg.paths.db_path = str(tmp_path / "processed" / "nba_forecast.db")
    cfg.paths.interim_dir = str(tmp_path / "interim" / "nba")
    cfg.paths.processed_dir = str(tmp_path / "processed" / "nba")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.research.source_dir = str(source_dir.parent)
    cfg.research.history_seasons = 1
    cfg.research.outer_folds = 2
    cfg.research.outer_valid_days = 16
    cfg.research.inner_folds = 2
    cfg.research.inner_valid_days = 12
    cfg.research.final_holdout_days = 12

    import_historical_data(cfg, history_seasons=1)

    def small_specs(feature_sets, *, selected_models=None, glm_feature_overrides=None):
        def _glm_features(fs, model_name: str) -> list[str]:
            if not glm_feature_overrides:
                return fs.screened_features
            requested = [str(feature) for feature in glm_feature_overrides.get(model_name, []) if str(feature).strip()]
            if not requested:
                return fs.screened_features
            screened = set(fs.screened_features)
            resolved = [feature for feature in requested if feature in screened]
            return resolved or fs.screened_features

        specs = [
            CandidateSpec(
                model_name="glm_ridge",
                display_name="GLM Ridge",
                param_grid=[{"c": 0.5}],
                builder=lambda fs, params: PenalizedLogitCandidate(
                    model_name="glm_ridge",
                    display_name="GLM Ridge",
                    features=_glm_features(fs, "glm_ridge"),
                    penalty="l2",
                    c=float(params["c"]),
                    solver="lbfgs",
                ),
            ),
            CandidateSpec(
                model_name="glm_vanilla",
                display_name="Vanilla GLM",
                param_grid=[{}],
                builder=lambda fs, params: VanillaGLMBinomialCandidate(features=_glm_features(fs, "glm_vanilla")),
            ),
        ]
        if not selected_models:
            return specs
        return [spec for spec in specs if spec.model_name in selected_models]

    monkeypatch.setattr("src.services.research_backtest._candidate_specs", small_specs)

    spec_path = tmp_path / "nba_structured_glm.yaml"
    spec_path.write_text(
        """
version: 1
league: NBA
experiment_name: unit_structured_glm_backtest
default_slate: baseline
default_width_variant: medium
slates:
  baseline:
    feature_order:
      - elo_home_prob
      - dyn_home_prob
      - rest_diff
      - travel_diff
    width_variants:
      medium:
        feature_count: 2
"""
    )

    result = run_research_backtest(
        cfg,
        report_slug="unit_research_backtest_structured_glm",
        candidate_models=["glm_ridge", "glm_vanilla"],
        feature_pool="research_broad",
        feature_map_model="glm_ridge",
        history_seasons=1,
        structured_glm_spec_path=str(spec_path),
    )

    assert result.report_path.exists()
    assert result.scorecard_path.exists()
    scorecard = pd.read_csv(result.scorecard_path)
    assert {"glm_ridge", "glm_vanilla"} <= set(scorecard["model_name"].tolist())
    assert "structured GLM experiment `unit_structured_glm_backtest`" in result.report_path.read_text()
