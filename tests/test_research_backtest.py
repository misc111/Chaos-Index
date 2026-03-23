import json

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.research.candidate_models import PenalizedLogitCandidate, VanillaGLMBinomialCandidate
from src.research.model_comparison import CandidateSpec
from src.services.history_import import import_historical_data
from src.services.research_backtest import _resolve_adaptive_min_train_days, run_research_backtest


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
