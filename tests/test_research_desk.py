import json
from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.services.research_backtest import ResearchBacktestResult
from src.services.research_desk import load_structured_research_briefs, run_research_desk
from src.storage.db import Database


def test_load_structured_research_briefs_uses_brief_key_and_defaults(tmp_path):
    brief_dir = tmp_path / "briefs"
    brief_dir.mkdir(parents=True, exist_ok=True)
    (brief_dir / "default.yaml").write_text(
        """
league: NBA
brief_key: default
title: Default brief
profile_key: default
candidate_models:
  - glm_elastic_net
"""
    )

    cfg = load_config("configs/nba.yaml")
    briefs = load_structured_research_briefs(cfg, brief="default", brief_dir=str(brief_dir))

    assert len(briefs) == 1
    assert briefs[0].brief_key == "default"
    assert briefs[0].title == "Default brief"
    assert briefs[0].league == "NBA"


def test_run_research_desk_promotes_and_persists_active_champion(tmp_path, monkeypatch):
    brief_dir = tmp_path / "briefs"
    brief_dir.mkdir(parents=True, exist_ok=True)
    (brief_dir / "default.yaml").write_text(
        """
league: NBA
brief_key: default
title: Default brief
profile_key: default
candidate_models:
  - glm_elastic_net
feature_pool: production_model_map
feature_map_model: glm_elastic_net
"""
    )

    artifacts_dir = tmp_path / "artifacts"
    run_dir = artifacts_dir / "research" / "nba" / "desk_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = run_dir / "candidate_scorecard.csv"
    fold_metrics_path = run_dir / "outer_fold_metrics.csv"
    promotion_path = run_dir / "promotion_summary.json"
    report_path = run_dir / "research_backtest_report.md"

    pd.DataFrame(
        [
            {
                "model_name": "glm_elastic_net",
                "strategy": "riskAdjusted",
                "mean_ending_bankroll": 5400,
                "mean_max_drawdown": 400,
                "mean_ece": 0.05,
                "profit_winning_folds": 8,
                "bet_count": 90,
            },
            {
                "model_name": "glm_ridge",
                "strategy": "riskAdjusted",
                "mean_ending_bankroll": 5100,
                "mean_max_drawdown": 350,
                "mean_ece": 0.06,
                "profit_winning_folds": 6,
                "bet_count": 90,
            },
        ]
    ).to_csv(scorecard_path, index=False)
    pd.DataFrame([{"fold": 1}]).to_csv(fold_metrics_path, index=False)
    promotion_path.write_text(json.dumps({"eligible": True, "reason": "all_checks_passed"}))
    report_path.write_text("# report\n")

    def fake_run_research_backtest(*args, **kwargs):
        return ResearchBacktestResult(
            league="NBA",
            report_path=report_path,
            scorecard_path=scorecard_path,
            fold_metrics_path=fold_metrics_path,
            promotion_path=promotion_path,
            best_candidate_model="glm_elastic_net",
        )

    monkeypatch.setattr("src.services.research_backtest.run_research_backtest", fake_run_research_backtest)

    cfg = load_config("configs/nba.yaml")
    cfg.paths.db_path = str(tmp_path / "processed" / "nba_forecast.db")
    cfg.paths.artifacts_dir = str(artifacts_dir)

    result = run_research_desk(cfg, brief_dir=str(brief_dir))
    db = Database(cfg.paths.db_path)

    champions = db.query("SELECT league, profile_key, model_name FROM active_champions")
    assert result.promoted is True
    assert result.active_model_name == "glm_elastic_net"
    assert champions == [{"league": "NBA", "profile_key": "default", "model_name": "glm_elastic_net"}]


def test_run_research_desk_keeps_default_incumbent_when_policy_rejects(tmp_path, monkeypatch):
    brief_dir = tmp_path / "briefs"
    brief_dir.mkdir(parents=True, exist_ok=True)
    (brief_dir / "default.yaml").write_text(
        """
league: NBA
brief_key: default
title: Default brief
profile_key: default
"""
    )

    artifacts_dir = tmp_path / "artifacts"
    run_dir = artifacts_dir / "research" / "nba" / "desk_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = run_dir / "candidate_scorecard.csv"
    fold_metrics_path = run_dir / "outer_fold_metrics.csv"
    promotion_path = run_dir / "promotion_summary.json"
    report_path = run_dir / "research_backtest_report.md"

    pd.DataFrame(
        [
            {
                "model_name": "glm_ridge",
                "strategy": "riskAdjusted",
                "mean_ending_bankroll": 5001,
                "mean_max_drawdown": 1400,
                "mean_ece": 0.2,
                "profit_winning_folds": 2,
                "bet_count": 10,
            },
            {
                "model_name": "glm_elastic_net",
                "strategy": "riskAdjusted",
                "mean_ending_bankroll": 5000,
                "mean_max_drawdown": 300,
                "mean_ece": 0.06,
                "profit_winning_folds": 6,
                "bet_count": 90,
            },
        ]
    ).to_csv(scorecard_path, index=False)
    pd.DataFrame([{"fold": 1}]).to_csv(fold_metrics_path, index=False)
    promotion_path.write_text(json.dumps({"eligible": False, "reason": "policy_failed"}))
    report_path.write_text("# report\n")

    def fake_run_research_backtest(*args, **kwargs):
        return ResearchBacktestResult(
            league="NBA",
            report_path=report_path,
            scorecard_path=scorecard_path,
            fold_metrics_path=fold_metrics_path,
            promotion_path=promotion_path,
            best_candidate_model="glm_ridge",
        )

    monkeypatch.setattr("src.services.research_backtest.run_research_backtest", fake_run_research_backtest)

    cfg = load_config("configs/nba.yaml")
    cfg.paths.db_path = str(tmp_path / "processed" / "nba_forecast.db")
    cfg.paths.artifacts_dir = str(artifacts_dir)

    result = run_research_desk(cfg, brief_dir=str(brief_dir))
    db = Database(cfg.paths.db_path)

    champions = db.query("SELECT league, profile_key, model_name FROM active_champions")
    assert result.promoted is False
    assert result.active_model_name == "glm_elastic_net"
    assert champions == []
