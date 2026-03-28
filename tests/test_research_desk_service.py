from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.common.config import load_config
from src.services.research_backtest import ResearchBacktestResult
from src.services.research_desk import ResearchDeskBrief, _load_brief, run_research_desk
from src.storage.db import Database


def _test_cfg(tmp_path: Path):
    cfg = load_config("configs/nba.yaml")
    cfg.paths.db_path = str(tmp_path / "processed" / "nba_forecast.db")
    cfg.paths.processed_dir = str(tmp_path / "processed")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.interim_dir = str(tmp_path / "interim")
    return cfg


def test_load_brief_validates_league_and_defaults(tmp_path: Path) -> None:
    brief_dir = tmp_path / "briefs"
    brief_dir.mkdir(parents=True)
    path = brief_dir / "default.yaml"
    path.write_text(
        "\n".join(
            [
                "brief_key: test-brief",
                "title: Test Brief",
                "league: NBA",
                "candidate_models:",
                "  - glm_elastic_net",
            ]
        )
        + "\n"
    )

    cfg = _test_cfg(tmp_path)
    brief = _load_brief(cfg, brief=None, brief_dir=str(brief_dir))

    assert isinstance(brief, ResearchDeskBrief)
    assert brief.brief_key == "test-brief"
    assert brief.profile_key == "default"
    assert brief.policy["max_mean_drawdown_dollars"] == pytest.approx(1250.0)
    assert brief.candidate_models == ["glm_elastic_net"]


def test_run_research_desk_promotes_candidate_and_persists_champion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _test_cfg(tmp_path)
    brief_dir = tmp_path / "briefs"
    brief_dir.mkdir(parents=True)
    (brief_dir / "default.yaml").write_text(
        "\n".join(
            [
                "brief_key: promo-brief",
                "title: Promo Brief",
                "league: NBA",
                "candidate_models:",
                "  - glm_elastic_net",
                "feature_map_model: glm_ridge",
            ]
        )
        + "\n"
    )

    report_dir = Path(cfg.paths.artifacts_dir) / "desk_run"
    report_dir.mkdir(parents=True)
    scorecard = pd.DataFrame(
        [
            {
                "model_name": "glm_ridge",
                "strategy": "default",
                "mean_ending_bankroll": 5100.0,
                "mean_log_loss": 0.60,
                "mean_ece": 0.035,
                "mean_max_drawdown": 900.0,
                "bet_count": 25,
            },
            {
                "model_name": "glm_elastic_net",
                "strategy": "default",
                "mean_ending_bankroll": 5600.0,
                "mean_log_loss": 0.58,
                "mean_ece": 0.03,
                "mean_max_drawdown": 950.0,
                "bet_count": 32,
            },
        ]
    )
    scorecard_path = report_dir / "candidate_scorecard.csv"
    scorecard.to_csv(scorecard_path, index=False)
    promotion_payload = {
        "eligible": True,
        "best_model": "glm_elastic_net",
        "baseline_model": "glm_ridge",
        "strategy": "default",
        "checks": {
            "mean_ending_bankroll": True,
            "median_roi": True,
            "outer_fold_profit_wins": True,
            "mean_log_loss": True,
            "mean_brier": True,
            "ece_guardrail": True,
            "integrity_checks": True,
        },
    }
    promotion_path = report_dir / "promotion_summary.json"
    promotion_path.write_text(json.dumps(promotion_payload))
    fold_metrics_path = report_dir / "outer_fold_metrics.csv"
    pd.DataFrame([{"fold": 1}]).to_csv(fold_metrics_path, index=False)
    report_path = report_dir / "research_backtest_report.md"
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

    result = run_research_desk(cfg, brief_dir=str(brief_dir))

    assert result.champion_promoted is True
    assert result.active_model_name == "glm_elastic_net"

    db = Database(cfg.paths.db_path)
    champions = db.query("SELECT * FROM active_champions")
    assert len(champions) == 1
    assert champions[0]["model_name"] == "glm_elastic_net"
    decisions = db.query("SELECT * FROM promotion_decisions")
    assert len(decisions) == 1
    assert int(decisions[0]["promoted"]) == 1


def test_run_research_desk_rejects_candidate_that_breaks_drawdown_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _test_cfg(tmp_path)
    brief_dir = tmp_path / "briefs"
    brief_dir.mkdir(parents=True)
    (brief_dir / "default.yaml").write_text(
        "\n".join(
            [
                "brief_key: reject-brief",
                "title: Reject Brief",
                "league: NBA",
                "candidate_models:",
                "  - glm_elastic_net",
                "policy:",
                "  max_mean_drawdown_dollars: 500",
                "  min_bet_count: 10",
            ]
        )
        + "\n"
    )

    db = Database(cfg.paths.db_path)
    db.init_schema()
    db.execute(
        """
        INSERT INTO active_champions(
          league, profile_key, model_name, source_run_id, source_brief_id, promoted_at_utc,
          descriptor_json, policy_json, created_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "NBA",
            "default",
            "glm_ridge",
            None,
            None,
            "2026-01-01T00:00:00+00:00",
            "{}",
            "{}",
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
        ),
    )

    report_dir = Path(cfg.paths.artifacts_dir) / "desk_run"
    report_dir.mkdir(parents=True)
    scorecard = pd.DataFrame(
        [
            {
                "model_name": "glm_ridge",
                "strategy": "default",
                "mean_ending_bankroll": 5200.0,
                "mean_log_loss": 0.60,
                "mean_ece": 0.03,
                "mean_max_drawdown": 350.0,
                "bet_count": 20,
            },
            {
                "model_name": "glm_elastic_net",
                "strategy": "default",
                "mean_ending_bankroll": 5800.0,
                "mean_log_loss": 0.58,
                "mean_ece": 0.029,
                "mean_max_drawdown": 800.0,
                "bet_count": 30,
            },
        ]
    )
    scorecard_path = report_dir / "candidate_scorecard.csv"
    scorecard.to_csv(scorecard_path, index=False)
    promotion_path = report_dir / "promotion_summary.json"
    promotion_path.write_text(
        json.dumps(
            {
                "eligible": True,
                "best_model": "glm_elastic_net",
                "baseline_model": "glm_ridge",
                "strategy": "default",
                "checks": {
                    "mean_ending_bankroll": True,
                    "median_roi": True,
                    "outer_fold_profit_wins": True,
                    "mean_log_loss": True,
                    "mean_brier": True,
                    "ece_guardrail": True,
                    "integrity_checks": True,
                },
            }
        )
    )
    fold_metrics_path = report_dir / "outer_fold_metrics.csv"
    pd.DataFrame([{"fold": 1}]).to_csv(fold_metrics_path, index=False)
    report_path = report_dir / "research_backtest_report.md"
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

    result = run_research_desk(cfg, brief_dir=str(brief_dir))

    assert result.champion_promoted is False
    assert result.active_model_name == "glm_ridge"
    champions = db.query("SELECT * FROM active_champions WHERE league = 'NBA' AND profile_key = 'default'")
    assert len(champions) == 1
    assert champions[0]["model_name"] == "glm_ridge"
    decisions = db.query("SELECT * FROM promotion_decisions")
    assert len(decisions) == 1
    assert int(decisions[0]["promoted"]) == 0
