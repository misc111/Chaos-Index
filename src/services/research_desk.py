from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import pandas as pd
import yaml

from src.common.config import AppConfig
from src.common.logging import get_logger
from src.research.model_comparison import CANDIDATE_MODEL_NAMES
from src.services import research_backtest as research_backtest_service
from src.storage.db import Database

logger = get_logger(__name__)

DEFAULT_PROFILE_KEY = "default"
DEFAULT_BRIEF_STATUS = "active"
DEFAULT_NBA_CHAMPION = "glm_elastic_net"
MAX_DRAWDOWN_LIMIT = 750.0
MIN_BET_COUNT = 10
MIN_PROFITABLE_FOLDS = 2
MAX_ECE_DELTA = 0.01
MATERIALIZABLE_MODEL_NAMES = {
    "ensemble",
    "glm_elastic_net",
    "glm_lasso",
    "glm_ridge",
    "dynamic_rating",
    "bayes_bt_state_space",
}


@dataclass(frozen=True)
class ResearchDeskRunResult:
    league: str
    run_id: str
    active_model_name: str
    promoted: bool
    report_path: Path
    promotion_path: Path

    @property
    def champion_promoted(self) -> bool:
        return self.promoted


@dataclass(frozen=True)
class StructuredBrief:
    brief_id: str
    league: str
    profile_key: str
    brief_key: str
    title: str
    status: str
    source_path: Path
    brief: dict[str, object]

    @property
    def policy(self) -> dict[str, object]:
        raw = self.brief.get("policy")
        base: dict[str, object] = {
            "max_mean_drawdown_dollars": 1250.0,
            "min_bet_count": MIN_BET_COUNT,
            "min_profitable_folds": MIN_PROFITABLE_FOLDS,
            "max_ece_delta": MAX_ECE_DELTA,
        }
        if isinstance(raw, dict):
            base.update(raw)
        return base

    @property
    def candidate_models(self) -> list[str]:
        return _candidate_models_from_brief(self.brief) or []


ResearchDeskBrief = StructuredBrief


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_league(league: str) -> str:
    return str(league or "").strip().upper()


def _default_brief_dir(league: str) -> Path:
    return _repo_root() / "configs" / "research_briefs" / league.lower()


def _candidate_models_from_brief(payload: dict[str, object]) -> list[str] | None:
    raw = payload.get("candidate_models")
    if raw is None:
        return None
    values = [str(item).strip() for item in raw if str(item).strip()] if isinstance(raw, list) else []
    return values or None


def _resolve_brief_file(brief: str, *, brief_dir: Path) -> Path:
    candidate = Path(brief)
    if candidate.exists():
        return candidate.resolve()
    suffixes = ("", ".yaml", ".yml")
    for suffix in suffixes:
        path = (brief_dir / f"{brief}{suffix}").resolve()
        if path.exists():
            return path
    raise FileNotFoundError(f"Unable to resolve research brief '{brief}' under {brief_dir}")


def _load_structured_brief(path: Path, *, expected_league: str) -> StructuredBrief:
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Structured brief at {path} must parse into a mapping")
    league = _normalize_league(str(payload.get("league", expected_league)))
    if league != expected_league:
        raise ValueError(f"Structured brief {path} is for league={league}, expected {expected_league}")
    brief_key = str(payload.get("brief_key") or payload.get("key") or path.stem).strip()
    if not brief_key:
        raise ValueError(f"Structured brief {path} is missing a key")
    title = str(payload.get("title") or brief_key).strip()
    if not title:
        raise ValueError(f"Structured brief {path} is missing a title")
    profile_key = str(payload.get("profile_key") or DEFAULT_PROFILE_KEY).strip() or DEFAULT_PROFILE_KEY
    status = str(payload.get("status") or DEFAULT_BRIEF_STATUS).strip() or DEFAULT_BRIEF_STATUS
    return StructuredBrief(
        brief_id=f"{league.lower()}::{profile_key}::{brief_key}",
        league=league,
        profile_key=profile_key,
        brief_key=brief_key,
        title=title,
        status=status,
        source_path=path.resolve(),
        brief=payload,
    )


def _load_briefs(*, cfg: AppConfig, brief: str | None, brief_dir: str | None) -> list[StructuredBrief]:
    league = _normalize_league(cfg.data.league)
    base_dir = Path(brief_dir).resolve() if brief_dir else _default_brief_dir(league)
    if brief:
        return [_load_structured_brief(_resolve_brief_file(brief, brief_dir=base_dir), expected_league=league)]
    if not base_dir.exists():
        return []
    files = sorted(base_dir.glob("*.y*ml"))
    briefs = [_load_structured_brief(path, expected_league=league) for path in files]
    return [entry for entry in briefs if entry.status == DEFAULT_BRIEF_STATUS]


def _load_brief(cfg: AppConfig, *, brief: str | None, brief_dir: str | None) -> StructuredBrief:
    briefs = _load_briefs(cfg=cfg, brief=brief, brief_dir=brief_dir)
    if not briefs:
        raise FileNotFoundError("No active structured research brief could be loaded")
    return briefs[0]


def load_structured_research_briefs(
    cfg: AppConfig, *, brief: str | None = None, brief_dir: str | None = None
) -> list[StructuredBrief]:
    return _load_briefs(cfg=cfg, brief=brief, brief_dir=brief_dir)


def _choose_incumbent_model(db: Database, *, league: str, profile_key: str) -> str:
    rows = db.query(
        """
        SELECT model_name
        FROM active_champions
        WHERE league = ? AND profile_key = ?
        """,
        (league, profile_key),
    )
    if rows and rows[0].get("model_name"):
        return str(rows[0]["model_name"])
    return DEFAULT_NBA_CHAMPION if league == "NBA" else "ensemble"


def _has_active_champion(db: Database, *, league: str, profile_key: str) -> bool:
    rows = db.query(
        """
        SELECT 1
        FROM active_champions
        WHERE league = ? AND profile_key = ?
        LIMIT 1
        """,
        (league, profile_key),
    )
    return bool(rows)


def _materializable_candidate(model_name: str) -> bool:
    normalized = str(model_name).strip()
    return normalized in CANDIDATE_MODEL_NAMES and normalized in MATERIALIZABLE_MODEL_NAMES


def _evaluate_promotion(
    *,
    promotion: dict[str, object],
    candidate_model_name: str,
    brief: StructuredBrief | None,
    bootstrap_mode: bool = False,
) -> dict[str, object]:
    best_row = promotion.get("best_candidate_row")
    baseline_row = promotion.get("baseline_row")
    if not isinstance(best_row, dict):
        return {
            "eligible": False,
            "promoted": False,
            "reason_summary": "Missing best candidate row",
            "gates": {"best_candidate_row_present": False},
        }
    if not isinstance(baseline_row, dict):
        return {
            "eligible": False,
            "promoted": False,
            "reason_summary": "Missing baseline row",
            "gates": {"baseline_row_present": False},
        }

    policy = brief.policy if brief else {}
    max_drawdown_limit = float(policy.get("max_mean_drawdown_dollars") or MAX_DRAWDOWN_LIMIT)
    min_bet_count = int(policy.get("min_bet_count") or MIN_BET_COUNT)
    min_profitable_folds = int(policy.get("min_profitable_folds") or MIN_PROFITABLE_FOLDS)
    max_ece_delta = float(policy.get("max_ece_delta") or MAX_ECE_DELTA)

    source_checks = promotion.get("checks") if isinstance(promotion.get("checks"), dict) else {}
    profitable_folds = int(best_row.get("profitable_folds") or best_row.get("profit_winning_folds") or 0)
    if profitable_folds <= 0 and bool(source_checks.get("outer_fold_profit_wins")):
        profitable_folds = min_profitable_folds

    gates = {
        "research_backtest_eligible": bool(promotion.get("eligible")),
        "materializable_candidate": _materializable_candidate(candidate_model_name),
        "beats_incumbent_bankroll": bootstrap_mode
        or float(best_row.get("mean_ending_bankroll") or 0.0) > float(baseline_row.get("mean_ending_bankroll") or 0.0),
        "max_drawdown_limit": float(best_row.get("mean_max_drawdown") or 0.0) <= max_drawdown_limit,
        "calibration_guardrail": float(best_row.get("mean_ece") or 0.0)
        <= float(baseline_row.get("mean_ece") or 0.0) + max_ece_delta,
        "minimum_bet_count": int(best_row.get("bet_count") or 0) >= min_bet_count,
        "minimum_profitable_folds": profitable_folds >= min_profitable_folds,
    }
    promoted = bool(all(gates.values()))
    reasons = [name for name, passed in gates.items() if not passed]
    return {
        "eligible": promoted,
        "promoted": promoted,
        "reason_summary": "Auto-promoted" if promoted else f"Rejected: {', '.join(reasons)}",
        "gates": gates,
        "source_checks": source_checks,
        "best_candidate_row": best_row,
        "baseline_row": baseline_row,
        "policy": {
            "max_mean_drawdown_dollars": max_drawdown_limit,
            "min_bet_count": min_bet_count,
            "min_profitable_folds": min_profitable_folds,
            "max_ece_delta": max_ece_delta,
        },
    }


def _hydrate_promotion_rows(
    promotion: dict[str, object],
    *,
    result: research_backtest_service.ResearchBacktestResult,
    incumbent_model_name: str,
    bootstrap_mode: bool,
) -> dict[str, object]:
    if isinstance(promotion.get("best_candidate_row"), dict) and isinstance(promotion.get("baseline_row"), dict):
        return promotion
    scorecard = pd.read_csv(result.scorecard_path)
    if scorecard.empty:
        return promotion
    best_model = str(promotion.get("best_model") or result.best_candidate_model or "").strip()
    baseline_model = str(promotion.get("baseline_model") or incumbent_model_name or "").strip()
    strategy = str(promotion.get("strategy") or "default").strip() or "default"

    def _select_row(model_name: str) -> dict[str, object] | None:
        if not model_name:
            return None
        strategy_rows = scorecard[(scorecard["model_name"] == model_name) & (scorecard["strategy"] == strategy)]
        if not strategy_rows.empty:
            return strategy_rows.iloc[0].to_dict()
        model_rows = scorecard[scorecard["model_name"] == model_name]
        if not model_rows.empty:
            return model_rows.iloc[0].to_dict()
        return None

    if best_model:
        best_row = _select_row(best_model)
        if best_row:
            promotion["best_candidate_row"] = best_row
    if baseline_model:
        baseline_row = _select_row(baseline_model)
        if baseline_row:
            promotion["baseline_row"] = baseline_row
    if not isinstance(promotion.get("baseline_row"), dict):
        candidate_rows = scorecard[scorecard["model_name"] != best_model]
        if "strategy" in candidate_rows.columns:
            strategy_rows = candidate_rows[candidate_rows["strategy"] == strategy]
            if not strategy_rows.empty:
                candidate_rows = strategy_rows
        if not candidate_rows.empty:
            ranked = candidate_rows.sort_values(by="mean_ending_bankroll", ascending=False)
            promotion["baseline_row"] = ranked.iloc[0].to_dict()
        elif bootstrap_mode and isinstance(promotion.get("best_candidate_row"), dict):
            promotion["baseline_row"] = dict(promotion["best_candidate_row"])
    return promotion


def _persist_brief(db: Database, brief: StructuredBrief) -> None:
    now = _utc_now()
    db.execute(
        """
        INSERT INTO experiment_briefs(
          brief_id, league, profile_key, brief_key, title, status,
          source_path, source_kind, brief_json, created_at_utc, updated_at_utc
        ) VALUES(?, ?, ?, ?, ?, ?, ?, 'structured_brief', ?, ?, ?)
        ON CONFLICT(brief_id) DO UPDATE SET
          title = excluded.title,
          status = excluded.status,
          source_path = excluded.source_path,
          brief_json = excluded.brief_json,
          updated_at_utc = excluded.updated_at_utc
        """,
        (
            brief.brief_id,
            brief.league,
            brief.profile_key,
            brief.brief_key,
            brief.title,
            brief.status,
            str(brief.source_path),
            json.dumps(brief.brief, sort_keys=True),
            now,
            now,
        ),
    )


def _persist_run(
    db: Database,
    *,
    run_id: str,
    league: str,
    profile_key: str,
    brief: StructuredBrief | None,
    incumbent_model_name: str,
    candidate_model_name: str,
    report_slug: str | None,
    result: research_backtest_service.ResearchBacktestResult,
    promotion_payload: dict[str, object],
) -> None:
    started_at = promotion_payload.get("started_at_utc") or _utc_now()
    completed_at = promotion_payload.get("completed_at_utc") or _utc_now()
    db.execute(
        """
        INSERT OR REPLACE INTO experiment_runs(
          run_id, league, profile_key, brief_id, brief_key, incumbent_model_name, candidate_model_name,
          report_slug, report_path, scorecard_path, fold_metrics_path, promotion_path, status,
          auto_promote, started_at_utc, completed_at_utc, summary_json
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
        """,
        (
            run_id,
            league,
            profile_key,
            brief.brief_id if brief else None,
            brief.brief_key if brief else None,
            incumbent_model_name,
            candidate_model_name,
            report_slug,
            str(result.report_path),
            str(result.scorecard_path),
            str(result.fold_metrics_path),
            str(result.promotion_path),
            "completed",
            str(started_at),
            str(completed_at),
            json.dumps(promotion_payload, sort_keys=True),
        ),
    )


def _persist_decision(
    db: Database,
    *,
    run_id: str,
    league: str,
    profile_key: str,
    incumbent_model_name: str,
    candidate_model_name: str,
    decision: dict[str, object],
) -> None:
    db.execute(
        """
        INSERT OR REPLACE INTO promotion_decisions(
          run_id, league, profile_key, incumbent_model_name, candidate_model_name,
          promoted, reason_summary, policy_json, created_at_utc
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            league,
            profile_key,
            incumbent_model_name,
            candidate_model_name,
            1 if decision.get("promoted") else 0,
            str(decision.get("reason_summary") or ""),
            json.dumps(decision, sort_keys=True),
            _utc_now(),
        ),
    )


def _persist_active_champion(
    db: Database,
    *,
    league: str,
    profile_key: str,
    model_name: str,
    run_id: str,
    brief: StructuredBrief | None,
    decision: dict[str, object],
) -> None:
    now = _utc_now()
    policy_payload = decision.get("policy") if isinstance(decision.get("policy"), dict) else {}
    if not isinstance(policy_payload, dict):
        policy_payload = {}
    policy_payload = {
        **policy_payload,
        "gates": decision.get("gates"),
        "reason_summary": decision.get("reason_summary"),
        "candidate_model_name": decision.get("candidate_model_name"),
        "incumbent_model_name": decision.get("incumbent_model_name"),
    }
    descriptor = {
        "model_name": model_name,
        "profile_key": profile_key,
        "league": league,
        "brief_key": brief.brief_key if brief else None,
        "candidate_count": len(brief.candidate_models) if brief else 0,
    }
    db.execute(
        """
        INSERT INTO active_champions(
          league, profile_key, model_name, source_run_id, source_brief_id,
          promoted_at_utc, descriptor_json, policy_json, created_at_utc, updated_at_utc
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(league, profile_key) DO UPDATE SET
          model_name = excluded.model_name,
          source_run_id = excluded.source_run_id,
          source_brief_id = excluded.source_brief_id,
          promoted_at_utc = excluded.promoted_at_utc,
          descriptor_json = excluded.descriptor_json,
          policy_json = excluded.policy_json,
          updated_at_utc = excluded.updated_at_utc
        """,
        (
            league,
            profile_key,
            model_name,
            run_id,
            brief.brief_id if brief else None,
            now,
            json.dumps(descriptor, sort_keys=True),
            json.dumps(policy_payload, sort_keys=True),
            now,
            now,
        ),
    )


def _brief_override(brief: StructuredBrief | None, key: str, fallback: object) -> object:
    if brief is None:
        return fallback
    return brief.brief.get(key, fallback)


def _resolved_candidate_models(cli_models: list[str] | None, brief: StructuredBrief | None) -> list[str] | None:
    if cli_models:
        return cli_models
    return _candidate_models_from_brief(brief.brief) if brief else None


def run_research_desk(
    cfg: AppConfig,
    *,
    report_slug: str | None = None,
    brief: str | None = None,
    brief_dir: str | None = None,
    candidate_models: list[str] | None = None,
    feature_pool: str = "research_broad",
    feature_map_model: str = "glm_ridge",
    history_seasons: int | None = None,
    structured_glm_spec_path: str | None = None,
    structured_glm_slate: str | None = None,
    structured_glm_width_variant: str | None = None,
) -> ResearchDeskRunResult:
    league = _normalize_league(cfg.data.league)
    if league != "NBA":
        raise ValueError("research_desk is NBA-only in v1")

    db = Database(cfg.paths.db_path)
    db.init_schema()

    briefs = _load_briefs(cfg=cfg, brief=brief, brief_dir=brief_dir)
    selected_brief = briefs[0] if briefs else None
    if selected_brief:
        _persist_brief(db, selected_brief)

    profile_key = selected_brief.profile_key if selected_brief else DEFAULT_PROFILE_KEY
    has_active_champion = _has_active_champion(db, league=league, profile_key=profile_key)
    incumbent_model_name = _choose_incumbent_model(db, league=league, profile_key=profile_key)

    resolved_feature_pool = str(_brief_override(selected_brief, "feature_pool", feature_pool or cfg.research.feature_pool))
    resolved_feature_map_model = str(_brief_override(selected_brief, "feature_map_model", feature_map_model))
    resolved_history_seasons = int(_brief_override(selected_brief, "history_seasons", history_seasons or cfg.research.history_seasons))
    resolved_structured_glm_spec_path = _brief_override(selected_brief, "structured_glm_spec_path", structured_glm_spec_path)
    resolved_structured_glm_slate = _brief_override(selected_brief, "structured_glm_slate", structured_glm_slate)
    resolved_structured_glm_width_variant = _brief_override(selected_brief, "structured_glm_width_variant", structured_glm_width_variant)
    resolved_candidate_models = _resolved_candidate_models(candidate_models, selected_brief)

    report_slug_token = report_slug or f"{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}_research_desk"
    run_id = f"{league.lower()}-desk-{uuid.uuid4().hex[:12]}"
    started_at_utc = _utc_now()

    result = research_backtest_service.run_research_backtest(
        cfg,
        report_slug=report_slug_token,
        candidate_models=resolved_candidate_models,
        feature_pool=resolved_feature_pool,
        feature_map_model=resolved_feature_map_model,
        history_seasons=resolved_history_seasons,
        structured_glm_spec_path=resolved_structured_glm_spec_path,
        structured_glm_slate=resolved_structured_glm_slate,
        structured_glm_width_variant=resolved_structured_glm_width_variant,
    )

    promotion = json.loads(result.promotion_path.read_text())
    promotion = _hydrate_promotion_rows(
        promotion,
        result=result,
        incumbent_model_name=incumbent_model_name,
        bootstrap_mode=not has_active_champion,
    )
    decision = _evaluate_promotion(
        promotion=promotion,
        candidate_model_name=str(result.best_candidate_model),
        brief=selected_brief,
        bootstrap_mode=not has_active_champion,
    )
    decision["started_at_utc"] = started_at_utc
    decision["completed_at_utc"] = _utc_now()
    decision["brief_key"] = selected_brief.brief_key if selected_brief else None
    decision["incumbent_model_name"] = incumbent_model_name
    decision["candidate_model_name"] = str(result.best_candidate_model)
    decision["league"] = league
    decision["profile_key"] = profile_key

    _persist_run(
        db,
        run_id=run_id,
        league=league,
        profile_key=profile_key,
        brief=selected_brief,
        incumbent_model_name=incumbent_model_name,
        candidate_model_name=str(result.best_candidate_model),
        report_slug=report_slug_token,
        result=result,
        promotion_payload=decision,
    )
    _persist_decision(
        db,
        run_id=run_id,
        league=league,
        profile_key=profile_key,
        incumbent_model_name=incumbent_model_name,
        candidate_model_name=str(result.best_candidate_model),
        decision=decision,
    )

    active_model_name = incumbent_model_name
    if decision.get("promoted"):
        active_model_name = str(result.best_candidate_model)
        _persist_active_champion(
            db,
            league=league,
            profile_key=profile_key,
            model_name=active_model_name,
            run_id=run_id,
            brief=selected_brief,
            decision=decision,
        )

    logger.info(
        "Research desk complete | league=%s run_id=%s incumbent=%s candidate=%s promoted=%s active=%s",
        league,
        run_id,
        incumbent_model_name,
        result.best_candidate_model,
        bool(decision.get("promoted")),
        active_model_name,
    )
    return ResearchDeskRunResult(
        league=league,
        run_id=run_id,
        active_model_name=active_model_name,
        promoted=bool(decision.get("promoted")),
        report_path=result.report_path,
        promotion_path=result.promotion_path,
    )
