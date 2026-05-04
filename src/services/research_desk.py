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
from src.registry.models import baseline_model_names, core_model_names, get_model_registry_entry, theory_extension_model_names
from src.research.model_comparison import CANDIDATE_MODEL_NAMES
from src.services import model_compare as model_compare_service
from src.services import research_backtest as research_backtest_service
from src.storage.db import Database
from src.training.contracts import CandidateScorecardRecord, PromotionDecisionRecord

logger = get_logger(__name__)

DEFAULT_PROFILE_KEY = "default"
DEFAULT_BRIEF_STATUS = "active"
MAX_DRAWDOWN_LIMIT = 750.0
MIN_BET_COUNT = 10
MIN_PROFITABLE_FOLDS = 2
MAX_ECE_DELTA = 0.01
MATERIALIZABLE_MODEL_NAMES = {
    "ensemble",
    *core_model_names(),
    *theory_extension_model_names(),
    *baseline_model_names(),
}
RESEARCH_DESK_TARGET_NAME = "moneyline_home_win"
RESEARCH_DESK_DISTRIBUTION = "binomial"
RESEARCH_DESK_LINK_FUNCTION = "logit"


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


def _safe_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if pd.notna(numeric) else None


def _safe_int(value: object) -> int | None:
    try:
        numeric = int(value)
    except Exception:
        return None
    return numeric


def _model_contract_metadata(model_name: str) -> dict[str, str]:
    token = str(model_name or "").strip()
    if token == "ensemble":
        return {
            "lane": "ensemble",
            "family": "ensemble",
            "display_name": "Ensemble",
            "governance_note": "Fallback ensemble retained until a candidate clears the promotion gate.",
        }
    try:
        entry = get_model_registry_entry(token)
    except KeyError:
        return {
            "lane": "unknown",
            "family": "unknown",
            "display_name": token,
            "governance_note": "",
        }
    return {
        "lane": entry.lane,
        "family": entry.family,
        "display_name": entry.display_label,
        "governance_note": entry.governance_note,
    }


def _metric_delta(candidate: dict[str, object], baseline: dict[str, object], key: str) -> float | None:
    candidate_value = _safe_float(candidate.get(key))
    baseline_value = _safe_float(baseline.get(key))
    if candidate_value is None or baseline_value is None:
        return None
    return candidate_value - baseline_value


def _promotion_failure_reasons(
    *,
    gates: dict[str, bool],
    best_row: dict[str, object],
    baseline_row: dict[str, object],
    policy: dict[str, object],
    candidate_model_name: str,
) -> list[str]:
    reasons: list[str] = []
    if not gates.get("research_backtest_eligible", True):
        reasons.append("The research backtest marked the candidate ineligible before promotion review.")
    if not gates.get("materializable_candidate", True):
        reasons.append(f"`{candidate_model_name}` is not materializable in the current CAS core promotion lane.")
    if not gates.get("theory_core_candidate", True):
        reasons.append(f"`{candidate_model_name}` is not in the core-supported theory lane, so it cannot auto-promote as champion.")
    if not gates.get("validation_contract_complete", True):
        reasons.append("A separate full validation-contract review has not been recorded yet, so champion promotion is blocked.")
    if not gates.get("beats_incumbent_bankroll", True):
        reasons.append(
            f"Mean ending bankroll {_safe_float(best_row.get('mean_ending_bankroll')) or 0.0:.1f} did not clear the incumbent at {_safe_float(baseline_row.get('mean_ending_bankroll')) or 0.0:.1f}."
        )
    if not gates.get("beats_incumbent_profit", True):
        reasons.append(
            f"Mean net profit {_safe_float(best_row.get('mean_net_profit')) or 0.0:.1f} did not clear the incumbent at {_safe_float(baseline_row.get('mean_net_profit')) or 0.0:.1f}."
        )
    if not gates.get("max_drawdown_limit", True):
        reasons.append(
            f"Mean max drawdown {_safe_float(best_row.get('mean_max_drawdown')) or 0.0:.1f} exceeded the policy cap of {float(policy.get('max_mean_drawdown_dollars') or MAX_DRAWDOWN_LIMIT):.1f}."
        )
    if not gates.get("calibration_guardrail", True):
        baseline_ece = _safe_float(baseline_row.get("mean_ece")) or 0.0
        tolerance = float(policy.get("max_ece_delta") or MAX_ECE_DELTA)
        reasons.append(
            f"Mean ECE {_safe_float(best_row.get('mean_ece')) or 0.0:.4f} exceeded the incumbent-plus-tolerance guardrail of {baseline_ece:.4f} + {tolerance:.4f}."
        )
    if not gates.get("minimum_bet_count", True):
        reasons.append(
            f"Bet count {_safe_int(best_row.get('bet_count')) or 0} fell short of the minimum {int(policy.get('min_bet_count') or MIN_BET_COUNT)}."
        )
    if not gates.get("minimum_profitable_folds", True):
        profitable = _safe_int(best_row.get("profitable_folds")) or _safe_int(best_row.get("profit_winning_folds")) or 0
        reasons.append(
            f"Profitable folds {profitable} fell short of the minimum {int(policy.get('min_profitable_folds') or MIN_PROFITABLE_FOLDS)}."
        )
    return reasons


def _scorecard_rejection_reasons(
    *,
    row: pd.Series,
    strategy: str,
    anchor_row: pd.Series | None,
    candidate_model_name: str,
    active_model_name: str,
    decision: dict[str, object],
) -> list[str]:
    model_name = str(row.get("model_name") or "")
    if model_name == active_model_name:
        return []
    if model_name == candidate_model_name and isinstance(decision.get("failed_reasons"), list):
        return [str(reason) for reason in decision["failed_reasons"] if str(reason).strip()]

    reasons: list[str] = []
    if anchor_row is not None:
        model_bankroll = _safe_float(row.get("mean_ending_bankroll"))
        anchor_bankroll = _safe_float(anchor_row.get("mean_ending_bankroll"))
        model_profit = _safe_float(row.get("mean_net_profit"))
        anchor_profit = _safe_float(anchor_row.get("mean_net_profit"))
        model_log_loss = _safe_float(row.get("mean_log_loss"))
        anchor_log_loss = _safe_float(anchor_row.get("mean_log_loss"))
        if model_bankroll is not None and anchor_bankroll is not None and model_bankroll < anchor_bankroll - 1e-12:
            reasons.append(
                f"Mean ending bankroll {model_bankroll:.1f} trailed `{active_model_name}` at {anchor_bankroll:.1f}."
            )
        if model_profit is not None and anchor_profit is not None and model_profit < anchor_profit - 1e-12:
            reasons.append(f"Mean net profit {model_profit:.1f} trailed `{active_model_name}` at {anchor_profit:.1f}.")
        if model_log_loss is not None and anchor_log_loss is not None and model_log_loss > anchor_log_loss + 1e-12:
            reasons.append(f"Mean log loss {model_log_loss:.4f} trailed `{active_model_name}` at {anchor_log_loss:.4f}.")
    integrity_value = row.get("all_integrity_checks")
    if pd.notna(integrity_value) and not bool(integrity_value):
        reasons.append("One or more pregame integrity checks did not stay green.")
    if not reasons:
        reasons.append(f"`{model_name}` was not selected for the `{strategy}` promotion review.")
    return reasons[:3]


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
    return "ensemble"


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


def _theory_core_candidate(model_name: str) -> bool:
    normalized = str(model_name).strip()
    try:
        return get_model_registry_entry(normalized).lane == "core"
    except KeyError:
        return False


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
            "status": "rejected",
            "reason_summary": "Promotion rejected because the best candidate row was missing.",
            "failed_reasons": ["The promotion payload did not include a best candidate row."],
            "gates": {"best_candidate_row_present": False},
        }
    if not isinstance(baseline_row, dict):
        return {
            "eligible": False,
            "promoted": False,
            "status": "rejected",
            "reason_summary": "Promotion rejected because the incumbent comparison row was missing.",
            "failed_reasons": ["The promotion payload did not include a baseline comparison row."],
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
    validation_contract_complete = bool(promotion.get("validation_contract_complete"))

    gates = {
        "research_backtest_eligible": bool(promotion.get("eligible")),
        "materializable_candidate": _materializable_candidate(candidate_model_name),
        "theory_core_candidate": _theory_core_candidate(candidate_model_name),
        "validation_contract_complete": validation_contract_complete,
        "beats_incumbent_bankroll": bootstrap_mode
        or float(best_row.get("mean_ending_bankroll") or 0.0) > float(baseline_row.get("mean_ending_bankroll") or 0.0),
        "beats_incumbent_profit": bootstrap_mode
        or float(best_row.get("mean_net_profit") or 0.0) > float(baseline_row.get("mean_net_profit") or 0.0),
        "max_drawdown_limit": float(best_row.get("mean_max_drawdown") or 0.0) <= max_drawdown_limit,
        "calibration_guardrail": float(best_row.get("mean_ece") or 0.0)
        <= float(baseline_row.get("mean_ece") or 0.0) + max_ece_delta,
        "minimum_bet_count": int(best_row.get("bet_count") or 0) >= min_bet_count,
        "minimum_profitable_folds": profitable_folds >= min_profitable_folds,
    }
    promoted = bool(all(gates.values()))
    baseline_model = str(baseline_row.get("model_name") or promotion.get("baseline_model") or "").strip()
    failed_reasons = _promotion_failure_reasons(
        gates=gates,
        best_row=best_row,
        baseline_row=baseline_row,
        policy={
            "max_mean_drawdown_dollars": max_drawdown_limit,
            "min_bet_count": min_bet_count,
            "min_profitable_folds": min_profitable_folds,
            "max_ece_delta": max_ece_delta,
        },
        candidate_model_name=candidate_model_name,
    )
    reason_summary = (
        f"Promoted `{candidate_model_name}` over `{baseline_model}` after it improved bankroll and profit while clearing drawdown, calibration, volume, theory-lane, and validation-contract gates."
        if promoted
        else f"Promotion rejected for `{candidate_model_name}`. {' '.join(failed_reasons)}"
    )
    return {
        "eligible": promoted,
        "promoted": promoted,
        "status": "promoted" if promoted else "rejected",
        "reason_summary": reason_summary,
        "failed_reasons": failed_reasons,
        "gates": gates,
        "source_checks": source_checks,
        "best_candidate_row": best_row,
        "baseline_row": baseline_row,
        "comparison_to_incumbent": {
            "mean_ending_bankroll_delta": _metric_delta(best_row, baseline_row, "mean_ending_bankroll"),
            "mean_net_profit_delta": _metric_delta(best_row, baseline_row, "mean_net_profit"),
            "mean_log_loss_delta": _metric_delta(best_row, baseline_row, "mean_log_loss"),
            "mean_brier_delta": _metric_delta(best_row, baseline_row, "mean_brier"),
            "mean_ece_delta": _metric_delta(best_row, baseline_row, "mean_ece"),
            "mean_max_drawdown_delta": _metric_delta(best_row, baseline_row, "mean_max_drawdown"),
        },
        "policy": {
            "max_mean_drawdown_dollars": max_drawdown_limit,
            "min_bet_count": min_bet_count,
            "min_profitable_folds": min_profitable_folds,
            "max_ece_delta": max_ece_delta,
        },
        "validation_contract_complete": validation_contract_complete,
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


def _promotion_strategy_rows(scorecard: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if scorecard.empty or "strategy" not in scorecard.columns:
        return scorecard.copy()
    filtered = scorecard[scorecard["strategy"] == strategy].copy()
    return filtered if not filtered.empty else scorecard.copy()


def _build_candidate_scorecards(
    *,
    scorecard: pd.DataFrame,
    strategy: str,
    decision: dict[str, object],
    candidate_model_name: str,
    active_model_name: str,
    incumbent_model_name: str,
) -> list[CandidateScorecardRecord]:
    scoped = _promotion_strategy_rows(scorecard, strategy)
    if scoped.empty:
        return []
    sort_plan = [
        ("mean_ending_bankroll", False),
        ("mean_net_profit", False),
        ("mean_log_loss", True),
        ("mean_brier", True),
    ]
    sort_columns = [column for column, _ in sort_plan if column in scoped.columns]
    if sort_columns:
        ranked = scoped.sort_values(
            sort_columns,
            ascending=[ascending for column, ascending in sort_plan if column in sort_columns],
        ).reset_index(drop=True)
    else:
        ranked = scoped.reset_index(drop=True)
    anchor_rows = ranked[ranked["model_name"] == active_model_name]
    anchor_row = anchor_rows.iloc[0] if not anchor_rows.empty else ranked.iloc[0]

    scorecards: list[CandidateScorecardRecord] = []
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        model_name = str(row.get("model_name") or "")
        metadata = _model_contract_metadata(model_name)
        role = "alternate_contender"
        if model_name == candidate_model_name:
            role = "candidate_under_review"
        elif model_name == incumbent_model_name:
            role = "incumbent"
        elif model_name == active_model_name:
            role = "active_model"
        scorecards.append(
            CandidateScorecardRecord(
                model_name=model_name,
                lane=metadata["lane"],
                target_name=RESEARCH_DESK_TARGET_NAME,
                distribution=RESEARCH_DESK_DISTRIBUTION,
                link_function=RESEARCH_DESK_LINK_FUNCTION,
                feature_count=None,
                active_parameter_count=None,
                validation_metrics={
                    "strategy": strategy,
                    "mean_ending_bankroll": _safe_float(row.get("mean_ending_bankroll")),
                    "mean_net_profit": _safe_float(row.get("mean_net_profit")),
                    "mean_roi": _safe_float(row.get("mean_roi")),
                    "median_roi": _safe_float(row.get("median_roi")),
                    "mean_log_loss": _safe_float(row.get("mean_log_loss")),
                    "mean_brier": _safe_float(row.get("mean_brier")),
                    "mean_auc": _safe_float(row.get("mean_auc")),
                    "bet_count": _safe_int(row.get("bet_count")),
                },
                stability_metrics={
                    "scorecard_rank": rank,
                    "mean_turnover": _safe_float(row.get("mean_turnover")),
                    "mean_max_drawdown": _safe_float(row.get("mean_max_drawdown")),
                    "profitable_folds": _safe_int(row.get("profitable_folds")),
                    "profit_winning_folds": _safe_int(row.get("profit_winning_folds")),
                    "all_integrity_checks": bool(row.get("all_integrity_checks")),
                    "prediction_before_game": bool(row.get("prediction_before_game")),
                    "unique_prediction_keys": bool(row.get("unique_prediction_keys")),
                    "no_missing_results_for_scored": bool(row.get("no_missing_results_for_scored")),
                    "embargo_respected": bool(row.get("embargo_respected")),
                },
                calibration_summary={
                    "mean_ece": _safe_float(row.get("mean_ece")),
                },
                complement_summary={
                    "display_name": metadata["display_name"],
                    "family": metadata["family"],
                    "governance_note": metadata["governance_note"],
                    "strategy": strategy,
                    "promotion_role": role,
                    "active_model_name": active_model_name,
                    "mean_ending_bankroll_delta_vs_active": (
                        _safe_float(row.get("mean_ending_bankroll")) - _safe_float(anchor_row.get("mean_ending_bankroll"))
                        if _safe_float(row.get("mean_ending_bankroll")) is not None
                        and _safe_float(anchor_row.get("mean_ending_bankroll")) is not None
                        else None
                    ),
                    "mean_log_loss_delta_vs_active": (
                        _safe_float(row.get("mean_log_loss")) - _safe_float(anchor_row.get("mean_log_loss"))
                        if _safe_float(row.get("mean_log_loss")) is not None
                        and _safe_float(anchor_row.get("mean_log_loss")) is not None
                        else None
                    ),
                },
                rejection_reasons=_scorecard_rejection_reasons(
                    row=row,
                    strategy=strategy,
                    anchor_row=anchor_row,
                    candidate_model_name=candidate_model_name,
                    active_model_name=active_model_name,
                    decision=decision,
                ),
            )
        )
    return scorecards


def _build_promotion_decision_record(
    *,
    decision: dict[str, object],
    candidate_scorecards: list[CandidateScorecardRecord],
    active_model_name: str,
    incumbent_model_name: str,
    candidate_model_name: str,
    strategy: str,
    report_path: Path,
    scorecard_path: Path,
    scorecard_contract_path: Path,
) -> PromotionDecisionRecord:
    rejected_models = {
        record.model_name: list(record.rejection_reasons)
        for record in candidate_scorecards
        if record.rejection_reasons
    }
    evidence = {
        "decision_scope": "research_desk_promotion_review",
        "strategy": strategy,
        "candidate_under_review": candidate_model_name,
        "active_model_after_decision": active_model_name,
        "gates": decision.get("gates"),
        "failed_reasons": decision.get("failed_reasons"),
        "policy": decision.get("policy"),
        "source_checks": decision.get("source_checks"),
        "comparison_to_incumbent": decision.get("comparison_to_incumbent"),
        "best_candidate_row": decision.get("best_candidate_row"),
        "baseline_row": decision.get("baseline_row"),
        "top_scorecards": [record.to_dict() for record in candidate_scorecards[:3]],
        "artifacts": {
            "report_path": str(report_path),
            "scorecard_csv": str(scorecard_path),
            "scorecard_contract_json": str(scorecard_contract_path),
        },
    }
    return PromotionDecisionRecord(
        recommended_model=active_model_name,
        baseline_model=incumbent_model_name,
        status=str(decision.get("status") or "rejected"),
        rationale=str(decision.get("reason_summary") or ""),
        evidence=evidence,
        rejected_models=rejected_models,
    )


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
        "status": decision.get("status"),
        "reason_summary": decision.get("reason_summary"),
        "failed_reasons": decision.get("failed_reasons"),
        "candidate_model_name": decision.get("candidate_model_name"),
        "incumbent_model_name": decision.get("incumbent_model_name"),
        "promotion_decision": decision.get("promotion_decision"),
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
    strategy = str(promotion.get("strategy") or "default").strip() or "default"

    active_model_name = incumbent_model_name
    if decision.get("promoted"):
        active_model_name = str(result.best_candidate_model)

    scorecard = pd.read_csv(result.scorecard_path) if result.scorecard_path.exists() else pd.DataFrame()
    candidate_scorecards = _build_candidate_scorecards(
        scorecard=scorecard,
        strategy=strategy,
        decision=decision,
        candidate_model_name=str(result.best_candidate_model),
        active_model_name=active_model_name,
        incumbent_model_name=incumbent_model_name,
    )
    scorecard_contract_path = result.scorecard_path.with_name("candidate_scorecards.json")
    scorecard_contract_path.write_text(json.dumps([record.to_dict() for record in candidate_scorecards], sort_keys=True) + "\n")
    promotion_decision_record = _build_promotion_decision_record(
        decision=decision,
        candidate_scorecards=candidate_scorecards,
        active_model_name=active_model_name,
        incumbent_model_name=incumbent_model_name,
        candidate_model_name=str(result.best_candidate_model),
        strategy=strategy,
        report_path=result.report_path,
        scorecard_path=result.scorecard_path,
        scorecard_contract_path=scorecard_contract_path,
    )
    decision["active_model_name"] = active_model_name
    decision["strategy"] = strategy
    decision["candidate_scorecards"] = [record.to_dict() for record in candidate_scorecards]
    decision["promotion_decision"] = promotion_decision_record.to_dict()
    decision["artifacts"] = {
        "report_path": str(result.report_path),
        "scorecard_csv": str(result.scorecard_path),
        "scorecard_contract_json": str(scorecard_contract_path),
        "fold_metrics_path": str(result.fold_metrics_path),
        "promotion_path": str(result.promotion_path),
        "market_truth_summary_path": str(result.market_truth_summary_path) if result.market_truth_summary_path else None,
        "market_truth_predictions_path": str(result.market_truth_predictions_path) if result.market_truth_predictions_path else None,
        "market_truth_calibration_path": str(result.market_truth_calibration_path) if result.market_truth_calibration_path else None,
    }
    promotion_payload = {**promotion, **decision}
    result.promotion_path.write_text(json.dumps(promotion_payload, sort_keys=True) + "\n")
    current_best_models_path = model_compare_service.write_current_best_models_from_promotion_review(
        cfg,
        run_id=run_id,
        promotion_payload=promotion_payload,
    )
    if current_best_models_path is not None:
        promotion_payload.setdefault("artifacts", {})
        if isinstance(promotion_payload["artifacts"], dict):
            promotion_payload["artifacts"]["current_best_models_path"] = str(current_best_models_path)
        result.promotion_path.write_text(json.dumps(promotion_payload, sort_keys=True) + "\n")

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
        promotion_payload=promotion_payload,
    )
    _persist_decision(
        db,
        run_id=run_id,
        league=league,
        profile_key=profile_key,
        incumbent_model_name=incumbent_model_name,
        candidate_model_name=str(result.best_candidate_model),
        decision=promotion_payload,
    )

    if decision.get("promoted"):
        _persist_active_champion(
            db,
            league=league,
            profile_key=profile_key,
            model_name=active_model_name,
            run_id=run_id,
            brief=selected_brief,
            decision=promotion_payload,
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
