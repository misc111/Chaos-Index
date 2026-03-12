from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

REFERENCE_BANKROLL_DOLLARS = 10_000
HISTORICAL_BANKROLL_START_DOLLARS = 5_000
STAKE_ROUNDING_DOLLARS = 5
REFERENCE_MARKET_WEIGHT = 0.7
REFERENCE_PEER_WEIGHT = 0.3
MIN_MODEL_CONFIDENCE_WEIGHT = 0.25
FULL_MARGIN_FOR_FULL_WEIGHT = 0.2
MIN_PEER_AGREEMENT_WEIGHT = 0.55
PEER_DISAGREEMENT_FOR_MIN_WEIGHT = 0.2


@dataclass(frozen=True)
class BetStrategyConfig:
    allow_underdogs: bool
    min_edge: float
    min_expected_value: float
    stake_scale: float
    max_bet_bankroll_percent: float
    max_daily_bankroll_percent: float | None


BETTING_STRATEGIES: dict[str, BetStrategyConfig] = {
    "riskAdjusted": BetStrategyConfig(True, 0.03, 0.02, 0.5, 1.25, 4.0),
    "aggressive": BetStrategyConfig(True, 0.03, 0.02, 0.75, 1.75, 6.0),
    "capitalPreservation": BetStrategyConfig(False, 0.03, 0.02, 0.25, 0.75, 2.5),
}


def _clamp_probability(value: Any) -> float:
    try:
        numeric = float(value)
    except Exception:
        return 0.5
    if not np.isfinite(numeric):
        return 0.5
    return max(0.0, min(1.0, numeric))


def american_to_implied_probability(odds: Any) -> float | None:
    try:
        value = float(odds)
    except Exception:
        return None
    if not np.isfinite(value) or value == 0:
        return None
    if value > 0:
        return 100.0 / (value + 100.0)
    return abs(value) / (abs(value) + 100.0)


def american_to_decimal_odds(odds: Any) -> float | None:
    try:
        value = float(odds)
    except Exception:
        return None
    if not np.isfinite(value) or value == 0:
        return None
    if value > 0:
        return 1.0 + value / 100.0
    return 1.0 + 100.0 / abs(value)


def _decimal_odds_to_base_share(probability: float, decimal_odds: float) -> float | None:
    if not np.isfinite(probability) or probability <= 0 or probability >= 1:
        return None
    if not np.isfinite(decimal_odds) or decimal_odds <= 1:
        return None
    net_odds = decimal_odds - 1.0
    fraction = (probability * decimal_odds - 1.0) / net_odds
    return float(fraction) if np.isfinite(fraction) else None


def _round_stake(amount: float) -> float:
    if not np.isfinite(amount) or amount <= 0:
        return 0.0
    rounded = round(float(amount) / STAKE_ROUNDING_DOLLARS) * STAKE_ROUNDING_DOLLARS
    return float(rounded) if rounded >= STAKE_ROUNDING_DOLLARS else 0.0


def _dollars_from_bankroll_share(share: float) -> float:
    return REFERENCE_BANKROLL_DOLLARS * share


def _resolve_peer_consensus(row: pd.Series, model_names: list[str], target_model: str, side: str) -> tuple[float | None, float | None]:
    values: list[float] = []
    selected_prob = _clamp_probability(row[target_model])
    selected_side_prob = selected_prob if side == "home" else 1.0 - selected_prob
    for model_name in model_names:
        if model_name == target_model or model_name not in row.index:
            continue
        value = row.get(model_name)
        if value is None or not np.isfinite(float(value)):
            continue
        prob = _clamp_probability(value)
        values.append(prob if side == "home" else 1.0 - prob)
    if not values:
        return None, None
    consensus = float(np.mean(values))
    return consensus, selected_side_prob - consensus


def _build_probability_adjustment(raw_probability: float, fair_probability: float, peer_consensus_probability: float | None) -> tuple[float, float, float]:
    normalized_raw = _clamp_probability(raw_probability)
    normalized_fair = _clamp_probability(fair_probability)
    normalized_peer = _clamp_probability(peer_consensus_probability) if peer_consensus_probability is not None else None
    reference_probability = (
        normalized_fair
        if normalized_peer is None
        else _clamp_probability(normalized_fair * REFERENCE_MARKET_WEIGHT + normalized_peer * REFERENCE_PEER_WEIGHT)
    )
    margin = abs(normalized_raw - 0.5)
    margin_weight = (
        1.0
        if margin >= FULL_MARGIN_FOR_FULL_WEIGHT
        else MIN_MODEL_CONFIDENCE_WEIGHT + ((1.0 - MIN_MODEL_CONFIDENCE_WEIGHT) * margin) / FULL_MARGIN_FOR_FULL_WEIGHT
    )
    peer_agreement_weight = (
        1.0
        if normalized_peer is None
        else max(MIN_PEER_AGREEMENT_WEIGHT, 1.0 - abs(normalized_raw - normalized_peer) / PEER_DISAGREEMENT_FOR_MIN_WEIGHT)
    )
    confidence_weight = max(MIN_MODEL_CONFIDENCE_WEIGHT, min(1.0, margin_weight * peer_agreement_weight))
    adjusted_probability = _clamp_probability(reference_probability + confidence_weight * (normalized_raw - reference_probability))
    return reference_probability, adjusted_probability, confidence_weight


def compute_strategy_decision(
    row: pd.Series,
    *,
    model_name: str,
    model_names: list[str],
    strategy_name: str,
) -> dict[str, Any]:
    strategy = BETTING_STRATEGIES[strategy_name]
    home_odds = row.get("home_moneyline")
    away_odds = row.get("away_moneyline")
    imp_home = american_to_implied_probability(home_odds)
    imp_away = american_to_implied_probability(away_odds)
    dec_home = american_to_decimal_odds(home_odds)
    dec_away = american_to_decimal_odds(away_odds)
    if imp_home is None or imp_away is None or dec_home is None or dec_away is None:
        return {
            "strategy": strategy_name,
            "model_name": model_name,
            "side": "none",
            "team": None,
            "stake": 0.0,
            "odds": None,
            "market_probability": None,
            "model_probability": None,
            "edge": None,
            "expected_value": None,
            "reason": "Missing odds",
        }

    fair_total = imp_home + imp_away
    fair_home = imp_home / fair_total
    fair_away = imp_away / fair_total
    raw_home = _clamp_probability(row.get(model_name))
    raw_away = 1.0 - raw_home
    home_peer, _ = _resolve_peer_consensus(row, model_names, model_name, "home")
    away_peer, _ = _resolve_peer_consensus(row, model_names, model_name, "away")
    _, adjusted_home, _ = _build_probability_adjustment(raw_home, fair_home, home_peer)
    _, adjusted_away, _ = _build_probability_adjustment(raw_away, fair_away, away_peer)
    ev_home = adjusted_home * dec_home - 1.0
    ev_away = adjusted_away * dec_away - 1.0
    if ev_home <= 0 and ev_away <= 0:
        return {
            "strategy": strategy_name,
            "model_name": model_name,
            "side": "none",
            "team": None,
            "stake": 0.0,
            "odds": None,
            "market_probability": None,
            "model_probability": None,
            "edge": None,
            "expected_value": None,
            "reason": "Adjusted price fair",
        }

    if ev_home > ev_away or (ev_home == ev_away and adjusted_home >= adjusted_away):
        side = "home"
        team = row.get("home_team")
        odds = float(home_odds)
        fair_probability = fair_home
        adjusted_probability = adjusted_home
        expected_value = ev_home
        decimal_odds = dec_home
    else:
        side = "away"
        team = row.get("away_team")
        odds = float(away_odds)
        fair_probability = fair_away
        adjusted_probability = adjusted_away
        expected_value = ev_away
        decimal_odds = dec_away

    edge = adjusted_probability - fair_probability
    if edge < strategy.min_edge or expected_value < strategy.min_expected_value:
        return {
            "strategy": strategy_name,
            "model_name": model_name,
            "side": "none",
            "team": None,
            "stake": 0.0,
            "odds": None,
            "market_probability": fair_probability,
            "model_probability": adjusted_probability,
            "edge": edge,
            "expected_value": expected_value,
            "reason": "Adjusted price fair",
        }

    if odds > 0 and not strategy.allow_underdogs:
        return {
            "strategy": strategy_name,
            "model_name": model_name,
            "side": "none",
            "team": None,
            "stake": 0.0,
            "odds": None,
            "market_probability": fair_probability,
            "model_probability": adjusted_probability,
            "edge": edge,
            "expected_value": expected_value,
            "reason": "Conservative skips underdogs",
        }

    base_share = _decimal_odds_to_base_share(adjusted_probability, decimal_odds)
    if base_share is None or base_share <= 0:
        stake = 0.0
    else:
        scaled_share = max(0.0, strategy.stake_scale * base_share)
        capped_share = min(strategy.max_bet_bankroll_percent / 100.0, scaled_share)
        stake = _round_stake(_dollars_from_bankroll_share(capped_share))

    return {
        "strategy": strategy_name,
        "model_name": model_name,
        "side": side if stake > 0 else "none",
        "team": team if stake > 0 else None,
        "stake": float(stake),
        "odds": odds if stake > 0 else None,
        "market_probability": fair_probability,
        "model_probability": adjusted_probability,
        "edge": edge,
        "expected_value": expected_value,
        "reason": (
            "Underdog underpriced after uncertainty adjustment"
            if stake > 0 and odds > 0
            else "Favorite underpriced after uncertainty adjustment"
            if stake > 0
            else "Adjusted price fair"
        ),
    }


def _apply_daily_risk_cap(decisions: list[dict[str, Any]], strategy_name: str) -> list[dict[str, Any]]:
    budget_percent = BETTING_STRATEGIES[strategy_name].max_daily_bankroll_percent
    if budget_percent is None or budget_percent <= 0:
        return decisions
    remaining = _dollars_from_bankroll_share(budget_percent / 100.0)
    ranked = sorted(
        decisions,
        key=lambda item: (
            float(item.get("expected_value") or float("-inf")),
            float(item.get("edge") or float("-inf")),
            float(item.get("stake") or 0.0),
        ),
        reverse=True,
    )
    for decision in ranked:
        stake = float(decision.get("stake") or 0.0)
        if stake <= 0:
            continue
        capped = _round_stake(min(stake, remaining))
        if capped <= 0:
            decision["stake"] = 0.0
            decision["side"] = "none"
            decision["team"] = None
            decision["odds"] = None
            decision["reason"] = "Daily risk budget exhausted"
            continue
        if capped < stake:
            decision["stake"] = capped
            decision["reason"] = f"{decision['reason']}; daily risk cap"
        remaining -= float(decision["stake"])
    return decisions


def settle_bet(decision: dict[str, Any], home_win: Any) -> dict[str, Any]:
    stake = float(decision.get("stake") or 0.0)
    side = str(decision.get("side") or "none")
    odds = decision.get("odds")
    home_won = bool(int(home_win)) if pd.notna(home_win) else False
    if stake <= 0 or side == "none" or odds is None:
        return {"outcome": "no_bet", "profit": 0.0, "payout": 0.0}
    won = (side == "home" and home_won) or (side == "away" and not home_won)
    if not won:
        return {"outcome": "loss", "profit": -stake, "payout": 0.0}
    odds_value = float(odds)
    profit = stake * (odds_value / 100.0) if odds_value > 0 else stake * (100.0 / abs(odds_value))
    return {"outcome": "win", "profit": profit, "payout": stake + profit}


def score_betting_performance(
    prediction_frame: pd.DataFrame,
    *,
    model_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if prediction_frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    decision_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    work = prediction_frame.copy()
    work["date_central"] = pd.to_datetime(work["start_time_utc"], errors="coerce", utc=True).dt.tz_convert("America/Chicago").dt.date.astype(str)

    for model_name in model_names:
        for strategy_name in BETTING_STRATEGIES:
            bankroll = HISTORICAL_BANKROLL_START_DOLLARS
            peak_bankroll = bankroll
            max_drawdown = 0.0
            total_profit = 0.0
            total_risked = 0.0
            total_bets = 0
            wins = 0
            bankroll_path: list[float] = [bankroll]
            for date_key, slate in work.groupby("date_central", sort=True):
                decisions = []
                for row in slate.itertuples(index=False):
                    decision = compute_strategy_decision(pd.Series(row._asdict()), model_name=model_name, model_names=model_names, strategy_name=strategy_name)
                    decision["game_id"] = row.game_id
                    decision["date_central"] = date_key
                    decisions.append(decision)
                decisions = _apply_daily_risk_cap(decisions, strategy_name)
                for decision in decisions:
                    row = slate[slate["game_id"] == decision["game_id"]].iloc[0]
                    settlement = settle_bet(decision, row["home_win"])
                    profit = float(settlement["profit"])
                    if settlement["outcome"] == "win":
                        wins += 1
                    if decision["stake"] > 0:
                        total_bets += 1
                        total_risked += float(decision["stake"])
                    total_profit += profit
                    bankroll += profit
                    peak_bankroll = max(peak_bankroll, bankroll)
                    if peak_bankroll > 0:
                        max_drawdown = max(max_drawdown, (peak_bankroll - bankroll) / peak_bankroll)
                    bankroll_path.append(bankroll)
                    decision_rows.append(
                        decision
                        | settlement
                        | {
                            "game_id": int(decision["game_id"]),
                            "date_central": date_key,
                            "bankroll_after": bankroll,
                        }
                    )
            summary_rows.append(
                {
                    "model_name": model_name,
                    "strategy": strategy_name,
                    "starting_bankroll": HISTORICAL_BANKROLL_START_DOLLARS,
                    "ending_bankroll": bankroll,
                    "net_profit": total_profit,
                    "total_risked": total_risked,
                    "roi": (total_profit / total_risked) if total_risked > 0 else 0.0,
                    "turnover": (total_risked / HISTORICAL_BANKROLL_START_DOLLARS) if HISTORICAL_BANKROLL_START_DOLLARS > 0 else 0.0,
                    "max_drawdown": max_drawdown,
                    "bet_count": total_bets,
                    "wins": wins,
                    "win_rate": (wins / total_bets) if total_bets > 0 else 0.0,
                    "path_points": len(bankroll_path),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(decision_rows)
