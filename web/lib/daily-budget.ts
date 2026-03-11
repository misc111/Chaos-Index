import { REFERENCE_BANKROLL_DOLLARS } from "@/lib/betting";
import type { BetStrategyRuleConfig } from "@/lib/betting-strategy";

export const DAILY_BUDGET_QUERY_PARAM = "dailyBudget";
export const DAILY_BUDGET_STEP_DOLLARS = 25;
export const MIN_DAILY_BUDGET_DOLLARS = 0;
export const MAX_DAILY_BUDGET_DOLLARS = REFERENCE_BANKROLL_DOLLARS;

export function clampDailyBudgetDollars(value: number): number {
  if (!Number.isFinite(value)) return MIN_DAILY_BUDGET_DOLLARS;
  return Math.max(MIN_DAILY_BUDGET_DOLLARS, Math.min(MAX_DAILY_BUDGET_DOLLARS, Math.round(value)));
}

export function defaultDailyBudgetDollars(
  strategyConfig?: Pick<BetStrategyRuleConfig, "maxDailyBankrollPercent"> | null
): number {
  const budgetPercent = strategyConfig?.maxDailyBankrollPercent;
  if (typeof budgetPercent !== "number" || !Number.isFinite(budgetPercent) || budgetPercent <= 0) {
    return 0;
  }

  return clampDailyBudgetDollars((budgetPercent / 100) * REFERENCE_BANKROLL_DOLLARS);
}

export function parseDailyBudgetParam(value: string | null | undefined, fallback: number): number {
  if (typeof value !== "string" || value.trim() === "") {
    return fallback;
  }

  const numeric = Number(value);
  return Number.isFinite(numeric) ? clampDailyBudgetDollars(numeric) : fallback;
}
