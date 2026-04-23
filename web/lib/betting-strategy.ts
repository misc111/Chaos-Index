import type { LeagueCode } from "@/lib/league";

export type BetStrategy = "riskAdjusted" | "aggressive" | "capitalPreservation";

export const BET_STRATEGIES = ["riskAdjusted", "aggressive", "capitalPreservation"] as const;
export const DEFAULT_BET_STRATEGY: BetStrategy = "riskAdjusted";
export const LEAGUE_DEFAULT_BET_STRATEGY: Record<LeagueCode, BetStrategy> = {
  MLB: "riskAdjusted",
};
export type BetRiskRegime = "normal" | "guarded";

export type BetStrategyConfig = {
  label: string;
  shortLabel: string;
  description: string;
  allowUnderdogs: boolean;
  maxUnderdogMoneyline?: number | null;
  minEdge: number;
  minExpectedValue: number;
  stakeScale: number;
  maxBetBankrollPercent: number;
  maxDailyBankrollPercent: number | null;
};

export type BetStrategyRuleConfig = Pick<
  BetStrategyConfig,
  | "allowUnderdogs"
  | "maxUnderdogMoneyline"
  | "minEdge"
  | "minExpectedValue"
  | "stakeScale"
  | "maxBetBankrollPercent"
  | "maxDailyBankrollPercent"
>;

const SHARED_MIN_EDGE = 0.03;
const SHARED_MIN_EXPECTED_VALUE = 0.02;

type BetStrategyTuning = Omit<BetStrategyConfig, "description">;
type BetStrategyExperimentOverride = Partial<Pick<BetStrategyConfig, "stakeScale" | "maxBetBankrollPercent" | "maxDailyBankrollPercent">>;

function formatPercent(value: number): string {
  return Number.isInteger(value) ? `${value}%` : `${value.toFixed(2)}%`;
}

function buildStrategyDescription(config: BetStrategyTuning): string {
  const dailyBudgetText =
    typeof config.maxDailyBankrollPercent === "number" && Number.isFinite(config.maxDailyBankrollPercent) && config.maxDailyBankrollPercent > 0
      ? `a ${formatPercent(config.maxDailyBankrollPercent)} nightly budget on the reference bankroll`
      : "no nightly budget cap";
  const longShotText =
    typeof config.maxUnderdogMoneyline === "number" && Number.isFinite(config.maxUnderdogMoneyline)
      ? ` Long-shot underdogs above +${Math.round(config.maxUnderdogMoneyline)} are skipped.`
      : "";
  return `${config.label} baseline with the shared value screen, a ${formatPercent(config.maxBetBankrollPercent)} per-bet cap, and ${dailyBudgetText}.${longShotText}`;
}

export const BETTING_STRATEGY_TUNING: Record<BetStrategy, BetStrategyTuning> = {
  riskAdjusted: {
    label: "Balanced",
    shortLabel: "Standard risk",
    allowUnderdogs: true,
    maxUnderdogMoneyline: null,
    minEdge: SHARED_MIN_EDGE,
    minExpectedValue: SHARED_MIN_EXPECTED_VALUE,
    stakeScale: 0.5,
    maxBetBankrollPercent: 1.25,
    maxDailyBankrollPercent: 4,
  },
  aggressive: {
    label: "Aggressive",
    shortLabel: "Wider caps",
    allowUnderdogs: true,
    maxUnderdogMoneyline: null,
    minEdge: SHARED_MIN_EDGE,
    minExpectedValue: SHARED_MIN_EXPECTED_VALUE,
    stakeScale: 0.75,
    maxBetBankrollPercent: 1.75,
    maxDailyBankrollPercent: 6,
  },
  capitalPreservation: {
    label: "Conservative",
    shortLabel: "Favorites only",
    allowUnderdogs: false,
    maxUnderdogMoneyline: null,
    minEdge: SHARED_MIN_EDGE,
    minExpectedValue: SHARED_MIN_EXPECTED_VALUE,
    stakeScale: 0.25,
    maxBetBankrollPercent: 0.75,
    maxDailyBankrollPercent: 2.5,
  },
};

const BET_STRATEGY_CONFIG: Record<BetStrategy, BetStrategyConfig> = Object.fromEntries(
  (Object.entries(BETTING_STRATEGY_TUNING) as [BetStrategy, BetStrategyTuning][]).map(([strategy, config]) => [
    strategy,
    {
      ...config,
      description: buildStrategyDescription(config),
    },
  ])
) as Record<BetStrategy, BetStrategyConfig>;

export const BETTING_STRATEGY_EXPERIMENT_OVERRIDES: Record<BetStrategy, BetStrategyExperimentOverride> = {
  riskAdjusted: {},
  aggressive: {},
  capitalPreservation: {},
};

function applyRiskRegimeAdjustments(
  config: BetStrategyConfig,
  riskRegime: BetRiskRegime = "normal"
): BetStrategyConfig {
  if (riskRegime !== "guarded") {
    return config;
  }
  return config;
}

export function applyBetStrategyExperimentOverrides(strategy: BetStrategy, config: BetStrategyConfig): BetStrategyConfig {
  const overrides = BETTING_STRATEGY_EXPERIMENT_OVERRIDES[strategy];
  if (!overrides || Object.keys(overrides).length === 0) {
    return config;
  }

  const merged: BetStrategyTuning = {
    label: config.label,
    shortLabel: config.shortLabel,
    allowUnderdogs: config.allowUnderdogs,
    maxUnderdogMoneyline: config.maxUnderdogMoneyline ?? null,
    minEdge: config.minEdge,
    minExpectedValue: config.minExpectedValue,
    stakeScale: overrides.stakeScale ?? config.stakeScale,
    maxBetBankrollPercent: overrides.maxBetBankrollPercent ?? config.maxBetBankrollPercent,
    maxDailyBankrollPercent:
      Object.prototype.hasOwnProperty.call(overrides, "maxDailyBankrollPercent")
        ? overrides.maxDailyBankrollPercent ?? null
        : config.maxDailyBankrollPercent,
  };

  return {
    ...config,
    ...merged,
    description: buildStrategyDescription(merged),
  };
}

export function normalizeBetStrategy(value?: string | null): BetStrategy {
  const normalized = String(value || "").trim().toLowerCase();
  switch (normalized) {
    case "riskadjusted":
    case "risk-adjusted":
    case "risk_adjusted":
    case "optimal":
    case "sharpe":
    case "sharpe-like":
    case "balanced":
      return "riskAdjusted";
    case "aggressiveev":
    case "aggressive-ev":
    case "aggressive_ev":
    case "highev":
    case "high-ev":
    case "high_ev":
    case "aggressive":
    case "riskloving":
    case "risk-loving":
    case "risk_loving":
      return "aggressive";
    case "capitalpreservation":
    case "capital-preservation":
    case "capital_preservation":
    case "preservation":
    case "lowrisk":
    case "low-risk":
    case "low_risk":
    case "riskaverse":
    case "risk-averse":
    case "risk_averse":
    case "conservative":
      return "capitalPreservation";
    default:
      return DEFAULT_BET_STRATEGY;
  }
}

export function getDefaultBetStrategyForLeague(league?: LeagueCode | null): BetStrategy {
  if (!league) {
    return DEFAULT_BET_STRATEGY;
  }
  return LEAGUE_DEFAULT_BET_STRATEGY[league] || DEFAULT_BET_STRATEGY;
}

export function strategyFromRequest(request: Request): BetStrategy {
  const url = new URL(request.url);
  const leagueParam = url.searchParams.get("league");
  const league = leagueParam === "MLB" ? leagueParam : null;
  const strategyParam = url.searchParams.get("strategy");
  return strategyParam ? normalizeBetStrategy(strategyParam) : getDefaultBetStrategyForLeague(league);
}

export function getBetStrategyConfig(
  strategy: BetStrategy,
  options?: {
    league?: LeagueCode | null;
    riskRegime?: BetRiskRegime;
  }
): BetStrategyConfig {
  const experimented = applyBetStrategyExperimentOverrides(strategy, BET_STRATEGY_CONFIG[strategy]);
  return applyRiskRegimeAdjustments(experimented, options?.riskRegime);
}

export function toBetStrategyRuleConfig(strategyConfig: BetStrategyConfig): BetStrategyRuleConfig {
  return {
    allowUnderdogs: strategyConfig.allowUnderdogs,
    maxUnderdogMoneyline: strategyConfig.maxUnderdogMoneyline ?? null,
    minEdge: strategyConfig.minEdge,
    minExpectedValue: strategyConfig.minExpectedValue,
    stakeScale: strategyConfig.stakeScale,
    maxBetBankrollPercent: strategyConfig.maxBetBankrollPercent,
    maxDailyBankrollPercent: strategyConfig.maxDailyBankrollPercent,
  };
}

export function getBetStrategyLabel(strategy: BetStrategy): string {
  return BET_STRATEGY_CONFIG[strategy].label;
}
