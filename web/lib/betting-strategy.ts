export type BetStrategy = "riskAdjusted" | "aggressive" | "capitalPreservation";

export const BET_STRATEGIES = ["riskAdjusted", "aggressive", "capitalPreservation"] as const;
export const DEFAULT_BET_STRATEGY: BetStrategy = "riskAdjusted";

export type BetStrategyConfig = {
  label: string;
  shortLabel: string;
  description: string;
  allowUnderdogs: boolean;
  minEdge: number;
  minExpectedValue: number;
  stakeScale: number;
  maxBetBankrollPercent: number;
  maxDailyBankrollPercent: number | null;
};

export type BetStrategyRuleConfig = Pick<
  BetStrategyConfig,
  "allowUnderdogs" | "minEdge" | "minExpectedValue" | "stakeScale" | "maxBetBankrollPercent" | "maxDailyBankrollPercent"
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
  return `${config.label} baseline with the shared value screen, a ${formatPercent(config.maxBetBankrollPercent)} per-bet cap, and ${dailyBudgetText}.`;
}

export const BETTING_STRATEGY_TUNING: Record<BetStrategy, BetStrategyTuning> = {
  riskAdjusted: {
    label: "Balanced",
    shortLabel: "Standard risk",
    allowUnderdogs: true,
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

export function applyBetStrategyExperimentOverrides(strategy: BetStrategy, config: BetStrategyConfig): BetStrategyConfig {
  const overrides = BETTING_STRATEGY_EXPERIMENT_OVERRIDES[strategy];
  if (!overrides || Object.keys(overrides).length === 0) {
    return config;
  }

  const merged: BetStrategyTuning = {
    label: config.label,
    shortLabel: config.shortLabel,
    allowUnderdogs: config.allowUnderdogs,
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

export function strategyFromRequest(request: Request): BetStrategy {
  const url = new URL(request.url);
  return normalizeBetStrategy(url.searchParams.get("strategy"));
}

export function getBetStrategyConfig(strategy: BetStrategy): BetStrategyConfig {
  return applyBetStrategyExperimentOverrides(strategy, BET_STRATEGY_CONFIG[strategy]);
}

export function toBetStrategyRuleConfig(strategyConfig: BetStrategyConfig): BetStrategyRuleConfig {
  return {
    allowUnderdogs: strategyConfig.allowUnderdogs,
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
