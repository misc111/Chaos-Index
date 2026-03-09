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
  maxDailyBankrollPercent: number;
};

export type BetStrategyRuleConfig = Pick<
  BetStrategyConfig,
  "allowUnderdogs" | "minEdge" | "minExpectedValue" | "stakeScale" | "maxBetBankrollPercent" | "maxDailyBankrollPercent"
>;

const SHARED_MIN_EDGE = 0.03;
const SHARED_MIN_EXPECTED_VALUE = 0.02;

const BET_STRATEGY_CONFIG: Record<BetStrategy, BetStrategyConfig> = {
  riskAdjusted: {
    label: "Balanced",
    shortLabel: "Standard risk",
    description:
      "Balanced baseline with the shared value screen, a 1.25% per-bet cap, and a 4% nightly budget on the reference bankroll.",
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
    description: "Higher-variance sizing with the same value screen, a 1.75% per-bet cap, and a 6% nightly budget on the reference bankroll.",
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
    description: "Lower-variance sizing, favorites only, a 0.75% per-bet cap, and a 2.5% nightly budget on the reference bankroll.",
    allowUnderdogs: false,
    minEdge: SHARED_MIN_EDGE,
    minExpectedValue: SHARED_MIN_EXPECTED_VALUE,
    stakeScale: 0.25,
    maxBetBankrollPercent: 0.75,
    maxDailyBankrollPercent: 2.5,
  },
};

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
  return BET_STRATEGY_CONFIG[strategy];
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
