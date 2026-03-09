export type BetStrategy = "riskAdjusted" | "aggressive" | "capitalPreservation";
export type BetSizingStyle = "continuous" | "bucketed";

export const BET_STRATEGIES = ["riskAdjusted", "aggressive", "capitalPreservation"] as const;
export const BET_SIZING_STYLES = ["continuous", "bucketed"] as const;
export const DEFAULT_BET_STRATEGY: BetStrategy = "riskAdjusted";
export const DEFAULT_BET_SIZING_STYLE: BetSizingStyle = "continuous";

export type BetStrategyConfig = {
  label: string;
  shortLabel: string;
  description: string;
  allowUnderdogs: boolean;
  minEdge: number;
  minExpectedValue: number;
  fractionalKelly: number;
  maxBetUnits: number;
  maxDailyUnits: number;
};

export type BetSizingStyleConfig = {
  label: string;
  shortLabel: string;
  description: string;
};

export type BetStrategyRuleConfig = Pick<
  BetStrategyConfig,
  "allowUnderdogs" | "minEdge" | "minExpectedValue" | "fractionalKelly" | "maxBetUnits" | "maxDailyUnits"
>;

const SHARED_MIN_EDGE = 0.03;
const SHARED_MIN_EXPECTED_VALUE = 0.02;

const BET_STRATEGY_CONFIG: Record<BetStrategy, BetStrategyConfig> = {
  riskAdjusted: {
    label: "Balanced",
    shortLabel: "Standard risk",
    description:
      "Balanced baseline with the shared value screen, a 1.25-unit per-bet cap, and a 4-unit daily budget.",
    allowUnderdogs: true,
    minEdge: SHARED_MIN_EDGE,
    minExpectedValue: SHARED_MIN_EXPECTED_VALUE,
    fractionalKelly: 0.5,
    maxBetUnits: 1.25,
    maxDailyUnits: 4,
  },
  aggressive: {
    label: "Aggressive",
    shortLabel: "Wider caps",
    description: "Higher-variance sizing with the same value screen, a 1.75-unit per-bet cap, and a 6-unit daily budget.",
    allowUnderdogs: true,
    minEdge: SHARED_MIN_EDGE,
    minExpectedValue: SHARED_MIN_EXPECTED_VALUE,
    fractionalKelly: 0.75,
    maxBetUnits: 1.75,
    maxDailyUnits: 6,
  },
  capitalPreservation: {
    label: "Conservative",
    shortLabel: "Favorites only",
    description: "Lower-variance sizing, favorites only, a 0.75-unit per-bet cap, and a 2.5-unit daily budget.",
    allowUnderdogs: false,
    minEdge: SHARED_MIN_EDGE,
    minExpectedValue: SHARED_MIN_EXPECTED_VALUE,
    fractionalKelly: 0.25,
    maxBetUnits: 0.75,
    maxDailyUnits: 2.5,
  },
};

const BET_SIZING_STYLE_CONFIG: Record<BetSizingStyle, BetSizingStyleConfig> = {
  continuous: {
    label: "Continuous",
    shortLabel: "Edge-scaled",
    description: "Lets the stake scale continuously with the uncertainty-adjusted edge and market price.",
  },
  bucketed: {
    label: "Bucketed",
    shortLabel: "Rounded units",
    description: "Rounds the continuous recommendation into the legacy $0, $50, $100, or $150 buckets.",
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

export function normalizeBetSizingStyle(value?: string | null): BetSizingStyle {
  const normalized = String(value || "").trim().toLowerCase();
  switch (normalized) {
    case "bucketed":
    case "bucket":
    case "legacy":
    case "legacybucket":
    case "legacy-bucket":
    case "legacy_bucket":
      return "bucketed";
    case "continuous":
    case "kelly":
    default:
      return DEFAULT_BET_SIZING_STYLE;
  }
}

export function sizingStyleFromRequest(request: Request): BetSizingStyle {
  const url = new URL(request.url);
  return normalizeBetSizingStyle(url.searchParams.get("sizingStyle"));
}

export function getBetStrategyConfig(strategy: BetStrategy): BetStrategyConfig {
  return BET_STRATEGY_CONFIG[strategy];
}

export function toBetStrategyRuleConfig(strategyConfig: BetStrategyConfig): BetStrategyRuleConfig {
  return {
    allowUnderdogs: strategyConfig.allowUnderdogs,
    minEdge: strategyConfig.minEdge,
    minExpectedValue: strategyConfig.minExpectedValue,
    fractionalKelly: strategyConfig.fractionalKelly,
    maxBetUnits: strategyConfig.maxBetUnits,
    maxDailyUnits: strategyConfig.maxDailyUnits,
  };
}

export function getBetStrategyLabel(strategy: BetStrategy): string {
  return BET_STRATEGY_CONFIG[strategy].label;
}

export function getBetSizingStyleConfig(sizingStyle: BetSizingStyle): BetSizingStyleConfig {
  return BET_SIZING_STYLE_CONFIG[sizingStyle];
}
