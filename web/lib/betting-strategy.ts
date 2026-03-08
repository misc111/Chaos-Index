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
  sizeMultiplier: number;
  maxBetUnits: number;
};

export type BetSizingStyleConfig = {
  label: string;
  shortLabel: string;
  description: string;
};

export type BetStrategyRuleConfig = Pick<
  BetStrategyConfig,
  "allowUnderdogs" | "minEdge" | "minExpectedValue" | "sizeMultiplier" | "maxBetUnits"
>;

const BET_STRATEGY_CONFIG: Record<BetStrategy, BetStrategyConfig> = {
  riskAdjusted: {
    label: "Risk-Adjusted Optimal",
    shortLabel: "Tangency",
    description: "Selected from the historical replay frontier at the highest zero-rate Sharpe ratio under continuous sizing.",
    allowUnderdogs: true,
    minEdge: 0.035,
    minExpectedValue: 0.025,
    sizeMultiplier: 0.75,
    maxBetUnits: 1.5,
  },
  aggressive: {
    label: "Aggressive",
    shortLabel: "Higher Return",
    description: "Selected from the same replay frontier at a higher-return, higher-volatility point than the tangency choice.",
    allowUnderdogs: true,
    minEdge: 0.025,
    minExpectedValue: 0.015,
    sizeMultiplier: 1.4,
    maxBetUnits: 3,
  },
  capitalPreservation: {
    label: "Capital Preservation",
    shortLabel: "Hates losses",
    description: "Selected from replay to minimize downside volatility and drawdown while maintaining a positive realized return.",
    allowUnderdogs: false,
    minEdge: 0.05,
    minExpectedValue: 0.035,
    sizeMultiplier: 0.45,
    maxBetUnits: 0.75,
  },
};

const BET_SIZING_STYLE_CONFIG: Record<BetSizingStyle, BetSizingStyleConfig> = {
  continuous: {
    label: "Continuous",
    shortLabel: "Kelly scaled",
    description: "Lets the stake scale continuously with the model edge and market price.",
  },
  bucketed: {
    label: "Bucketed",
    shortLabel: "Legacy buckets",
    description: "Snaps the stake into the legacy $0, $50, $100, or $150 buckets.",
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
    sizeMultiplier: strategyConfig.sizeMultiplier,
    maxBetUnits: strategyConfig.maxBetUnits,
  };
}

export function getBetStrategyLabel(strategy: BetStrategy): string {
  return BET_STRATEGY_CONFIG[strategy].label;
}

export function getBetSizingStyleConfig(sizingStyle: BetSizingStyle): BetSizingStyleConfig {
  return BET_SIZING_STYLE_CONFIG[sizingStyle];
}
