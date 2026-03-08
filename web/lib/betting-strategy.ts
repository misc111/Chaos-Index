export type BetStrategy = "balanced" | "riskAverse" | "riskLoving";
export type BetSizingStyle = "continuous" | "bucketed";

export const BET_STRATEGIES = ["balanced", "riskAverse", "riskLoving"] as const;
export const BET_SIZING_STYLES = ["continuous", "bucketed"] as const;
export const DEFAULT_BET_STRATEGY: BetStrategy = "balanced";
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

const BET_STRATEGY_CONFIG: Record<BetStrategy, BetStrategyConfig> = {
  balanced: {
    label: "Balanced",
    shortLabel: "Balanced",
    description: "Bets favorites and underdogs with the standard edge filter and stake sizing.",
    allowUnderdogs: true,
    minEdge: 0.03,
    minExpectedValue: 0.02,
    sizeMultiplier: 1,
    maxBetUnits: 2,
  },
  riskAverse: {
    label: "Risk Averse",
    shortLabel: "Conservative",
    description: "Skips underdogs, demands a stronger edge, and sizes smaller to reduce drawdowns.",
    allowUnderdogs: false,
    minEdge: 0.04,
    minExpectedValue: 0.03,
    sizeMultiplier: 0.6,
    maxBetUnits: 1,
  },
  riskLoving: {
    label: "Risk Loving",
    shortLabel: "Aggressive",
    description: "Accepts thinner edges and sizes up more aggressively when the market looks wrong.",
    allowUnderdogs: true,
    minEdge: 0.025,
    minExpectedValue: 0.015,
    sizeMultiplier: 1.4,
    maxBetUnits: 3,
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
    case "balanced":
      return "balanced";
    case "riskaverse":
    case "risk-averse":
    case "risk_averse":
    case "conservative":
      return "riskAverse";
    case "riskloving":
    case "risk-loving":
    case "risk_loving":
    case "aggressive":
      return "riskLoving";
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

export function getBetStrategyLabel(strategy: BetStrategy): string {
  return BET_STRATEGY_CONFIG[strategy].label;
}

export function getBetSizingStyleConfig(sizingStyle: BetSizingStyle): BetSizingStyleConfig {
  return BET_SIZING_STYLE_CONFIG[sizingStyle];
}
