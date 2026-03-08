export type BetStrategy = "balanced" | "riskAverse" | "riskLoving";

export const BET_STRATEGIES = ["balanced", "riskAverse", "riskLoving"] as const;
export const DEFAULT_BET_STRATEGY: BetStrategy = "balanced";

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

export function getBetStrategyConfig(strategy: BetStrategy): BetStrategyConfig {
  return BET_STRATEGY_CONFIG[strategy];
}

export function getBetStrategyLabel(strategy: BetStrategy): string {
  return BET_STRATEGY_CONFIG[strategy].label;
}
