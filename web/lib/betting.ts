import {
  DEFAULT_BET_SIZING_STYLE,
  DEFAULT_BET_STRATEGY,
  getBetStrategyConfig,
  type BetStrategyRuleConfig,
  type BetSizingStyle,
  type BetStrategy,
} from "@/lib/betting-strategy";

export type ExpectedSide = "home" | "away" | "none";

export type BetInput = {
  home_team: string;
  away_team: string;
  home_win_probability: number;
  home_moneyline?: number | null;
  away_moneyline?: number | null;
};

export type BetDecision = {
  bet: string;
  reason: string;
  side: ExpectedSide;
  team: string | null;
  stake: number;
  odds: number | null;
  modelProbability: number | null;
  marketProbability: number | null;
  edge: number | null;
  expectedValue: number | null;
};

export type BetSettlement = {
  outcome: "win" | "loss" | "no_bet";
  profit: number;
  payout: number;
};

export type BetDecisionTrace = {
  decision: BetDecision;
  strategyLabel: string;
  sizingStyle: BetSizingStyle;
  strategyConfig: BetStrategyRuleConfig;
  homeModelProbability: number | null;
  awayModelProbability: number | null;
  homeFairProbability: number | null;
  awayFairProbability: number | null;
  homeExpectedValue: number | null;
  awayExpectedValue: number | null;
  candidateSide: ExpectedSide;
  candidateTeam: string | null;
  candidateOdds: number | null;
  candidateIsUnderdog: boolean | null;
  candidateModelProbability: number | null;
  candidateMarketProbability: number | null;
  candidateEdge: number | null;
  candidateExpectedValue: number | null;
  kellyFraction: number | null;
  rawKellyUnits: number | null;
  cappedKellyUnits: number | null;
  continuousStake: number;
  bucketedStake: number;
  finalStake: number;
  gates: {
    oddsAvailable: boolean;
    confidence: boolean;
    positiveExpectedValue: boolean;
    edge: boolean;
    expectedValue: boolean;
    underdogAllowed: boolean;
  };
};

export const BET_UNIT_DOLLARS = 100;
export const BET_UNIT_LABEL = `Bet per $${BET_UNIT_DOLLARS}`;

type BetDisplayRecommendation = {
  team: string | null;
  stake: number;
  reason: string;
};

const KELLY_FRACTION_PER_UNIT = 0.15;
const STAKE_ROUNDING_DOLLARS = 5;
const LEGACY_BUCKET_STAKES = [0, 50, 100, 150] as const;

function betAmountFromUnits(units: number): number {
  return BET_UNIT_DOLLARS * units;
}

function roundStakeAmount(amount: number): number {
  if (!Number.isFinite(amount) || amount <= 0) return 0;
  const rounded = Math.round(amount / STAKE_ROUNDING_DOLLARS) * STAKE_ROUNDING_DOLLARS;
  return rounded >= STAKE_ROUNDING_DOLLARS ? rounded : 0;
}

function decimalOddsToKellyFraction(probability: number, decimalOdds: number): number | null {
  if (!Number.isFinite(probability) || probability <= 0 || probability >= 1) return null;
  if (!Number.isFinite(decimalOdds) || decimalOdds <= 1) return null;

  const netOdds = decimalOdds - 1;
  const fraction = (probability * decimalOdds - 1) / netOdds;
  return Number.isFinite(fraction) ? fraction : null;
}

function continuousStakeFromKelly(kellyFraction: number, sizeMultiplier: number, maxBetUnits: number): number {
  if (!Number.isFinite(kellyFraction) || kellyFraction <= 0) return 0;

  const units = Math.min(maxBetUnits, Math.max(0, (kellyFraction / KELLY_FRACTION_PER_UNIT) * sizeMultiplier));
  return roundStakeAmount(betAmountFromUnits(units));
}

function bucketedStakeFromAmount(amount: number): number {
  if (!Number.isFinite(amount) || amount <= 0) return 0;

  const cappedAmount = Math.min(amount, LEGACY_BUCKET_STAKES[LEGACY_BUCKET_STAKES.length - 1]);
  let closestStake: number = LEGACY_BUCKET_STAKES[0];
  let closestDistance = Math.abs(cappedAmount - closestStake);

  for (const candidate of LEGACY_BUCKET_STAKES.slice(1)) {
    const distance = Math.abs(cappedAmount - candidate);
    if (distance < closestDistance) {
      closestStake = candidate;
      closestDistance = distance;
    }
  }

  return closestStake;
}

export function expectedSide(homeWinProbability: number): ExpectedSide {
  if (homeWinProbability > 0.55) return "home";
  if (homeWinProbability < 0.45) return "away";
  return "none";
}

export function expectedWinChance(homeWinProbability: number, side: ExpectedSide): number {
  if (side === "home") return homeWinProbability;
  if (side === "away") return 1 - homeWinProbability;
  return Math.max(homeWinProbability, 1 - homeWinProbability);
}

export function americanToImpliedProbability(odds: number): number | null {
  if (!Number.isFinite(odds) || odds === 0) return null;
  if (odds > 0) return 100 / (odds + 100);
  const absOdds = Math.abs(odds);
  return absOdds / (absOdds + 100);
}

export function americanToDecimalOdds(odds: number): number | null {
  if (!Number.isFinite(odds) || odds === 0) return null;
  if (odds > 0) return 1 + odds / 100;
  return 1 + 100 / Math.abs(odds);
}

export function formatBetLabel(team: string | null, stake: number): string {
  if (stake <= 0 || !team) return "$0";
  if (!Number.isFinite(stake)) return "$0";

  const fractionDigits = Number.isInteger(stake) ? 0 : 2;
  return `$${stake.toFixed(fractionDigits)} ${team}`;
}

export function formatBetUnitLabel(team: string | null, stake: number): string {
  return formatBetLabel(team, stake);
}

export function formatBetUnitRecommendation(recommendation: BetDisplayRecommendation): { label: string; reason: string } {
  return {
    label: formatBetUnitLabel(recommendation.team, recommendation.stake),
    reason: recommendation.reason,
  };
}

function buildDecision(
  row: BetInput,
  side: ExpectedSide,
  stake: number,
  reason: string,
  fairProb?: number | null,
  ev?: number | null,
  edge?: number | null
): BetDecision {
  const team = side === "home" ? row.home_team : side === "away" ? row.away_team : null;
  const odds = side === "home" ? Number(row.home_moneyline) : side === "away" ? Number(row.away_moneyline) : null;

  return {
    bet: formatBetLabel(team, stake),
    reason,
    side,
    team,
    stake,
    odds: Number.isFinite(odds) ? odds : null,
    modelProbability: team ? (side === "home" ? row.home_win_probability : 1 - row.home_win_probability) : null,
    marketProbability: typeof fairProb === "number" && Number.isFinite(fairProb) ? fairProb : null,
    edge: typeof edge === "number" && Number.isFinite(edge) ? edge : null,
    expectedValue: typeof ev === "number" && Number.isFinite(ev) ? ev : null,
  };
}

export function computeBetDecision(
  row: BetInput,
  strategy: BetStrategy = DEFAULT_BET_STRATEGY,
  sizingStyle: BetSizingStyle = DEFAULT_BET_SIZING_STYLE,
  strategyConfigOverride?: BetStrategyRuleConfig
): BetDecision {
  return explainBetDecision(row, strategy, sizingStyle, strategyConfigOverride).decision;
}

export function explainBetDecision(
  row: BetInput,
  strategy: BetStrategy = DEFAULT_BET_STRATEGY,
  sizingStyle: BetSizingStyle = DEFAULT_BET_SIZING_STYLE,
  strategyConfigOverride?: BetStrategyRuleConfig,
  strategyLabelOverride?: string
): BetDecisionTrace {
  const strategyConfig = strategyConfigOverride || getBetStrategyConfig(strategy);
  const strategyLabel = strategyLabelOverride || getBetStrategyConfig(strategy).label;
  const homeOdds = Number(row.home_moneyline);
  const awayOdds = Number(row.away_moneyline);
  const oddsAvailable = Number.isFinite(homeOdds) && Number.isFinite(awayOdds) && homeOdds !== 0 && awayOdds !== 0;
  const baseGates = {
    oddsAvailable,
    confidence: false,
    positiveExpectedValue: false,
    edge: false,
    expectedValue: false,
    underdogAllowed: false,
  };

  if (!oddsAvailable) {
    return {
      decision: buildDecision(row, "none", 0, "Missing odds"),
      strategyLabel,
      sizingStyle,
      strategyConfig,
      homeModelProbability: null,
      awayModelProbability: null,
      homeFairProbability: null,
      awayFairProbability: null,
      homeExpectedValue: null,
      awayExpectedValue: null,
      candidateSide: "none",
      candidateTeam: null,
      candidateOdds: null,
      candidateIsUnderdog: null,
      candidateModelProbability: null,
      candidateMarketProbability: null,
      candidateEdge: null,
      candidateExpectedValue: null,
      kellyFraction: null,
      rawKellyUnits: null,
      cappedKellyUnits: null,
      continuousStake: 0,
      bucketedStake: 0,
      finalStake: 0,
      gates: baseGates,
    };
  }

  if (!Number.isFinite(homeOdds) || !Number.isFinite(awayOdds) || homeOdds === 0 || awayOdds === 0) {
    return {
      decision: buildDecision(row, "none", 0, "Missing odds"),
      strategyLabel,
      sizingStyle,
      strategyConfig,
      homeModelProbability: null,
      awayModelProbability: null,
      homeFairProbability: null,
      awayFairProbability: null,
      homeExpectedValue: null,
      awayExpectedValue: null,
      candidateSide: "none",
      candidateTeam: null,
      candidateOdds: null,
      candidateIsUnderdog: null,
      candidateModelProbability: null,
      candidateMarketProbability: null,
      candidateEdge: null,
      candidateExpectedValue: null,
      kellyFraction: null,
      rawKellyUnits: null,
      cappedKellyUnits: null,
      continuousStake: 0,
      bucketedStake: 0,
      finalStake: 0,
      gates: baseGates,
    };
  }

  const pHomeRaw = Number(row.home_win_probability);
  if (!Number.isFinite(pHomeRaw)) {
    return {
      decision: buildDecision(row, "none", 0, "Missing odds"),
      strategyLabel,
      sizingStyle,
      strategyConfig,
      homeModelProbability: null,
      awayModelProbability: null,
      homeFairProbability: null,
      awayFairProbability: null,
      homeExpectedValue: null,
      awayExpectedValue: null,
      candidateSide: "none",
      candidateTeam: null,
      candidateOdds: null,
      candidateIsUnderdog: null,
      candidateModelProbability: null,
      candidateMarketProbability: null,
      candidateEdge: null,
      candidateExpectedValue: null,
      kellyFraction: null,
      rawKellyUnits: null,
      cappedKellyUnits: null,
      continuousStake: 0,
      bucketedStake: 0,
      finalStake: 0,
      gates: baseGates,
    };
  }
  const pHome = Math.min(1, Math.max(0, pHomeRaw));
  const pAway = 1 - pHome;
  const confidence = Math.max(pHome, pAway) >= 0.55;
  if (!confidence) {
    return {
      decision: buildDecision(row, "none", 0, "Too close"),
      strategyLabel,
      sizingStyle,
      strategyConfig,
      homeModelProbability: pHome,
      awayModelProbability: pAway,
      homeFairProbability: null,
      awayFairProbability: null,
      homeExpectedValue: null,
      awayExpectedValue: null,
      candidateSide: "none",
      candidateTeam: null,
      candidateOdds: null,
      candidateIsUnderdog: null,
      candidateModelProbability: null,
      candidateMarketProbability: null,
      candidateEdge: null,
      candidateExpectedValue: null,
      kellyFraction: null,
      rawKellyUnits: null,
      cappedKellyUnits: null,
      continuousStake: 0,
      bucketedStake: 0,
      finalStake: 0,
      gates: {
        ...baseGates,
        confidence,
      },
    };
  }

  const impHome = americanToImpliedProbability(homeOdds);
  const impAway = americanToImpliedProbability(awayOdds);
  if (impHome === null || impAway === null) {
    return {
      decision: buildDecision(row, "none", 0, "Missing odds"),
      strategyLabel,
      sizingStyle,
      strategyConfig,
      homeModelProbability: pHome,
      awayModelProbability: pAway,
      homeFairProbability: null,
      awayFairProbability: null,
      homeExpectedValue: null,
      awayExpectedValue: null,
      candidateSide: "none",
      candidateTeam: null,
      candidateOdds: null,
      candidateIsUnderdog: null,
      candidateModelProbability: null,
      candidateMarketProbability: null,
      candidateEdge: null,
      candidateExpectedValue: null,
      kellyFraction: null,
      rawKellyUnits: null,
      cappedKellyUnits: null,
      continuousStake: 0,
      bucketedStake: 0,
      finalStake: 0,
      gates: {
        ...baseGates,
        confidence,
      },
    };
  }
  const impTotal = impHome + impAway;
  if (!Number.isFinite(impTotal) || impTotal <= 0) {
    return {
      decision: buildDecision(row, "none", 0, "Missing odds"),
      strategyLabel,
      sizingStyle,
      strategyConfig,
      homeModelProbability: pHome,
      awayModelProbability: pAway,
      homeFairProbability: null,
      awayFairProbability: null,
      homeExpectedValue: null,
      awayExpectedValue: null,
      candidateSide: "none",
      candidateTeam: null,
      candidateOdds: null,
      candidateIsUnderdog: null,
      candidateModelProbability: null,
      candidateMarketProbability: null,
      candidateEdge: null,
      candidateExpectedValue: null,
      kellyFraction: null,
      rawKellyUnits: null,
      cappedKellyUnits: null,
      continuousStake: 0,
      bucketedStake: 0,
      finalStake: 0,
      gates: {
        ...baseGates,
        confidence,
      },
    };
  }

  const fairHome = impHome / impTotal;
  const fairAway = impAway / impTotal;

  const decHome = americanToDecimalOdds(homeOdds);
  const decAway = americanToDecimalOdds(awayOdds);
  if (decHome === null || decAway === null) {
    return {
      decision: buildDecision(row, "none", 0, "Missing odds"),
      strategyLabel,
      sizingStyle,
      strategyConfig,
      homeModelProbability: pHome,
      awayModelProbability: pAway,
      homeFairProbability: fairHome,
      awayFairProbability: fairAway,
      homeExpectedValue: null,
      awayExpectedValue: null,
      candidateSide: "none",
      candidateTeam: null,
      candidateOdds: null,
      candidateIsUnderdog: null,
      candidateModelProbability: null,
      candidateMarketProbability: null,
      candidateEdge: null,
      candidateExpectedValue: null,
      kellyFraction: null,
      rawKellyUnits: null,
      cappedKellyUnits: null,
      continuousStake: 0,
      bucketedStake: 0,
      finalStake: 0,
      gates: {
        ...baseGates,
        confidence,
      },
    };
  }

  const evHome = pHome * decHome - 1;
  const evAway = pAway * decAway - 1;
  const positiveExpectedValue = evHome > 0 || evAway > 0;
  if (!positiveExpectedValue) {
    return {
      decision: buildDecision(row, "none", 0, "Price fair"),
      strategyLabel,
      sizingStyle,
      strategyConfig,
      homeModelProbability: pHome,
      awayModelProbability: pAway,
      homeFairProbability: fairHome,
      awayFairProbability: fairAway,
      homeExpectedValue: evHome,
      awayExpectedValue: evAway,
      candidateSide: "none",
      candidateTeam: null,
      candidateOdds: null,
      candidateIsUnderdog: null,
      candidateModelProbability: null,
      candidateMarketProbability: null,
      candidateEdge: null,
      candidateExpectedValue: null,
      kellyFraction: null,
      rawKellyUnits: null,
      cappedKellyUnits: null,
      continuousStake: 0,
      bucketedStake: 0,
      finalStake: 0,
      gates: {
        ...baseGates,
        confidence,
        positiveExpectedValue,
      },
    };
  }

  const side = evHome > evAway ? "home" : evAway > evHome ? "away" : pHome >= pAway ? "home" : "away";
  const modelProb = side === "home" ? pHome : pAway;
  const fairProb = side === "home" ? fairHome : fairAway;
  const edge = modelProb - fairProb;
  const ev = side === "home" ? evHome : evAway;
  const candidateTeam = side === "home" ? row.home_team : row.away_team;
  const candidateOdds = side === "home" ? homeOdds : awayOdds;
  const candidateIsUnderdog = candidateOdds > 0;
  const edgeGate = edge >= strategyConfig.minEdge;
  const expectedValueGate = ev >= strategyConfig.minExpectedValue;
  const underdogAllowed = strategyConfig.allowUnderdogs || !candidateIsUnderdog;

  if (!edgeGate || !expectedValueGate) {
    return {
      decision: buildDecision(row, "none", 0, "Price fair"),
      strategyLabel,
      sizingStyle,
      strategyConfig,
      homeModelProbability: pHome,
      awayModelProbability: pAway,
      homeFairProbability: fairHome,
      awayFairProbability: fairAway,
      homeExpectedValue: evHome,
      awayExpectedValue: evAway,
      candidateSide: side,
      candidateTeam,
      candidateOdds,
      candidateIsUnderdog,
      candidateModelProbability: modelProb,
      candidateMarketProbability: fairProb,
      candidateEdge: edge,
      candidateExpectedValue: ev,
      kellyFraction: null,
      rawKellyUnits: null,
      cappedKellyUnits: null,
      continuousStake: 0,
      bucketedStake: 0,
      finalStake: 0,
      gates: {
        ...baseGates,
        confidence,
        positiveExpectedValue,
        edge: edgeGate,
        expectedValue: expectedValueGate,
        underdogAllowed,
      },
    };
  }

  if (!underdogAllowed) {
    return {
      decision: buildDecision(row, "none", 0, `${strategyLabel} skips underdogs`, fairProb, ev, edge),
      strategyLabel,
      sizingStyle,
      strategyConfig,
      homeModelProbability: pHome,
      awayModelProbability: pAway,
      homeFairProbability: fairHome,
      awayFairProbability: fairAway,
      homeExpectedValue: evHome,
      awayExpectedValue: evAway,
      candidateSide: side,
      candidateTeam,
      candidateOdds,
      candidateIsUnderdog,
      candidateModelProbability: modelProb,
      candidateMarketProbability: fairProb,
      candidateEdge: edge,
      candidateExpectedValue: ev,
      kellyFraction: null,
      rawKellyUnits: null,
      cappedKellyUnits: null,
      continuousStake: 0,
      bucketedStake: 0,
      finalStake: 0,
      gates: {
        ...baseGates,
        confidence,
        positiveExpectedValue,
        edge: edgeGate,
        expectedValue: expectedValueGate,
        underdogAllowed,
      },
    };
  }
  const sideDecimalOdds = side === "home" ? decHome : decAway;
  const kellyFraction = decimalOddsToKellyFraction(modelProb, sideDecimalOdds);
  const rawKellyUnits =
    kellyFraction === null ? null : Math.max(0, (kellyFraction / KELLY_FRACTION_PER_UNIT) * strategyConfig.sizeMultiplier);
  const cappedKellyUnits =
    rawKellyUnits === null ? null : Math.min(strategyConfig.maxBetUnits, Math.max(0, rawKellyUnits));
  const continuousStake =
    kellyFraction === null ? 0 : continuousStakeFromKelly(kellyFraction, strategyConfig.sizeMultiplier, strategyConfig.maxBetUnits);
  const bucketedStake = bucketedStakeFromAmount(continuousStake);
  const stake = sizingStyle === "bucketed" ? bucketedStakeFromAmount(continuousStake) : continuousStake;
  if (stake <= 0) {
    return {
      decision: buildDecision(row, "none", 0, "Price fair"),
      strategyLabel,
      sizingStyle,
      strategyConfig,
      homeModelProbability: pHome,
      awayModelProbability: pAway,
      homeFairProbability: fairHome,
      awayFairProbability: fairAway,
      homeExpectedValue: evHome,
      awayExpectedValue: evAway,
      candidateSide: side,
      candidateTeam,
      candidateOdds,
      candidateIsUnderdog,
      candidateModelProbability: modelProb,
      candidateMarketProbability: fairProb,
      candidateEdge: edge,
      candidateExpectedValue: ev,
      kellyFraction,
      rawKellyUnits,
      cappedKellyUnits,
      continuousStake,
      bucketedStake,
      finalStake: 0,
      gates: {
        ...baseGates,
        confidence,
        positiveExpectedValue,
        edge: edgeGate,
        expectedValue: expectedValueGate,
        underdogAllowed,
      },
    };
  }

  const decision = buildDecision(
    row,
    side,
    stake,
    candidateIsUnderdog ? "Underdog underpriced" : "Favorite underpriced",
    fairProb,
    ev,
    edge
  );

  return {
    decision,
    strategyLabel,
    sizingStyle,
    strategyConfig,
    homeModelProbability: pHome,
    awayModelProbability: pAway,
    homeFairProbability: fairHome,
    awayFairProbability: fairAway,
    homeExpectedValue: evHome,
    awayExpectedValue: evAway,
    candidateSide: side,
    candidateTeam,
    candidateOdds,
    candidateIsUnderdog,
    candidateModelProbability: modelProb,
    candidateMarketProbability: fairProb,
    candidateEdge: edge,
    candidateExpectedValue: ev,
    kellyFraction,
    rawKellyUnits,
    cappedKellyUnits,
    continuousStake,
    bucketedStake,
    finalStake: stake,
    gates: {
      ...baseGates,
      confidence,
      positiveExpectedValue,
      edge: edgeGate,
      expectedValue: expectedValueGate,
      underdogAllowed,
    },
  };
}

export function settleBet(decision: BetDecision, outcomeHomeWin?: number | boolean | null): BetSettlement {
  const homeWin = typeof outcomeHomeWin === "boolean" ? outcomeHomeWin : Number(outcomeHomeWin) === 1;
  if (decision.stake <= 0 || decision.side === "none" || !Number.isFinite(decision.odds)) {
    return { outcome: "no_bet", profit: 0, payout: 0 };
  }

  const won = (decision.side === "home" && homeWin) || (decision.side === "away" && !homeWin);
  if (!won) {
    return { outcome: "loss", profit: -decision.stake, payout: 0 };
  }

  const odds = Number(decision.odds);
  const profit = odds > 0 ? decision.stake * (odds / 100) : decision.stake * (100 / Math.abs(odds));
  return { outcome: "win", profit, payout: decision.stake + profit };
}
