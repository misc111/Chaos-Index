import { DEFAULT_BET_STRATEGY, getBetStrategyConfig, type BetStrategy } from "@/lib/betting-strategy";

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

export const BET_UNIT_DOLLARS = 100;
export const BET_UNIT_LABEL = `Bet per $${BET_UNIT_DOLLARS}`;

type BetDisplayRecommendation = {
  team: string | null;
  stake: number;
  reason: string;
};

const KELLY_FRACTION_PER_UNIT = 0.15;
const STAKE_ROUNDING_DOLLARS = 5;

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

export function computeBetDecision(row: BetInput, strategy: BetStrategy = DEFAULT_BET_STRATEGY): BetDecision {
  const strategyConfig = getBetStrategyConfig(strategy);
  const homeOdds = Number(row.home_moneyline);
  const awayOdds = Number(row.away_moneyline);
  if (!Number.isFinite(homeOdds) || !Number.isFinite(awayOdds) || homeOdds === 0 || awayOdds === 0) {
    return buildDecision(row, "none", 0, "Missing odds");
  }

  const pHomeRaw = Number(row.home_win_probability);
  if (!Number.isFinite(pHomeRaw)) return buildDecision(row, "none", 0, "Missing odds");
  const pHome = Math.min(1, Math.max(0, pHomeRaw));
  const pAway = 1 - pHome;
  if (Math.max(pHome, pAway) < 0.55) return buildDecision(row, "none", 0, "Too close");

  const impHome = americanToImpliedProbability(homeOdds);
  const impAway = americanToImpliedProbability(awayOdds);
  if (impHome === null || impAway === null) return buildDecision(row, "none", 0, "Missing odds");
  const impTotal = impHome + impAway;
  if (!Number.isFinite(impTotal) || impTotal <= 0) return buildDecision(row, "none", 0, "Missing odds");

  const fairHome = impHome / impTotal;
  const fairAway = impAway / impTotal;

  const decHome = americanToDecimalOdds(homeOdds);
  const decAway = americanToDecimalOdds(awayOdds);
  if (decHome === null || decAway === null) return buildDecision(row, "none", 0, "Missing odds");

  const evHome = pHome * decHome - 1;
  const evAway = pAway * decAway - 1;
  if (evHome <= 0 && evAway <= 0) return buildDecision(row, "none", 0, "Price fair");

  const side = evHome > evAway ? "home" : evAway > evHome ? "away" : pHome >= pAway ? "home" : "away";
  const modelProb = side === "home" ? pHome : pAway;
  const fairProb = side === "home" ? fairHome : fairAway;
  const edge = modelProb - fairProb;
  const ev = side === "home" ? evHome : evAway;
  if (edge < strategyConfig.minEdge || ev < strategyConfig.minExpectedValue) {
    return buildDecision(row, "none", 0, "Price fair");
  }

  const sideOdds = side === "home" ? homeOdds : awayOdds;
  if (!strategyConfig.allowUnderdogs && sideOdds > 0) {
    return buildDecision(row, "none", 0, "Risk-averse profile skips underdogs", fairProb, ev, edge);
  }
  const sideDecimalOdds = side === "home" ? decHome : decAway;
  const kellyFraction = decimalOddsToKellyFraction(modelProb, sideDecimalOdds);
  const stake =
    kellyFraction === null ? 0 : continuousStakeFromKelly(kellyFraction, strategyConfig.sizeMultiplier, strategyConfig.maxBetUnits);
  if (stake <= 0) return buildDecision(row, "none", 0, "Price fair");

  const isUnderdog = sideOdds > 0;
  return buildDecision(row, side, stake, isUnderdog ? "Underdog underpriced" : "Favorite underpriced", fairProb, ev, edge);
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
