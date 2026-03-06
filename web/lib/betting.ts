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

type BetLabelOptions = {
  stakeScale?: number;
};

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

export function formatBetLabel(team: string | null, stake: number, options: BetLabelOptions = {}): string {
  const scale = Number(options.stakeScale);
  if (stake <= 0 || !team) return "$0";
  if (!Number.isFinite(stake) || !Number.isFinite(scale) || scale <= 0) return "$0";

  const normalizedStake = stake / scale;
  const fractionDigits = scale === 1 ? 0 : 2;
  return `$${normalizedStake.toFixed(fractionDigits)} ${team}`;
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

export function computeBetDecision(row: BetInput): BetDecision {
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
  if (edge < 0.03 || ev < 0.02) return buildDecision(row, "none", 0, "Price fair");

  const sideOdds = side === "home" ? homeOdds : awayOdds;
  const isUnderdog = sideOdds > 0;

  if (isUnderdog) {
    if (edge >= 0.08 && ev >= 0.1) return buildDecision(row, side, 150, "Underdog underpriced", fairProb, ev, edge);
    if (edge >= 0.05 && ev >= 0.05) return buildDecision(row, side, 100, "Underdog underpriced", fairProb, ev, edge);
    return buildDecision(row, side, 50, "Underdog underpriced", fairProb, ev, edge);
  }

  if (edge >= 0.08 && ev >= 0.1) return buildDecision(row, side, 100, "Favorite underpriced", fairProb, ev, edge);
  if (edge >= 0.05 && ev >= 0.05) return buildDecision(row, side, 50, "Favorite underpriced", fairProb, ev, edge);
  return buildDecision(row, "none", 0, "Price fair");
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
