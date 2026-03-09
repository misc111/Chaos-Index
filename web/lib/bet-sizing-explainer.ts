import { REFERENCE_BANKROLL_DOLLARS } from "@/lib/betting";
import { formatUsd } from "@/lib/currency";
import type { BetSizingGamePreview, BetSizingPolicyPreview, BetSizingSlate } from "@/lib/bet-sizing-view";

export type BetSizingScreeningStep = {
  key: "all" | "odds" | "value" | "thresholds" | "profile" | "budget";
  label: string;
  description: string;
  count: number;
};

export type BetSizingAllocationStep = {
  gameId: number;
  matchupLabel: string;
  team: string | null;
  requestedStake: number;
  finalStake: number;
  budgetBefore: number;
  budgetAfter: number;
  allocationRank: number;
  wasTrimmedByBudget: boolean;
  shareOfBudget: number;
  note: string;
};

export type BetSizingExplainerGame = {
  preview: BetSizingGamePreview;
  matchupLabel: string;
  status: "bet" | "pass";
  passLabel: string;
  stopStage: BetSizingScreeningStep["key"];
  requestedStake: number;
  finalStake: number;
  budgetBefore: number | null;
  budgetAfter: number | null;
  allocationRank: number | null;
  shareOfBudget: number;
  wasTrimmedByBudget: boolean;
  baseStakeShareOfBankroll: number | null;
  scaledStakeShareOfBankroll: number | null;
  cappedStakeShareOfBankroll: number | null;
  finalStakeShareOfBankroll: number | null;
  laymanSummary: string;
};

export type BetSizingExplainerModel = {
  headline: string;
  dek: string;
  totalBudget: number;
  maxBetSize: number;
  allocatedBudget: number;
  remainingBudget: number;
  fundedBetCount: number;
  passCount: number;
  requestedBetCount: number;
  trimmedBetCount: number;
  slateLabel: string;
  screening: BetSizingScreeningStep[];
  allocationSteps: BetSizingAllocationStep[];
  games: BetSizingExplainerGame[];
  selectedGame: BetSizingExplainerGame | null;
};

function matchupLabel(preview: BetSizingGamePreview): string {
  return `${preview.row.away_team} at ${preview.row.home_team}`;
}

function compareByAllocationPriority(left: BetSizingGamePreview, right: BetSizingGamePreview): number {
  return (
    (right.trace.candidateExpectedValue ?? Number.NEGATIVE_INFINITY) - (left.trace.candidateExpectedValue ?? Number.NEGATIVE_INFINITY) ||
    (right.trace.candidateEdge ?? Number.NEGATIVE_INFINITY) - (left.trace.candidateEdge ?? Number.NEGATIVE_INFINITY) ||
    right.trace.finalStake - left.trace.finalStake ||
    String(left.trace.decision.team || "").localeCompare(String(right.trace.decision.team || ""))
  );
}

function stopStageForPreview(preview: BetSizingGamePreview): BetSizingScreeningStep["key"] {
  const { gates } = preview.trace;

  if (!gates.oddsAvailable) return "odds";
  if (!gates.positiveExpectedValue) return "value";
  if (!gates.edge || !gates.expectedValue) return "thresholds";
  if (!gates.underdogAllowed) return "profile";
  if (!gates.dailyBudget) return "budget";
  return preview.trace.finalStake > 0 ? "budget" : "thresholds";
}

function passLabelForPreview(preview: BetSizingGamePreview): string {
  const { gates } = preview.trace;

  if (!gates.oddsAvailable) return "Missing odds";
  if (!gates.positiveExpectedValue) return "Price looks fair";
  if (!gates.edge && !gates.expectedValue) return "Edge and EV too small";
  if (!gates.edge) return "Edge too small";
  if (!gates.expectedValue) return "EV too small";
  if (!gates.underdogAllowed) return "Profile skips this dog";
  if (!gates.dailyBudget) return "Budget spent elsewhere";
  return preview.trace.finalStake > 0 ? "Funded bet" : "Pass";
}

function requestedStakeForPreview(preview: BetSizingGamePreview): number {
  return preview.trace.preDailyCapStake ?? preview.trace.finalStake;
}

function laymanSummaryForGame(
  preview: BetSizingGamePreview,
  requestedStake: number,
  totalBudget: number,
  budgetBefore: number | null,
  budgetAfter: number | null,
  allocationRank: number | null
): string {
  const { trace } = preview;
  const team = trace.decision.team || "this side";

  if (trace.finalStake > 0 && trace.dailyRiskCapApplied && requestedStake > trace.finalStake) {
    return `${team} asked for ${formatUsd(requestedStake)}, but only ${formatUsd(trace.finalStake)} remained in the daily budget.`;
  }

  if (trace.finalStake > 0) {
    const share = totalBudget > 0 ? Math.round((trace.finalStake / totalBudget) * 100) : 0;
    const rankText = allocationRank ? ` It ranked #${allocationRank} among today's funded bets.` : "";
    const afterText =
      typeof budgetAfter === "number" ? ` ${formatUsd(budgetAfter)} remains after this bet.` : "";
    return `${team} receives ${formatUsd(trace.finalStake)}, or about ${share}% of today's budget.${rankText}${afterText}`;
  }

  if (!trace.gates.oddsAvailable) {
    return "This game never enters sizing because the app does not have a complete moneyline.";
  }

  if (!trace.gates.positiveExpectedValue) {
    return "After shrinking the model toward market reality, the price does not look cheap enough to warrant any stake.";
  }

  if (!trace.gates.edge || !trace.gates.expectedValue) {
    return "The game has some value signal, but it does not clear the minimum edge and payoff floors needed to spend budget.";
  }

  if (!trace.gates.underdogAllowed) {
    return "The profile blocks this bet because it is an underdog and the conservative rules only allow favorites.";
  }

  if (!trace.gates.dailyBudget && requestedStake > 0) {
    const beforeText = typeof budgetBefore === "number" ? ` Only ${formatUsd(budgetBefore)} was left` : "The daily budget was already used";
    return `${team} would have asked for ${formatUsd(requestedStake)}, but${beforeText.toLowerCase()} by the time this game was considered.`;
  }

  return "This game does not receive any of today's budget.";
}

function bankrollShareFromDollars(amount: number | null | undefined): number | null {
  if (typeof amount !== "number" || !Number.isFinite(amount) || amount <= 0) return null;
  return amount / REFERENCE_BANKROLL_DOLLARS;
}

export function buildBetSizingExplainerModel(
  previews: BetSizingGamePreview[],
  policy: BetSizingPolicyPreview,
  slate: BetSizingSlate,
  selectedGameId: number | null
): BetSizingExplainerModel {
  const totalBudget = (policy.maxDailyBankrollPercent / 100) * REFERENCE_BANKROLL_DOLLARS;
  const maxBetSize = (policy.maxBetBankrollPercent / 100) * REFERENCE_BANKROLL_DOLLARS;

  const screening: BetSizingScreeningStep[] = [
    {
      key: "all",
      label: "On the slate",
      description: "All games under consideration for the active day.",
      count: previews.length,
    },
    {
      key: "odds",
      label: "Have moneylines",
      description: "Only games with a price on both sides can be sized.",
      count: previews.filter((preview) => preview.trace.gates.oddsAvailable).length,
    },
    {
      key: "value",
      label: "Still show value",
      description: "At least one side stays profitable after the model is shrunk toward market reality.",
      count: previews.filter((preview) => preview.trace.gates.positiveExpectedValue).length,
    },
    {
      key: "thresholds",
      label: "Clear the floors",
      description: "The candidate must clear the minimum edge and expected value rules.",
      count: previews.filter((preview) => preview.trace.gates.edge && preview.trace.gates.expectedValue).length,
    },
    {
      key: "profile",
      label: "Fit the profile",
      description: "Risk rules such as underdog restrictions are applied here.",
      count: previews.filter((preview) => preview.trace.gates.underdogAllowed).length,
    },
    {
      key: "budget",
      label: "Receive budget",
      description: "The daily cap funds the best surviving opportunities first.",
      count: previews.filter((preview) => preview.trace.finalStake > 0).length,
    },
  ];

  const allocationCandidates = previews
    .filter((preview) => requestedStakeForPreview(preview) > 0)
    .sort(compareByAllocationPriority);

  let budgetAfterPrevious = totalBudget;
  const allocationByGameId = new Map<number, BetSizingAllocationStep>();
  const allocationSteps: BetSizingAllocationStep[] = allocationCandidates.map((preview, index) => {
    const requestedStake = requestedStakeForPreview(preview);
    const finalStake = preview.trace.finalStake;
    const budgetBefore = budgetAfterPrevious;
    const budgetAfter = Math.max(0, budgetBefore - finalStake);
    budgetAfterPrevious = budgetAfter;

    const step: BetSizingAllocationStep = {
      gameId: preview.row.game_id,
      matchupLabel: matchupLabel(preview),
      team: preview.trace.decision.team,
      requestedStake,
      finalStake,
      budgetBefore,
      budgetAfter,
      allocationRank: index + 1,
      wasTrimmedByBudget: Boolean(preview.trace.dailyRiskCapApplied && requestedStake > finalStake),
      shareOfBudget: totalBudget > 0 ? finalStake / totalBudget : 0,
      note:
        finalStake > 0
          ? `${preview.trace.decision.team || "This side"} receives ${formatUsd(finalStake)}.`
          : `${preview.trace.decision.team || "This side"} asked for ${formatUsd(requestedStake)} but the budget was already exhausted.`,
    };
    allocationByGameId.set(step.gameId, step);
    return step;
  });

  const games = previews.map((preview) => {
    const requestedStake = requestedStakeForPreview(preview);
    const allocation = allocationByGameId.get(preview.row.game_id) || null;

    const game: BetSizingExplainerGame = {
      preview,
      matchupLabel: matchupLabel(preview),
      status: preview.trace.finalStake > 0 ? "bet" : "pass",
      passLabel: passLabelForPreview(preview),
      stopStage: stopStageForPreview(preview),
      requestedStake,
      finalStake: preview.trace.finalStake,
      budgetBefore: allocation?.budgetBefore ?? null,
      budgetAfter: allocation?.budgetAfter ?? null,
      allocationRank: allocation?.allocationRank ?? null,
      shareOfBudget: totalBudget > 0 ? preview.trace.finalStake / totalBudget : 0,
      wasTrimmedByBudget: Boolean(allocation?.wasTrimmedByBudget),
      baseStakeShareOfBankroll: preview.trace.baseStakeShareOfBankroll,
      scaledStakeShareOfBankroll: preview.trace.scaledStakeShareOfBankroll,
      cappedStakeShareOfBankroll: preview.trace.cappedStakeShareOfBankroll,
      finalStakeShareOfBankroll: bankrollShareFromDollars(preview.trace.finalStake),
      laymanSummary: "",
    };

    game.laymanSummary = laymanSummaryForGame(
      preview,
      requestedStake,
      totalBudget,
      game.budgetBefore,
      game.budgetAfter,
      game.allocationRank
    );

    return game;
  });

  const fundedBetCount = games.filter((game) => game.finalStake > 0).length;
  const requestedBetCount = games.filter((game) => game.requestedStake > 0).length;
  const trimmedBetCount = games.filter((game) => game.wasTrimmedByBudget).length;
  const allocatedBudget = games.reduce((sum, game) => sum + game.finalStake, 0);
  const remainingBudget = Math.max(0, totalBudget - allocatedBudget);
  const passCount = games.length - fundedBetCount;
  const selectedGame =
    games.find((game) => game.preview.row.game_id === selectedGameId) ||
    games.find((game) => game.status === "bet") ||
    games[0] ||
    null;

  const headline =
    slate.source === "upcoming"
      ? `How ${formatUsd(totalBudget)} turns into ${fundedBetCount} ${fundedBetCount === 1 ? "bet" : "bets"} today`
      : `How ${formatUsd(totalBudget)} would have been allocated on this replay slate`;
  const dek = `The ${policy.label.toLowerCase()} profile lets each surviving game ask for a stake, caps each bet at ${formatUsd(
    maxBetSize
  )}, then allocates up to ${formatUsd(totalBudget)} across the slate.`;

  return {
    headline,
    dek,
    totalBudget,
    maxBetSize,
    allocatedBudget,
    remainingBudget,
    fundedBetCount,
    passCount,
    requestedBetCount,
    trimmedBetCount,
    slateLabel: slate.label,
    screening,
    allocationSteps,
    games,
    selectedGame,
  };
}
