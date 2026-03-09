export type ModelWinProbabilities = Record<string, number | null>;

function numericOrNull(value: unknown): number | null {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

export function parseModelWinProbabilities(value?: string | null): ModelWinProbabilities {
  if (!value) return {};

  try {
    const parsed = JSON.parse(value) as Record<string, unknown>;
    return Object.fromEntries(
      Object.entries(parsed).map(([key, probability]) => [key, numericOrNull(probability)])
    );
  } catch {
    return {};
  }
}

export function selectBettingModelProbability(
  ensembleHomeWinProbability: number,
  modelWinProbabilities?: ModelWinProbabilities | null,
  preferredModelName?: string | null
): {
  betting_model_name: string;
  home_win_probability: number;
  model_win_probabilities: ModelWinProbabilities;
} {
  const combinedProbabilities: ModelWinProbabilities = {
    ensemble: numericOrNull(ensembleHomeWinProbability),
    ...(modelWinProbabilities || {}),
  };

  const resolvedModelName =
    preferredModelName && typeof combinedProbabilities[preferredModelName] === "number" ? preferredModelName : "ensemble";

  const selectedProbability = combinedProbabilities[resolvedModelName];
  return {
    betting_model_name: resolvedModelName,
    home_win_probability:
      typeof selectedProbability === "number" ? selectedProbability : numericOrNull(ensembleHomeWinProbability) ?? 0.5,
    model_win_probabilities: combinedProbabilities,
  };
}
