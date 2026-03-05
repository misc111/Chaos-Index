import { ENSEMBLE_CORE_MODEL_NAMES, isShadowModel } from "@/lib/model-groups";

const MODEL_REPORT_ORDER = [
  "ensemble",
  ...ENSEMBLE_CORE_MODEL_NAMES,
  "gbdt",
  "rf",
  "two_stage",
  "bayes_bt_state_space",
  "bayes_goals",
  "simulation_first",
  "nn_mlp",
] as const;

const MODEL_DISPLAY_LABELS: Record<string, string> = {
  ensemble: "Ensemble",
  elo_baseline: "Elo",
  glm_logit: "GLM",
  dynamic_rating: "Dyn Rating",
  rf: "RF",
  goals_poisson: "Goals Pois",
  gbdt: "GBDT",
  two_stage: "Two Stage",
  bayes_bt_state_space: "Bayes BT",
  bayes_goals: "Bayes Goals",
  simulation_first: "Sim",
  nn_mlp: "NN",
};

const BASE_MODEL_TRUST_NOTES: Record<string, string> = {
  ensemble: "All models combined. Best default pick. Can share the same blind spot.",
  elo_baseline: "Standard sports betting baseline based on past wins/losses. Good long-run read. Slow on sudden changes.",
  glm_logit: "Statistical model that uses a checklist. Usually steady. Weird matchups can slip through.",
  dynamic_rating: "Hot/cold meter. Good for momentum. Can overreact to short streaks.",
  rf: "Machine learning model that blends many different predictions from random slices of past games. Good at smoothing out flukes. Can be too cautious on close matchups.",
  goals_poisson: "Score-based model. Good for normal scoring games. Messy games hurt it.",
  gbdt: "Machine learning model that finds hidden combos. Sometimes too confident.",
  two_stage:
    "Machine learning model with two steps: first predicts game type (fast/slow, close/lopsided), then predicts winner. Good when style matchups matter. If step 1 is wrong, final pick can be wrong.",
  bayes_bt_state_space:
    "Tracks team strength after every game and gives a range, not just one number. Good for spotting rising/falling teams with uncertainty shown. Can move fast after injuries, trades, or short weird stretches.",
  bayes_goals: "Scoring strength + confidence meter. Good trend read. Can lag sudden lineup changes.",
  simulation_first:
    "Runs the matchup thousands of times using set assumptions (team strength, pace, and scoring). Good for seeing different paths. If those assumptions are off, this number can be off.",
  nn_mlp: "Machine learning model that finds subtle patterns. Hardest to explain.",
};

export const MODEL_TRUST_NOTES: Record<string, string> = Object.fromEntries(
  Object.entries(BASE_MODEL_TRUST_NOTES).map(([model, note]) => [
    model,
    isShadowModel(model) ? `Tracked separately. Not used in the ensemble. ${note}` : note,
  ])
);

function titleCaseIdentifier(value: string): string {
  return value
    .split("_")
    .filter(Boolean)
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(" ");
}

export function displayPredictionModel(model: string): string {
  return MODEL_DISPLAY_LABELS[model] || titleCaseIdentifier(model);
}

export function predictionTrustNote(model: string): string {
  return (
    MODEL_TRUST_NOTES[model] ||
    (isShadowModel(model)
      ? "Tracked separately. Not used in the ensemble. Built on that model's own rule set. Watch for large gaps versus the ensemble."
      : "Built on that model's own rule set. Good for a second opinion. Watch for large gaps versus the ensemble.")
  );
}

export function orderPredictionModels(models: Iterable<string>): string[] {
  const unique = new Set(
    Array.from(models)
      .map((value) => String(value || "").trim())
      .filter(Boolean)
  );

  return [
    ...MODEL_REPORT_ORDER.filter((model) => unique.has(model)),
    ...Array.from(unique)
      .filter((model) => !MODEL_REPORT_ORDER.includes(model as (typeof MODEL_REPORT_ORDER)[number]))
      .sort(),
  ];
}

export function parseModelWinProbabilities(raw: unknown): Record<string, number | null> {
  if (typeof raw !== "string" || !raw.trim()) {
    return {};
  }

  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }

    return Object.fromEntries(
      Object.entries(parsed).map(([key, value]) => {
        const numeric = typeof value === "number" ? value : Number(value);
        return [key, Number.isFinite(numeric) ? numeric : null];
      })
    );
  } catch {
    return {};
  }
}

export function formatPredictionProbability(value?: number | null): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "-";
  }
  return `${(value * 100).toFixed(1)}%`;
}

export function formatPredictionDate(value?: string | null): string {
  const raw = String(value || "").trim();
  if (!raw) {
    return "-";
  }

  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) {
    return raw.slice(0, 10);
  }

  return parsed.toISOString().slice(0, 10);
}

export function formatPredictionAsOf(value?: string | null): string {
  const raw = String(value || "").trim();
  if (!raw) {
    return "Unknown";
  }

  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) {
    return raw;
  }

  return parsed.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  });
}
