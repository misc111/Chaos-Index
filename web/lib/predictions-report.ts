const MODEL_REPORT_ORDER = [
  "ensemble",
  "elo_baseline",
  "glm_ridge",
  "dynamic_rating",
  "rf",
  "goals_poisson",
  "gbdt",
  "two_stage",
  "bayes_bt_state_space",
  "bayes_goals",
  "simulation_first",
  "nn_mlp",
] as const;

const MODEL_DISPLAY_LABELS: Record<string, string> = {
  ensemble: "Ensemble",
  elo_baseline: "Elo",
  glm_ridge: "GLM Ridge",
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

const LEGACY_MODEL_ALIASES: Record<string, string> = {
  glm_logit: "glm_ridge",
};

export const MODEL_TRUST_NOTES: Record<string, string> = {
  ensemble: "All models combined. Best default pick. Can share the same blind spot.",
  elo_baseline: "Standard sports betting baseline based on past wins/losses. Good long-run read. Slow on sudden changes.",
  glm_ridge: "Ridge-penalized logistic model. Usually steady. Weird matchups can slip through.",
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

function titleCaseIdentifier(value: string): string {
  return value
    .split("_")
    .filter(Boolean)
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(" ");
}

export function canonicalizePredictionModel(model: string): string {
  const token = String(model || "").trim();
  return LEGACY_MODEL_ALIASES[token] || token;
}

export function displayPredictionModel(model: string): string {
  const canonicalModel = canonicalizePredictionModel(model);
  return MODEL_DISPLAY_LABELS[canonicalModel] || titleCaseIdentifier(canonicalModel);
}

function normalizeLeagueLabel(league?: string | null): string {
  const leagueCode = String(league || "").trim().toUpperCase();
  return leagueCode === "NHL" ? "NHL" : leagueCode === "NBA" ? "NBA" : "";
}

export function predictionTrustNote(model: string, league?: string | null): string {
  const canonicalModel = canonicalizePredictionModel(model);
  const leagueCode = normalizeLeagueLabel(league);

  if (canonicalModel === "glm_ridge" && leagueCode === "NBA") {
    return "Linear pregame model anchored by projected rotation strength, matchup splits, rest, and absence pressure. It is only as good as the lineup view going into tipoff.";
  }

  if (canonicalModel === "glm_ridge" && leagueCode === "NHL") {
    return "Linear pregame model anchored by form, xG share, roster strength, and goalie uncertainty. It is strongest when starter and availability info are current.";
  }

  return (
    MODEL_TRUST_NOTES[canonicalModel] ||
    "Built on that model's own rule set. Good for a second opinion. Watch for large gaps versus the ensemble."
  );
}

export function predictionModelHeadline(model: string, league?: string | null, activeFeatures?: string[]): string | undefined {
  const canonicalModel = canonicalizePredictionModel(model);
  const leagueCode = normalizeLeagueLabel(league);
  const features = Array.isArray(activeFeatures) ? activeFeatures : [];
  const hasDarkoInputs = features.some(
    (feature) => feature.includes("darko_like") || feature.includes("projected_")
  );

  if (canonicalModel === "ensemble") {
    return "Default forecast that blends the live model stack into one probability.";
  }

  if (canonicalModel === "elo_baseline") {
    return "Single-rating baseline built from historical results and home/away context.";
  }

  if (canonicalModel === "glm_ridge" && leagueCode === "NBA") {
    return hasDarkoInputs
      ? "Now using DARKO-like projected rotation inputs before tipoff."
      : "Pregame ridge logistic regression driven by the current NBA feature map.";
  }

  if (canonicalModel === "glm_ridge" && leagueCode === "NHL") {
    return "Pregame ridge logistic regression driven by form, roster, goalie, and xG context.";
  }

  if (canonicalModel === "dynamic_rating") {
    return "Fast-moving strength estimate that reacts more quickly than Elo.";
  }

  if (canonicalModel === "goals_poisson") {
    return leagueCode === "NBA"
      ? "Score-rate model that turns projected offense and defense into a win probability."
      : "Goal-rate model that turns projected scoring into a win probability.";
  }

  if (canonicalModel === "bayes_bt_state_space") {
    return "Bayesian rating layer that tracks team strength with uncertainty over time.";
  }

  if (canonicalModel === "bayes_goals") {
    return "Bayesian scoring model that estimates team strength from expected scoring rates.";
  }

  if (canonicalModel === "simulation_first") {
    return "Scenario simulator that turns repeated matchup draws into a probability estimate.";
  }

  if (features.length > 0) {
    return `${features.length} active inputs from the current ${leagueCode || "league"} feature map.`;
  }

  return undefined;
}

export function orderPredictionModels(models: Iterable<string>): string[] {
  const unique = new Set(
    Array.from(models)
      .map((value) => canonicalizePredictionModel(String(value || "").trim()))
      .filter(Boolean)
  );

  return [
    ...MODEL_REPORT_ORDER.filter((model) => unique.has(model)),
    ...Array.from(unique)
      .filter((model) => !MODEL_REPORT_ORDER.includes(model as (typeof MODEL_REPORT_ORDER)[number]))
      .sort(),
  ];
}

export function canonicalizePredictionModelProbabilities(
  probabilities: Record<string, number | null>
): Record<string, number | null> {
  const out: Record<string, number | null> = {};
  for (const [key, value] of Object.entries(probabilities)) {
    const canonicalKey = canonicalizePredictionModel(key);
    if (!canonicalKey) {
      continue;
    }
    if (!(canonicalKey in out) || out[canonicalKey] == null) {
      out[canonicalKey] = value;
    }
  }
  return out;
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

    return canonicalizePredictionModelProbabilities(
      Object.fromEntries(
        Object.entries(parsed).map(([key, value]) => {
          const numeric = typeof value === "number" ? value : Number(value);
          return [key, Number.isFinite(numeric) ? numeric : null];
        })
      )
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
