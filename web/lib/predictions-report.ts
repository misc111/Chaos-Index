import {
  MODEL_ALIASES,
  MODEL_DISPLAY_LABELS,
  MODEL_REGISTRY,
  MODEL_REPORT_ORDER,
} from "@/lib/generated/model-manifest";

export const MODEL_TRUST_NOTES: Record<string, string> = {
  ensemble: "All models combined. Best default pick. Can share the same blind spot.",
  glm_ridge: "Ridge-penalized logistic model. Usually steady. Weird matchups can slip through.",
  glm_lasso:
    "Lasso-penalized logistic model. Good for pruning weak or redundant inputs. Can zero out small but real shared effects.",
  glm_elastic_net:
    "Elastic-net logistic model. Good when related signals travel in packs. Can still mute smaller edges if the penalty is too strong.",
  glm_vanilla: "Unpenalized logistic model. Good for checking whether regularization is washing out real signal. Most likely to overfit.",
  gam_spline: "Spline-based logistic model. Good at smooth nonlinear edges. Can get wobbly if the shape fit is too ambitious.",
  glmm_logit: "Mixed-effects logistic model. Good when team-level structure matters. Can be slower and harder to keep stable.",
  dglm_margin: "Margin-first model that turns score-shape estimates into win probabilities. Good when spread shape matters. Can drift if score variance is misspecified.",
  goals_poisson: "Score-based model. Good for normal scoring games. Messy games hurt it.",
};

function titleCaseIdentifier(value: string): string {
  return value
    .split("_")
    .filter(Boolean)
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(" ");
}

export function canonicalizePredictionModel(model: string): string {
  const token = String(model || "").trim().toLowerCase();
  return MODEL_ALIASES[token] || token;
}

export function displayPredictionModel(model: string): string {
  const canonicalModel = canonicalizePredictionModel(model);
  return MODEL_DISPLAY_LABELS[canonicalModel] || titleCaseIdentifier(canonicalModel);
}

function normalizeLeagueLabel(league?: string | null): string {
  const leagueCode = String(league || "").trim().toUpperCase();
  return leagueCode === "MLB" ? "MLB" : "";
}

function isExperimentalPredictionModel(model: string): boolean {
  const registryEntry = (MODEL_REGISTRY as Record<string, { lane?: string }>)[model];
  return registryEntry?.lane === "experimental";
}

export function predictionTrustNote(model: string, league?: string | null): string {
  const canonicalModel = canonicalizePredictionModel(model);
  const leagueCode = normalizeLeagueLabel(league);

  if (canonicalModel === "glm_ridge" && leagueCode === "MLB") {
    return "Linear pregame model anchored by starting pitching, bullpen quality, lineup strength, park effects, and rest-travel context. It is only as good as the pregame starter and lineup view.";
  }

  if (canonicalModel === "glm_elastic_net" && leagueCode === "MLB") {
    return "Elastic-net pregame model using the MLB feature map with extra shrinkage on overlapping starter, bullpen, lineup, park, and weather signals.";
  }

  if (canonicalModel === "glm_lasso" && leagueCode === "MLB") {
    return "Lasso pregame model using the MLB feature map with stronger pruning on overlapping starter, bullpen, lineup, park, and weather signals.";
  }

  if (canonicalModel === "glm_vanilla") {
    return "Unpenalized logistic challenger. Useful when you want to know whether shrinkage is suppressing a real betting edge.";
  }

  if (canonicalModel === "gam_spline") {
    return "Spline-based challenger that lets a few continuous signals bend instead of forcing everything to stay linear.";
  }

  if (canonicalModel === "glmm_logit") {
    return "Mixed-effects challenger that keeps fixed matchup signals while allowing team-level structure to matter.";
  }

  if (canonicalModel === "dglm_margin") {
    return "Margin-shape challenger that models score expectation and variance before converting that into a win probability.";
  }

  if (isExperimentalPredictionModel(canonicalModel)) {
    return "Experimental challenger retained for comparison only. It is not part of the default MLB theory lane unless explicitly requested.";
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

  if (canonicalModel === "ensemble") {
    return "Default forecast that blends the live model stack into one probability.";
  }

  if (canonicalModel === "glm_ridge" && leagueCode === "MLB") {
    return "Pregame ridge logistic regression driven by starting pitcher, bullpen, lineup, park, weather, and scheduling context.";
  }

  if (canonicalModel === "glm_elastic_net" && leagueCode === "MLB") {
    return "Pregame elastic-net logistic regression driven by starting pitcher, bullpen, lineup, park, weather, and schedule context.";
  }

  if (canonicalModel === "glm_lasso" && leagueCode === "MLB") {
    return "Pregame lasso logistic regression driven by starting pitcher, bullpen, lineup, park, weather, and schedule context.";
  }

  if (canonicalModel === "glm_vanilla") {
    return "Unpenalized logistic challenger built to test whether shrinkage is muting profit signal.";
  }

  if (canonicalModel === "gam_spline") {
    return "Spline-based challenger that lets a small nonlinear feature block bend away from a straight-line fit.";
  }

  if (canonicalModel === "glmm_logit") {
    return "Mixed-effects challenger that layers team-level structure on top of fixed pregame features.";
  }

  if (canonicalModel === "dglm_margin") {
    return "Two-step margin challenger that estimates both expected spread and spread uncertainty before deriving win odds.";
  }

  if (canonicalModel === "goals_poisson") {
    return "Run-rate model that turns projected scoring and run prevention into a win probability.";
  }

  if (isExperimentalPredictionModel(canonicalModel)) {
    return "Experimental challenger. Not part of the default MLB theory lane unless explicitly requested.";
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
