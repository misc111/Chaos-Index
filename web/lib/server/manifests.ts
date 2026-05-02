import path from "node:path";
import {
  ALL_LEAGUES,
  LEAGUE_RUNTIME,
  LEGACY_COMPARISON_LEAGUES,
  PRIMARY_LEAGUE,
  PRIMARY_REBUILD_LEAGUES,
  type LeagueCode,
} from "@/lib/generated/league-registry";
import {
    BASELINE_MODEL_KEYS,
    CORE_MODEL_KEYS,
    DEFAULT_TRAINING_MODELS,
    EXPERIMENTAL_MODEL_KEYS,
    LEGACY_MODEL_KEYS,
    MODEL_ALIASES,
    MODEL_DISPLAY_LABELS,
    MODEL_LANE_LABELS,
    MODEL_LANE_NOTES,
    MODEL_REGISTRY,
    MODEL_REPORT_ORDER,
    PRIMARY_MODEL_LANE,
    THEORY_EXTENSION_MODEL_KEYS,
    TRAINABLE_MODELS,
} from "@/lib/generated/model-manifest";

export type LeagueManifestEntry = (typeof LEAGUE_RUNTIME)[LeagueCode];

export type LeagueManifestPayload = {
  version: number;
  source: string;
  primary_league: LeagueCode;
  primary_rebuild_leagues: LeagueCode[];
  legacy_comparison_leagues: typeof LEGACY_COMPARISON_LEAGUES;
  leagues: Record<LeagueCode, LeagueManifestEntry>;
};

export type ModelManifestPayload = {
  version: number;
  source: string;
  primary_lane: string;
  trainable_models: string[];
  default_training_models: string[];
  core_models: string[];
  theory_extension_models: string[];
  baseline_models: string[];
  experimental_models: string[];
  lane_labels: Record<string, string>;
  lane_notes: Record<string, string>;
  aliases: Record<string, string>;
  legacy_model_keys: Record<string, string[]>;
  prediction_report_order: string[];
  display_labels: Record<string, string>;
  models: Record<string, typeof MODEL_REGISTRY[keyof typeof MODEL_REGISTRY]>;
};

function repoRootPath(): string {
  return path.resolve(process.cwd(), "..");
}

/**
 * Return the generated league manifest payload backed by the code registry.
 */
export function loadLeagueManifest(): LeagueManifestPayload {
  return {
    version: 1,
    source: "code_registry",
    primary_league: PRIMARY_LEAGUE,
    primary_rebuild_leagues: [...PRIMARY_REBUILD_LEAGUES],
    legacy_comparison_leagues: LEGACY_COMPARISON_LEAGUES,
    leagues: Object.fromEntries(ALL_LEAGUES.map((league) => [league, LEAGUE_RUNTIME[league]])) as Record<
      LeagueCode,
      LeagueManifestEntry
    >,
  };
}

/**
 * Return the generated model manifest payload backed by the code registry.
 */
export function loadModelManifest(): ModelManifestPayload {
  return {
    version: 1,
    source: "code_registry",
    primary_lane: PRIMARY_MODEL_LANE,
    trainable_models: [...TRAINABLE_MODELS],
    default_training_models: [...DEFAULT_TRAINING_MODELS],
    core_models: [...CORE_MODEL_KEYS],
    theory_extension_models: [...THEORY_EXTENSION_MODEL_KEYS],
    baseline_models: [...BASELINE_MODEL_KEYS],
    experimental_models: [...EXPERIMENTAL_MODEL_KEYS],
    lane_labels: { ...MODEL_LANE_LABELS },
    lane_notes: { ...MODEL_LANE_NOTES },
    aliases: { ...MODEL_ALIASES },
    legacy_model_keys: Object.fromEntries(
      Object.entries(LEGACY_MODEL_KEYS).map(([model, aliases]) => [model, [...aliases]])
    ),
    prediction_report_order: [...MODEL_REPORT_ORDER],
    display_labels: { ...MODEL_DISPLAY_LABELS },
    models: { ...MODEL_REGISTRY },
  };
}

/**
 * Resolve canonical runtime metadata for a league code.
 */
export function getLeagueRuntime(league: LeagueCode): LeagueManifestEntry {
  return LEAGUE_RUNTIME[league];
}

/**
 * Resolve the config path for a league, honoring registry-declared env overrides.
 */
export function resolveConfigPathForLeague(league: LeagueCode): string {
  const runtime = getLeagueRuntime(league);
  const envOverride = process.env[runtime.configEnvVar];
  return path.resolve(repoRootPath(), envOverride || runtime.defaultConfigPath);
}

/**
 * Resolve the DB path for a league from generated runtime metadata.
 */
export function resolveDbPathForLeague(league: LeagueCode): string {
  const runtime = getLeagueRuntime(league);
  return path.resolve(repoRootPath(), runtime.dbPath);
}

/**
 * Return the canonical trainable-model list shared with Python.
 */
export function getTrainableModels(): string[] {
  return [...TRAINABLE_MODELS];
}

/**
 * Return the canonical core-model list shared with Python.
 */
export function getCoreModels(): string[] {
  return [...CORE_MODEL_KEYS];
}

/**
 * Return the canonical default training models for the MLB primary lane.
 */
export function getDefaultTrainingModels(): string[] {
  return [...DEFAULT_TRAINING_MODELS];
}

/**
 * Return the canonical model alias map shared with Python.
 */
export function getModelAliases(): Record<string, string> {
  return { ...MODEL_ALIASES };
}

export { repoRootPath };
