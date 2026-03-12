import path from "node:path";
import { ALL_LEAGUES, LEAGUE_RUNTIME, type LeagueCode } from "@/lib/generated/league-registry";
import {
  LEGACY_MODEL_KEYS,
  MODEL_ALIASES,
  MODEL_DISPLAY_LABELS,
  MODEL_REGISTRY,
  MODEL_REPORT_ORDER,
  TRAINABLE_MODELS,
} from "@/lib/generated/model-manifest";

export type LeagueManifestEntry = (typeof LEAGUE_RUNTIME)[LeagueCode];

export type LeagueManifestPayload = {
  version: number;
  source: string;
  leagues: Record<LeagueCode, LeagueManifestEntry>;
};

export type ModelManifestPayload = {
  version: number;
  source: string;
  trainable_models: string[];
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
    trainable_models: [...TRAINABLE_MODELS],
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
 * Return the canonical model alias map shared with Python.
 */
export function getModelAliases(): Record<string, string> {
  return { ...MODEL_ALIASES };
}

export { repoRootPath };
