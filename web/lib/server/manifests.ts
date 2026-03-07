import fs from "node:fs";
import path from "node:path";
import { type LeagueCode } from "@/lib/league";

export type LeagueManifestEntry = {
  code: LeagueCode;
  slug: string;
  default_config_path: string;
  env_config_var: string;
  project_name: string;
  db_path: string;
  championship_name: string;
  championship_probability_key: string;
  uncertainty_policy_name: string;
};

export type LeagueManifestPayload = {
  version: number;
  generated_at_utc: string;
  leagues: Record<LeagueCode, LeagueManifestEntry>;
};

export type ModelManifestPayload = {
  version: number;
  generated_at_utc: string;
  trainable_models: string[];
  aliases: Record<string, string>;
  legacy_model_keys: Record<string, string[]>;
  prediction_report_order: string[];
  display_labels: Record<string, string>;
};

function repoRootPath(): string {
  return path.resolve(process.cwd(), "..");
}

function readGeneratedJson<T>(fileName: string): T {
  const filePath = path.join(repoRootPath(), "configs", "generated", fileName);
  return JSON.parse(fs.readFileSync(filePath, "utf8")) as T;
}

let leagueManifestCache: LeagueManifestPayload | null = null;
let modelManifestCache: ModelManifestPayload | null = null;

export function loadLeagueManifest(): LeagueManifestPayload {
  if (!leagueManifestCache) {
    leagueManifestCache = readGeneratedJson<LeagueManifestPayload>("league_manifest.json");
  }
  return leagueManifestCache;
}

export function loadModelManifest(): ModelManifestPayload {
  if (!modelManifestCache) {
    modelManifestCache = readGeneratedJson<ModelManifestPayload>("model_manifest.json");
  }
  return modelManifestCache;
}

export function getLeagueRuntime(league: LeagueCode): LeagueManifestEntry {
  return loadLeagueManifest().leagues[league];
}

export function resolveConfigPathForLeague(league: LeagueCode): string {
  const runtime = getLeagueRuntime(league);
  const envOverride = process.env[runtime.env_config_var];
  return path.resolve(repoRootPath(), envOverride || runtime.default_config_path);
}

export function resolveDbPathForLeague(league: LeagueCode): string {
  const runtime = getLeagueRuntime(league);
  return path.resolve(repoRootPath(), runtime.db_path);
}

export function getTrainableModels(): string[] {
  return [...loadModelManifest().trainable_models];
}

export function getModelAliases(): Record<string, string> {
  return { ...loadModelManifest().aliases };
}

export { repoRootPath };
