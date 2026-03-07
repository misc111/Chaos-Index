import fs from "node:fs/promises";
import path from "node:path";
import { type LeagueCode } from "@/lib/league";

export type ModelFeatureMapPayload = {
  path: string;
  updated_at_utc?: string;
  models: Record<string, string[]>;
};

export async function loadServerModelFeatureMap(league: LeagueCode): Promise<ModelFeatureMapPayload> {
  const fileName = `model_feature_map_${league.toLowerCase()}.json`;
  const candidatePath = path.resolve(process.cwd(), "..", "configs", "generated", fileName);
  try {
    const raw = JSON.parse(await fs.readFile(candidatePath, "utf8")) as {
      updated_at_utc?: string;
      models?: Record<string, { active_features?: string[] }>;
    };
    return {
      path: candidatePath,
      updated_at_utc: raw.updated_at_utc,
      models: Object.fromEntries(
        Object.entries(raw.models || {}).map(([modelName, payload]) => [modelName, payload.active_features || []])
      ),
    };
  } catch {
    return {
      path: candidatePath,
      models: {},
    };
  }
}
