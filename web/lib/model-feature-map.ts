import fs from "node:fs/promises";
import path from "node:path";
import { type LeagueCode } from "@/lib/league";

export type ModelFeatureMapPayload = {
  path: string;
  updated_at_utc?: string;
  models: Record<string, string[]>;
};

function stripYamlScalar(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) {
    return "";
  }

  if (
    (trimmed.startsWith("'") && trimmed.endsWith("'")) ||
    (trimmed.startsWith('"') && trimmed.endsWith('"'))
  ) {
    return trimmed.slice(1, -1);
  }

  return trimmed;
}

function featureMapFileName(league: LeagueCode): string {
  return `model_feature_map_${league.toLowerCase()}.yaml`;
}

function featureMapCandidatePaths(league: LeagueCode): string[] {
  const fileName = featureMapFileName(league);
  return [
    path.resolve(process.cwd(), "..", "configs", fileName),
    path.resolve(process.cwd(), "configs", fileName),
  ];
}

async function readFeatureMapFile(league: LeagueCode): Promise<{ path: string; raw: string } | null> {
  for (const candidate of featureMapCandidatePaths(league)) {
    try {
      const raw = await fs.readFile(candidate, "utf8");
      return { path: candidate, raw };
    } catch {
      continue;
    }
  }

  return null;
}

function parseModelFeatureMap(raw: string, filePath: string): ModelFeatureMapPayload {
  const models: Record<string, string[]> = {};
  let updatedAtUtc: string | undefined;
  let inModels = false;
  let currentModel: string | null = null;
  let readingActiveFeatures = false;

  for (const rawLine of raw.split(/\r?\n/)) {
    const line = rawLine.replace(/\t/g, "    ");

    const updatedAtMatch = line.match(/^updated_at_utc:\s*(.+?)\s*$/);
    if (updatedAtMatch) {
      updatedAtUtc = stripYamlScalar(updatedAtMatch[1]);
      continue;
    }

    if (/^models:\s*$/.test(line)) {
      inModels = true;
      currentModel = null;
      readingActiveFeatures = false;
      continue;
    }

    if (!inModels) {
      continue;
    }

    const modelMatch = line.match(/^  ([A-Za-z0-9_]+):\s*$/);
    if (modelMatch) {
      currentModel = modelMatch[1];
      models[currentModel] = [];
      readingActiveFeatures = false;
      continue;
    }

    if (!currentModel) {
      continue;
    }

    if (/^    active_features:\s*$/.test(line)) {
      readingActiveFeatures = true;
      continue;
    }

    if (/^    [A-Za-z0-9_]+:\s*/.test(line)) {
      readingActiveFeatures = false;
      continue;
    }

    if (!readingActiveFeatures) {
      continue;
    }

    const featureMatch = line.match(/^    -\s+(.+?)\s*$/);
    if (featureMatch) {
      const feature = stripYamlScalar(featureMatch[1]);
      if (feature) {
        models[currentModel].push(feature);
      }
    }
  }

  return {
    path: filePath,
    updated_at_utc: updatedAtUtc,
    models,
  };
}

export async function loadModelFeatureMap(league: LeagueCode): Promise<ModelFeatureMapPayload> {
  const loaded = await readFeatureMapFile(league);
  if (!loaded) {
    return {
      path: path.join("configs", featureMapFileName(league)),
      models: {},
    };
  }

  return parseModelFeatureMap(loaded.raw, loaded.path);
}
