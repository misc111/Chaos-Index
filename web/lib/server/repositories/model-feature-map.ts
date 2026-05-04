import { existsSync } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import { MODEL_LANE_LABELS, MODEL_REGISTRY } from "@/lib/generated/model-manifest";
import { type LeagueCode } from "@/lib/league";
import { repoRootPath } from "@/lib/server/manifests";

export type ModelFeatureMapPayload = {
  path: string;
  source?: "latest_fit_artifacts" | "generated_registry" | "missing";
  updated_at_utc?: string;
  model_run_id?: string;
  feature_set_version?: string;
  models: Record<string, ModelFeatureDiagnostics>;
};

export type ModelFeatureTheoryTrace = {
  lane?: string;
  lane_label?: string;
  classification?: string;
  governance?: "direct theory implementation" | "theory-compatible engineering support" | "betting overlay";
  note?: string;
};

export type ModelCoefficientDiagnostic = {
  feature: string;
  active_rank?: number | null;
  coef_scaled?: number | null;
  coef_original?: number | null;
  abs_coef_scaled?: number | null;
  direction?: string | null;
  odds_multiplier_1sd?: number | null;
};

export type ModelFeatureDiagnostics = {
  active_features: string[];
  active_feature_count: number;
  fit_active_features: string[];
  fit_active_feature_count: number;
  candidate_feature_count?: number;
  coefficient_top_features: string[];
  coefficient_summary: ModelCoefficientDiagnostic[];
  coefficient_totals?: Record<string, unknown> | null;
  penalty_family?: string | null;
  distribution?: string | null;
  link_function?: string | null;
  complement_kind?: string | null;
  complement_column?: string | null;
  complement_label?: string | null;
  artifact_path?: string | null;
  fit_metadata_path?: string | null;
  theory_trace?: ModelFeatureTheoryTrace;
};

type LoadArtifactOptions = {
  repoRoot?: string;
};

type JsonRecord = Record<string, unknown>;

function defaultArtifactRepoRoot(): string {
  const cwd = process.cwd();
  if (existsSync(path.join(cwd, "artifacts", "models"))) {
    return cwd;
  }
  const parent = path.resolve(cwd, "..");
  if (existsSync(path.join(parent, "artifacts", "models"))) {
    return parent;
  }
  const manifestRoot = repoRootPath();
  if (existsSync(path.join(manifestRoot, "artifacts", "models"))) {
    return manifestRoot;
  }
  return manifestRoot;
}

function isRecord(value: unknown): value is JsonRecord {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function cleanStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return Array.from(new Set(value.map((entry) => String(entry || "").trim()).filter(Boolean)));
}

function numberOrNull(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function stringOrNull(value: unknown): string | null {
  const text = String(value || "").trim();
  return text || null;
}

function theoryGovernanceForLane(lane?: string | null): ModelFeatureTheoryTrace["governance"] {
  if (lane === "core") return "direct theory implementation";
  if (lane === "experimental") return "betting overlay";
  return "theory-compatible engineering support";
}

function theoryTraceForModel(modelName: string): ModelFeatureTheoryTrace | undefined {
  const registry = MODEL_REGISTRY as Record<string, JsonRecord | undefined>;
  const entry = registry[modelName];
  if (!entry) return undefined;
  const lane = stringOrNull(entry.lane);
  return {
    lane: lane || undefined,
    lane_label: lane ? MODEL_LANE_LABELS[lane] : undefined,
    classification: stringOrNull(entry.theory_classification) || undefined,
    governance: theoryGovernanceForLane(lane),
    note: stringOrNull(entry.governance_note) || undefined,
  };
}

function normalizeCoefficientSummary(value: unknown): ModelCoefficientDiagnostic[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter(isRecord)
    .map((row) => ({
      feature: String(row.feature || "").trim(),
      active_rank: numberOrNull(row.active_rank),
      coef_scaled: numberOrNull(row.coef_scaled),
      coef_original: numberOrNull(row.coef_original),
      abs_coef_scaled: numberOrNull(row.abs_coef_scaled),
      direction: stringOrNull(row.direction),
      odds_multiplier_1sd: numberOrNull(row.odds_multiplier_1sd),
    }))
    .filter((row) => row.feature);
}

async function readJsonRecord(filePath: string): Promise<JsonRecord | null> {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    const sanitized = raw
      .replace(/(^|[\s:[,])-?Infinity(?=\s*[,}\]])/g, "$1null")
      .replace(/(^|[\s:[,])NaN(?=\s*[,}\]])/g, "$1null");
    const parsed = JSON.parse(sanitized) as unknown;
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

async function listRunPayloadPaths(root: string): Promise<string[]> {
  return listFilesMatching(root, "run_payload.json", path.join(root, "artifacts", "models"));
}

async function listFilesMatching(root: string, fileName: string, startDir = path.join(root, "artifacts", "models")): Promise<string[]> {
  async function walk(dir: string): Promise<string[]> {
    let entries: Awaited<ReturnType<typeof fs.readdir>>;
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch {
      return [];
    }

    const out: string[] = [];
    for (const entry of entries) {
      const entryPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        out.push(...(await walk(entryPath)));
      } else if (entry.isFile() && (!fileName || entry.name === fileName)) {
        out.push(entryPath);
      }
    }
    return out;
  }

  return walk(startDir);
}

async function latestRunPayloadPath(root: string, league: LeagueCode): Promise<string | null> {
  const candidates = await listRunPayloadPaths(root);
  const scored: Array<{ filePath: string; mtimeMs: number }> = [];
  for (const filePath of candidates) {
    const payload = await readJsonRecord(filePath);
    const payloadLeague = String(payload?.league || league).trim().toUpperCase();
    if (payloadLeague !== league) continue;
    const stat = await fs.stat(filePath);
    scored.push({ filePath, mtimeMs: stat.mtimeMs });
  }
  scored.sort((left, right) => right.mtimeMs - left.mtimeMs || right.filePath.localeCompare(left.filePath));
  return scored[0]?.filePath || null;
}

function recordStringArrayMap(value: unknown): Record<string, string[]> {
  if (!isRecord(value)) return {};
  return Object.fromEntries(Object.entries(value).map(([key, raw]) => [key, cleanStringArray(raw)]));
}

function fitSummaryFromArtifact(artifact: JsonRecord): JsonRecord {
  return isRecord(artifact.fit_summary) ? artifact.fit_summary : {};
}

function buildDiagnosticsFromArtifact(args: {
  root: string;
  modelName: string;
  modelRunDir: string;
  modelFeatureColumns: Record<string, string[]>;
  artifact: JsonRecord;
}): ModelFeatureDiagnostics {
  const { root, modelName, modelRunDir, modelFeatureColumns, artifact } = args;
  const fitSummary = fitSummaryFromArtifact(artifact);
  const coefficientMetadata = isRecord(fitSummary.coefficient_path_metadata) ? fitSummary.coefficient_path_metadata : {};
  const activeFeatures =
    cleanStringArray(artifact.feature_columns).length > 0
      ? cleanStringArray(artifact.feature_columns)
      : modelFeatureColumns[modelName] || cleanStringArray(fitSummary.active_features);
  const fitActiveFeatures = cleanStringArray(fitSummary.active_features);
  const artifactFiles = isRecord(artifact.artifact_files) ? artifact.artifact_files : {};
  const fitMetadataFile = stringOrNull(artifactFiles.fit_metadata);

  return {
    active_features: activeFeatures,
    active_feature_count: activeFeatures.length,
    fit_active_features: fitActiveFeatures,
    fit_active_feature_count: fitActiveFeatures.length,
    candidate_feature_count:
      Number(fitSummary.feature_count || artifact.feature_count || coefficientMetadata.n_features_total) || activeFeatures.length,
    coefficient_top_features: cleanStringArray(coefficientMetadata.top_feature_names),
    coefficient_summary: normalizeCoefficientSummary(fitSummary.active_coefficient_summary),
    coefficient_totals: isRecord(fitSummary.active_coefficient_totals) ? fitSummary.active_coefficient_totals : null,
    penalty_family: stringOrNull(fitSummary.penalty_family || artifact.penalty_family),
    distribution: stringOrNull(artifact.distribution),
    link_function: stringOrNull(artifact.link_function),
    complement_kind: stringOrNull(artifact.complement_kind),
    complement_column: stringOrNull(artifact.complement_column),
    complement_label: stringOrNull(artifact.complement_label),
    artifact_path: path.relative(root, modelRunDir),
    fit_metadata_path: fitMetadataFile ? path.relative(root, path.join(modelRunDir, fitMetadataFile)) : null,
    theory_trace: theoryTraceForModel(modelName),
  };
}

function modelNameFromCredibilityMetadataPath(filePath: string): string {
  return path.basename(filePath).replace(/_credibility_metadata\.json$/, "");
}

function buildDiagnosticsFromCredibilityMetadata(args: {
  root: string;
  metadataPath: string;
  modelName: string;
  metadata: JsonRecord;
}): ModelFeatureDiagnostics {
  const { root, metadataPath, modelName, metadata } = args;
  const relativitySummary = isRecord(metadata.relativity_summary) ? metadata.relativity_summary : {};
  const coefficientSummary = normalizeCoefficientSummary(relativitySummary.active_coefficient_summary);
  const coefficientMetadata = isRecord(relativitySummary.coefficient_path_metadata)
    ? relativitySummary.coefficient_path_metadata
    : {};
  const fitActiveFeatures = coefficientSummary.map((row) => row.feature);
  const activeFeatures = cleanStringArray(metadata.driver_features);
  return {
    active_features: activeFeatures,
    active_feature_count: activeFeatures.length,
    fit_active_features: fitActiveFeatures,
    fit_active_feature_count: Number(relativitySummary.active_driver_feature_count) || fitActiveFeatures.length,
    candidate_feature_count: Number(relativitySummary.driver_feature_count) || activeFeatures.length,
    coefficient_top_features: cleanStringArray(coefficientMetadata.top_feature_names),
    coefficient_summary: coefficientSummary,
    coefficient_totals: null,
    penalty_family: "lasso_credibility",
    distribution: "binomial",
    link_function: "logit",
    complement_kind: stringOrNull(metadata.complement_kind),
    complement_column: stringOrNull(metadata.complement_column),
    complement_label: stringOrNull(metadata.complement_label),
    artifact_path: path.relative(root, path.dirname(metadataPath)),
    fit_metadata_path: path.relative(root, metadataPath),
    theory_trace: theoryTraceForModel(modelName),
  };
}

async function loadCredibilitySidecarDiagnostics(root: string): Promise<Record<string, ModelFeatureDiagnostics>> {
  const candidates = await listFilesMatching(root, "", path.join(root, "artifacts", "models"));
  const sidecars = candidates.filter((filePath) => filePath.endsWith("_credibility_metadata.json"));
  const latestByModel = new Map<string, { filePath: string; mtimeMs: number }>();

  for (const filePath of sidecars) {
    const modelName = modelNameFromCredibilityMetadataPath(filePath);
    const stat = await fs.stat(filePath);
    const existing = latestByModel.get(modelName);
    if (!existing || stat.mtimeMs > existing.mtimeMs) {
      latestByModel.set(modelName, { filePath, mtimeMs: stat.mtimeMs });
    }
  }

  const entries: Array<[string, ModelFeatureDiagnostics]> = [];
  for (const [modelName, { filePath }] of latestByModel.entries()) {
    const metadata = await readJsonRecord(filePath);
    if (!metadata) continue;
    entries.push([
      modelName,
      buildDiagnosticsFromCredibilityMetadata({
        root,
        metadataPath: filePath,
        modelName,
        metadata,
      }),
    ]);
  }
  return Object.fromEntries(entries);
}

export async function loadModelFeatureMapFromArtifacts(
  league: LeagueCode,
  options: LoadArtifactOptions = {}
): Promise<ModelFeatureMapPayload> {
  const root = options.repoRoot || defaultArtifactRepoRoot();
  const runPayloadPath = await latestRunPayloadPath(root, league);
  const credibilityModels = await loadCredibilitySidecarDiagnostics(root);
  if (!runPayloadPath) {
    return {
      path: path.join(root, "artifacts", "models"),
      source: Object.keys(credibilityModels).length > 0 ? "latest_fit_artifacts" : "missing",
      models: credibilityModels,
    };
  }

  const runPayload = (await readJsonRecord(runPayloadPath)) || {};
  const modelFeatureColumns = recordStringArrayMap(runPayload.model_feature_columns);
  const modelArtifacts = Array.isArray(runPayload.model_artifacts) ? runPayload.model_artifacts.filter(isRecord) : [];
  const modelRunDir = path.dirname(runPayloadPath);
  const models = {
    ...Object.fromEntries(
      modelArtifacts
        .map((artifact) => [String(artifact.model_name || "").trim(), artifact] as const)
        .filter(([modelName]) => modelName)
        .map(([modelName, artifact]) => [
          modelName,
          buildDiagnosticsFromArtifact({
            root,
            modelName,
            modelRunDir,
            modelFeatureColumns,
            artifact,
          }),
        ])
    ),
    ...credibilityModels,
  };

  return {
    path: runPayloadPath,
    source: "latest_fit_artifacts",
    updated_at_utc: new Date((await fs.stat(runPayloadPath)).mtimeMs).toISOString(),
    model_run_id: stringOrNull(runPayload.model_run_id) || undefined,
    feature_set_version: stringOrNull(runPayload.feature_set_version) || undefined,
    models,
  };
}

function diagnosticsFromGeneratedRegistry(modelName: string, payload: unknown): ModelFeatureDiagnostics {
  const record = isRecord(payload) ? payload : {};
  const activeFeatures = cleanStringArray(record.active_features);
  return {
    active_features: activeFeatures,
    active_feature_count: activeFeatures.length,
    fit_active_features: activeFeatures,
    fit_active_feature_count: activeFeatures.length,
    candidate_feature_count: Number(record.feature_count) || activeFeatures.length,
    coefficient_top_features: [],
    coefficient_summary: [],
    coefficient_totals: null,
    theory_trace: theoryTraceForModel(modelName),
  };
}

async function loadGeneratedModelFeatureMap(league: LeagueCode): Promise<ModelFeatureMapPayload> {
  const fileName = `model_feature_map_${league.toLowerCase()}.json`;
  const candidatePath = path.resolve(defaultArtifactRepoRoot(), "configs", "generated", fileName);
  const raw = await readJsonRecord(candidatePath);
  if (!raw) {
    return {
      path: candidatePath,
      source: "missing",
      models: {},
    };
  }
  const models = isRecord(raw.models) ? raw.models : {};
  return {
    path: candidatePath,
    source: "generated_registry",
    updated_at_utc: stringOrNull(raw.updated_at_utc) || undefined,
    models: Object.fromEntries(
      Object.entries(models).map(([modelName, payload]) => [modelName, diagnosticsFromGeneratedRegistry(modelName, payload)])
    ),
  };
}

export async function loadServerModelFeatureMap(league: LeagueCode): Promise<ModelFeatureMapPayload> {
  const artifacts = await loadModelFeatureMapFromArtifacts(league);
  if (Object.keys(artifacts.models).length > 0) {
    return artifacts;
  }
  return loadGeneratedModelFeatureMap(league);
}
