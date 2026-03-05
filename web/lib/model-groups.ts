export const ENSEMBLE_CORE_MODEL_NAMES = [
  "elo_baseline",
  "dynamic_rating",
  "glm_logit",
  "goals_poisson",
] as const;

const ENSEMBLE_CORE_MODEL_SET = new Set<string>(ENSEMBLE_CORE_MODEL_NAMES);

export function isEnsembleCoreModel(model: string): boolean {
  return ENSEMBLE_CORE_MODEL_SET.has(String(model || "").trim());
}

export function isShadowModel(model: string): boolean {
  const normalized = String(model || "").trim();
  return Boolean(normalized) && normalized !== "ensemble" && !isEnsembleCoreModel(normalized);
}

export function partitionModels(models: Iterable<string>): {
  primaryModels: string[];
  shadowModels: string[];
} {
  const unique = Array.from(
    new Set(
      Array.from(models)
        .map((value) => String(value || "").trim())
        .filter(Boolean)
    )
  );

  return {
    primaryModels: unique.filter((model) => !isShadowModel(model)),
    shadowModels: unique.filter((model) => isShadowModel(model)),
  };
}

export function partitionModelRows<T extends Record<string, unknown>>(
  rows: T[],
  modelKey: keyof T = "model_name" as keyof T
): {
  primaryRows: T[];
  shadowRows: T[];
  otherRows: T[];
} {
  const primaryRows: T[] = [];
  const shadowRows: T[] = [];
  const otherRows: T[] = [];

  for (const row of rows) {
    const rawModel = row[modelKey];
    if (typeof rawModel !== "string" || !rawModel.trim()) {
      otherRows.push(row);
      continue;
    }

    if (isShadowModel(rawModel)) {
      shadowRows.push(row);
      continue;
    }

    primaryRows.push(row);
  }

  return { primaryRows, shadowRows, otherRows };
}
