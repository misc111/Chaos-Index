import { NextResponse } from "next/server";
import { leagueFromRequest } from "@/lib/league";
import fs from "node:fs";
import path from "node:path";

export const dynamic = "force-dynamic";

type SectionSpec = {
  section: string;
  file_name: string;
  kind: "csv" | "json";
  tail_rows?: number | null;
};

type ParsedCsvRow = Record<string, number | string>;
type ParsedJsonRow = Record<string, unknown>;

function parseCsv(filePath: string): ParsedCsvRow[] {
  if (!fs.existsSync(filePath)) return [];
  const text = fs.readFileSync(filePath, "utf8").trim();
  if (!text) return [];
  const [header, ...lines] = text.split(/\r?\n/);
  const cols = header.split(",");
  return lines.map((line) => {
    const vals = line.split(",");
    const obj: ParsedCsvRow = {};
    cols.forEach((c, i) => {
      const v = vals[i] ?? "";
      const n = Number(v);
      obj[c] = Number.isFinite(n) && v !== "" ? n : v;
    });
    return obj;
  });
}

function parseJson(filePath: string): ParsedJsonRow[] {
  if (!fs.existsSync(filePath)) return [];
  const raw = fs.readFileSync(filePath, "utf8");
  try {
    return [JSON.parse(raw) as ParsedJsonRow];
  } catch {
    const sanitized = raw.replace(/\bNaN\b/g, "null").replace(/\bInfinity\b/g, "null").replace(/\b-Infinity\b/g, "null");
    try {
      return [JSON.parse(sanitized) as ParsedJsonRow];
    } catch {
      return [];
    }
  }
}

const DEFAULT_SECTION_SPECS: SectionSpec[] = [
  { section: "split_summary", file_name: "validation_split_summary.json", kind: "json" },
  { section: "glm_residual_summary", file_name: "validation_glm_residual_summary.json", kind: "json" },
  { section: "glm_residual_feature_summary", file_name: "validation_glm_residual_feature_summary.csv", kind: "csv" },
  { section: "glm_working_residual_bins_linear_predictor", file_name: "validation_glm_working_residual_bins_linear_predictor.csv", kind: "csv" },
  { section: "glm_working_residual_bins_features", file_name: "validation_glm_working_residual_bins_features.csv", kind: "csv" },
  { section: "glm_working_residual_bins_weight", file_name: "validation_glm_working_residual_bins_weight.csv", kind: "csv" },
  { section: "glm_partial_residual_bins", file_name: "validation_glm_partial_residual_bins.csv", kind: "csv" },
  { section: "significance", file_name: "validation_significance.csv", kind: "csv" },
  { section: "information_criteria_summary", file_name: "validation_information_criteria_summary.json", kind: "json" },
  { section: "information_criteria_candidates", file_name: "validation_information_criteria_candidates.csv", kind: "csv" },
  { section: "coef_paths", file_name: "validation_coef_paths.csv", kind: "csv", tail_rows: 200 },
  { section: "cv_summary", file_name: "validation_cv_summary.json", kind: "json" },
  { section: "cv_fold_metrics", file_name: "validation_cv_fold_metrics.csv", kind: "csv" },
  { section: "cv_feature_stability", file_name: "validation_cv_feature_stability.csv", kind: "csv" },
  { section: "cv_coefficients", file_name: "validation_cv_coefficients.csv", kind: "csv" },
  { section: "bootstrap_summary", file_name: "validation_bootstrap_summary.json", kind: "json" },
  { section: "bootstrap_feature_summary", file_name: "validation_bootstrap_feature_summary.csv", kind: "csv" },
  { section: "bootstrap_coefficients", file_name: "validation_bootstrap_coefficients.csv", kind: "csv" },
  { section: "collinearity_summary", file_name: "validation_collinearity_summary.json", kind: "json" },
  { section: "vif", file_name: "validation_vif.csv", kind: "csv" },
  { section: "collinearity_structural", file_name: "validation_collinearity_structural.csv", kind: "csv" },
  { section: "collinearity_pairs", file_name: "validation_collinearity_pairs.csv", kind: "csv" },
  { section: "collinearity_condition", file_name: "validation_collinearity_condition.csv", kind: "csv" },
  {
    section: "collinearity_variance_decomposition",
    file_name: "validation_collinearity_variance_decomposition.csv",
    kind: "csv",
  },
  { section: "nonlinearity_summary", file_name: "validation_nonlinearity_summary.json", kind: "json" },
  { section: "nonlinearity_feature_summary", file_name: "validation_nonlinearity_feature_summary.csv", kind: "csv" },
  { section: "nonlinearity_curve_points", file_name: "validation_nonlinearity_curve_points.csv", kind: "csv" },
  { section: "influence_top", file_name: "validation_influence_top.csv", kind: "csv" },
  { section: "fragility_missingness", file_name: "validation_fragility_missingness.csv", kind: "csv" },
  { section: "break_test", file_name: "validation_break_test.json", kind: "json" },
  { section: "influence_summary", file_name: "validation_influence_summary.json", kind: "json" },
  { section: "fragility_perturbation", file_name: "validation_fragility_perturbation.json", kind: "json" },
  { section: "calibration_robustness", file_name: "validation_calibration_robustness.json", kind: "json" },
  { section: "logit_quantile_summary", file_name: "validation_logit_quantile_summary.json", kind: "json" },
  { section: "logit_quantile_curve", file_name: "validation_logit_quantile_curve.csv", kind: "csv" },
  { section: "logit_actual_vs_predicted_summary", file_name: "validation_logit_actual_vs_predicted_summary.json", kind: "json" },
  { section: "logit_actual_vs_predicted_curve", file_name: "validation_logit_actual_vs_predicted_curve.csv", kind: "csv" },
  { section: "logit_lift_summary", file_name: "validation_logit_lift_summary.json", kind: "json" },
  { section: "logit_lift_curve", file_name: "validation_logit_lift_curve.csv", kind: "csv" },
  { section: "logit_lorenz_summary", file_name: "validation_logit_lorenz_summary.json", kind: "json" },
  { section: "logit_lorenz_curve", file_name: "validation_logit_lorenz_curve.csv", kind: "csv" },
  { section: "logit_roc_summary", file_name: "validation_logit_roc_summary.json", kind: "json" },
  { section: "logit_roc_curve", file_name: "validation_logit_roc_curve.csv", kind: "csv" },
  { section: "logit_operating_points", file_name: "validation_logit_operating_points.csv", kind: "csv" },
  { section: "logit_tossup_summary", file_name: "validation_logit_tossup_summary.json", kind: "json" },
  { section: "logit_tossup_sweep", file_name: "validation_logit_tossup_sweep.csv", kind: "csv" },
  { section: "backtest_integrity", file_name: "backtest_integrity.json", kind: "json" },
];

function loadSectionSpecs(root: string): SectionSpec[] {
  const manifestPath = path.join(root, "validation_manifest.json");
  if (!fs.existsSync(manifestPath)) {
    return DEFAULT_SECTION_SPECS;
  }

  try {
    const raw = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
    if (!Array.isArray(raw?.sections)) {
      return DEFAULT_SECTION_SPECS;
    }
    const specs = raw.sections.filter(
      (value: unknown): value is SectionSpec =>
        value &&
        typeof value === "object" &&
        typeof value.section === "string" &&
        typeof value.file_name === "string" &&
        (value.kind === "csv" || value.kind === "json"),
    );
    return specs.length ? specs : DEFAULT_SECTION_SPECS;
  } catch {
    return DEFAULT_SECTION_SPECS;
  }
}

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  const baseRoot = path.resolve(process.cwd(), "..", "artifacts", "validation");
  const leagueRoot = path.join(baseRoot, league.toLowerCase());
  const root = fs.existsSync(leagueRoot) ? leagueRoot : baseRoot;
  const sectionSpecs = loadSectionSpecs(root);
  const sections = Object.fromEntries(
    sectionSpecs.map((spec) => {
      const filePath = path.join(root, spec.file_name);
      const rawRows = spec.kind === "csv" ? parseCsv(filePath) : parseJson(filePath);
      const rows = spec.tail_rows && spec.kind === "csv" ? rawRows.slice(-spec.tail_rows) : rawRows;
      return [spec.section, rows];
    }),
  ) as Record<string, Array<Record<string, unknown>>>;

  return NextResponse.json({
    league,
    significance: sections.significance,
    sections,
  });
}
