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

function parseCsv(filePath: string): Record<string, any>[] {
  if (!fs.existsSync(filePath)) return [];
  const text = fs.readFileSync(filePath, "utf8").trim();
  if (!text) return [];
  const [header, ...lines] = text.split(/\r?\n/);
  const cols = header.split(",");
  return lines.map((line) => {
    const vals = line.split(",");
    const obj: Record<string, any> = {};
    cols.forEach((c, i) => {
      const v = vals[i] ?? "";
      const n = Number(v);
      obj[c] = Number.isFinite(n) && v !== "" ? n : v;
    });
    return obj;
  });
}

function parseJson(filePath: string): Record<string, any>[] {
  if (!fs.existsSync(filePath)) return [];
  const raw = fs.readFileSync(filePath, "utf8");
  try {
    return [JSON.parse(raw)];
  } catch {
    const sanitized = raw.replace(/\bNaN\b/g, "null").replace(/\bInfinity\b/g, "null").replace(/\b-Infinity\b/g, "null");
    try {
      return [JSON.parse(sanitized)];
    } catch {
      return [];
    }
  }
}

const DEFAULT_SECTION_SPECS: SectionSpec[] = [
  { section: "significance", file_name: "validation_significance.csv", kind: "csv" },
  { section: "coef_paths", file_name: "validation_coef_paths.csv", kind: "csv", tail_rows: 200 },
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
  { section: "influence_top", file_name: "validation_influence_top.csv", kind: "csv" },
  { section: "fragility_missingness", file_name: "validation_fragility_missingness.csv", kind: "csv" },
  { section: "break_test", file_name: "validation_break_test.json", kind: "json" },
  { section: "influence_summary", file_name: "validation_influence_summary.json", kind: "json" },
  { section: "fragility_perturbation", file_name: "validation_fragility_perturbation.json", kind: "json" },
  { section: "calibration_robustness", file_name: "validation_calibration_robustness.json", kind: "json" },
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
      (value: any): value is SectionSpec =>
        value &&
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
  ) as Record<string, Record<string, any>[]>;

  return NextResponse.json({
    league,
    significance: sections.significance,
    sections,
  });
}
