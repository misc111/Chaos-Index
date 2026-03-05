import { NextResponse } from "next/server";
import { leagueFromRequest } from "@/lib/league";
import fs from "node:fs";
import path from "node:path";

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

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  const baseRoot = path.resolve(process.cwd(), "..", "artifacts", "validation");
  const leagueRoot = path.join(baseRoot, league.toLowerCase());
  const root = fs.existsSync(leagueRoot) ? leagueRoot : baseRoot;

  const sections: Record<string, Record<string, any>[]> = {
    significance: parseCsv(path.join(root, "validation_significance.csv")),
    coef_paths: parseCsv(path.join(root, "validation_coef_paths.csv")).slice(-200),
    vif: parseCsv(path.join(root, "validation_vif.csv")),
    influence_top: parseCsv(path.join(root, "validation_influence_top.csv")),
    fragility_missingness: parseCsv(path.join(root, "validation_fragility_missingness.csv")),
    break_test: parseJson(path.join(root, "validation_break_test.json")),
    influence_summary: parseJson(path.join(root, "validation_influence_summary.json")),
    fragility_perturbation: parseJson(path.join(root, "validation_fragility_perturbation.json")),
    calibration_robustness: parseJson(path.join(root, "validation_calibration_robustness.json")),
    backtest_integrity: parseJson(path.join(root, "backtest_integrity.json")),
  };

  return NextResponse.json({
    league,
    significance: sections.significance,
    sections,
  });
}
