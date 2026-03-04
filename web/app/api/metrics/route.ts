import { NextResponse } from "next/server";
import { runSqlJson } from "@/lib/db";
import fs from "node:fs";
import path from "node:path";

function maybeCsv(filePath: string): any[] {
  if (!fs.existsSync(filePath)) return [];
  const text = fs.readFileSync(filePath, "utf8").trim();
  if (!text) return [];
  const [header, ...lines] = text.split(/\r?\n/);
  const cols = header.split(",");
  return lines.map((line) => {
    const vals = line.split(",");
    const obj: Record<string, string> = {};
    cols.forEach((c, i) => (obj[c] = vals[i] ?? ""));
    return obj;
  });
}

export async function GET() {
  const leaderboard = runSqlJson(
    `SELECT model_name, window_label, n_games, log_loss, brier, accuracy, ece, calibration_alpha, calibration_beta
     FROM performance_aggregates
     ORDER BY as_of_utc DESC, window_label ASC, log_loss ASC
     LIMIT 200`
  );

  const calibration = runSqlJson(
    `SELECT model_name, window_label, calibration_alpha, calibration_beta, ece, mce, n_games
     FROM performance_aggregates
     ORDER BY as_of_utc DESC, log_loss ASC
     LIMIT 200`
  );

  const slicesPath = path.resolve(process.cwd(), "..", "artifacts", "validation", "slice_analysis.csv");
  const slices = maybeCsv(slicesPath);

  return NextResponse.json({ leaderboard, calibration, slices });
}
