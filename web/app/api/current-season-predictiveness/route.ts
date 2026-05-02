import fs from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server.js";
import { leagueFromRequest } from "@/lib/league";

export const dynamic = "force-dynamic";

type JsonRecord = Record<string, unknown>;

function readJson(filePath: string): JsonRecord | null {
  if (!fs.existsSync(filePath)) return null;
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8")) as JsonRecord;
  } catch {
    return null;
  }
}

function emptyPayload(league: string): JsonRecord {
  return {
    league,
    status: "not_measurable",
    scoreboard: {
      regular_season_results: 0,
      scored_live_predictions: 0,
      joined_prediction_rows: 0,
      ignored_non_live_predictions: 0,
      quarantined_post_start_predictions: 0,
      quarantined_missing_start_time_predictions: 0,
      quarantined_invalid_probability_predictions: 0,
      unsupported_target_predictions: 0,
    },
    targets: {
      moneyline_home_win: {
        status: "not_measurable",
        reason: "No current-season predictiveness artifact has been generated yet.",
        models: {},
        baselines: {},
        model_vs_baseline: {},
      },
      runline_home_cover: {
        status: "ledger_not_available",
        reason: "The frozen live ledger currently stores moneyline prob_home_win only.",
      },
      totals_over: {
        status: "ledger_not_available",
        reason: "The frozen live ledger currently stores moneyline prob_home_win only.",
      },
    },
    betting_overlay: {
      status: "not_measured",
      reason: "Predictive probability skill is evaluated separately from ROI.",
    },
  };
}

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  const repoRoot = path.resolve(process.cwd(), "..");
  const reportPath = path.resolve(
    repoRoot,
    "artifacts",
    "reports",
    league.toLowerCase(),
    "current_season_predictiveness.json"
  );
  return NextResponse.json(readJson(reportPath) || emptyPayload(league));
}
