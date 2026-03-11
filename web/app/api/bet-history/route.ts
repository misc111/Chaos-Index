import fs from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server.js";
import { getBetHistory } from "@/lib/bet-history";
import type { BetHistoryResponse } from "@/lib/bet-history-types";
import { leagueFromRequest } from "@/lib/league";

export const dynamic = "force-dynamic";

function readCommittedBetHistorySnapshot(league: string): BetHistoryResponse | null {
  const filePath = path.join(process.cwd(), "public", "staging-data", league.toLowerCase(), "bet-history.json");
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8")) as BetHistoryResponse;
  } catch {
    return null;
  }
}

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  return NextResponse.json(readCommittedBetHistorySnapshot(league) || getBetHistory(league));
}
