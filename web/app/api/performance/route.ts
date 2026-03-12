import { NextResponse } from "next/server.js";
import { leagueFromRequest } from "@/lib/league";
import { performanceReplayExperimentFromRequest } from "@/lib/performance-replay-experiments";
import { getPerformancePayload } from "@/lib/server/services/performance";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  const experiment = performanceReplayExperimentFromRequest(request);
  return NextResponse.json(getPerformancePayload(league, experiment));
}
