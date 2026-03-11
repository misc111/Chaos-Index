import { NextResponse } from "next/server.js";
import { leagueFromRequest } from "@/lib/league";
import { getPerformancePayload } from "@/lib/server/services/performance";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  return NextResponse.json(getPerformancePayload(league));
}
