import { NextResponse } from "next/server.js";
import { getBetHistory } from "@/lib/bet-history";
import { leagueFromRequest } from "@/lib/league";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  return NextResponse.json(getBetHistory(league));
}
