import { NextResponse } from "next/server";
import { getBetHistory } from "@/lib/bet-history";
import { leagueFromRequest } from "@/lib/league";

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  return NextResponse.json(getBetHistory(league));
}
