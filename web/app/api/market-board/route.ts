import { NextResponse } from "next/server.js";
import { leagueFromRequest } from "@/lib/league";
import { getMarketBoardPayload } from "@/lib/server/services/market-board";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  return NextResponse.json(await getMarketBoardPayload(leagueFromRequest(request)));
}
