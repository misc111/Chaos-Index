import { NextResponse } from "next/server";
import { leagueFromRequest } from "@/lib/league";
import { getGamesTodayPayload } from "@/lib/server/services/games-today";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  return NextResponse.json(await getGamesTodayPayload(leagueFromRequest(request)));
}
