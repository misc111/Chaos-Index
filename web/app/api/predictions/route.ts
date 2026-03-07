import { NextResponse } from "next/server";
import { leagueFromRequest } from "@/lib/league";
import { getPredictionsPayload } from "@/lib/server/services/predictions";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  return NextResponse.json(await getPredictionsPayload(leagueFromRequest(request)));
}
