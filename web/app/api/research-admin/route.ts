import { NextResponse } from "next/server.js";
import { leagueFromRequest } from "@/lib/league";
import { getResearchAdminPayload } from "@/lib/server/services/research-admin";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  return NextResponse.json(await getResearchAdminPayload(leagueFromRequest(request)));
}
