import { NextResponse } from "next/server";
import { leagueFromRequest } from "@/lib/league";
import { getRefreshState, runSimpleLeagueTask, trimLog } from "@/lib/server/services/process-tasks";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const league = leagueFromRequest(request);
  const refreshState = getRefreshState("__sportsModelingRefreshState");

  if (refreshState.running) {
    return NextResponse.json(
      {
        ok: false,
        league,
        error: `A refresh is already running for ${refreshState.league || "another league"}.`,
        started_at_utc: refreshState.startedAtUtc,
      },
      { status: 409 }
    );
  }

  refreshState.running = true;
  refreshState.league = league;
  refreshState.startedAtUtc = new Date().toISOString();

  try {
    const runResult = await runSimpleLeagueTask("refresh-data", league);
    if (runResult.code !== 0) {
      return NextResponse.json(
        {
          ok: false,
          league,
          error: "Data refresh failed.",
          details: trimLog(runResult.stderr || runResult.stdout || `Command exited with code ${runResult.code}`),
        },
        { status: 500 }
      );
    }

    return NextResponse.json({
      ok: true,
      league,
      started_at_utc: refreshState.startedAtUtc,
      refreshed_at_utc: new Date().toISOString(),
    });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        league,
        error: `Data refresh failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      },
      { status: 500 }
    );
  } finally {
    refreshState.running = false;
    refreshState.league = undefined;
    refreshState.startedAtUtc = undefined;
  }
}
