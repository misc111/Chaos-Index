import { NextResponse } from "next/server";
import { runSqlJson } from "@/lib/db";
import { leagueFromRequest } from "@/lib/league";
import { getRefreshState, runSimpleLeagueTask, trimLog } from "@/lib/server/services/process-tasks";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const league = leagueFromRequest(request);
  const refreshState = getRefreshState("__sportsModelingRefreshOddsState");

  if (refreshState.running) {
    return NextResponse.json(
      {
        ok: false,
        league,
        error: `An odds refresh is already running for ${refreshState.league || "another league"}.`,
        started_at_utc: refreshState.startedAtUtc,
      },
      { status: 409 }
    );
  }

  refreshState.running = true;
  refreshState.league = league;
  refreshState.startedAtUtc = new Date().toISOString();

  try {
    const runResult = await runSimpleLeagueTask("fetch-odds", league);
    if (runResult.code !== 0) {
      return NextResponse.json(
        {
          ok: false,
          league,
          error: "Odds refresh failed.",
          details: trimLog(runResult.stderr || runResult.stdout || `Command exited with code ${runResult.code}`),
        },
        { status: 500 }
      );
    }

    const latestOddsSnapshot = runSqlJson(
      `
      SELECT odds_snapshot_id, as_of_utc, event_count, row_count
      FROM odds_snapshots
      WHERE league = '${league}'
      ORDER BY DATETIME(as_of_utc) DESC
      LIMIT 1
      `,
      { league }
    );

    return NextResponse.json({
      ok: true,
      league,
      started_at_utc: refreshState.startedAtUtc,
      refreshed_at_utc: new Date().toISOString(),
      odds_snapshot_id: latestOddsSnapshot?.[0]?.odds_snapshot_id || null,
      odds_as_of_utc: latestOddsSnapshot?.[0]?.as_of_utc || null,
      event_count: latestOddsSnapshot?.[0]?.event_count ?? null,
      row_count: latestOddsSnapshot?.[0]?.row_count ?? null,
    });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        league,
        error: `Odds refresh failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      },
      { status: 500 }
    );
  } finally {
    refreshState.running = false;
    refreshState.league = undefined;
    refreshState.startedAtUtc = undefined;
  }
}
