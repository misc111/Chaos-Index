import { NextResponse } from "next/server.js";
import { leagueFromRequest } from "@/lib/league";
import { getTrainingState, parseRequestedModels, startTrainingTask, summarizeTrainingState } from "@/lib/server/services/process-tasks";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  return NextResponse.json({ ok: true, requested_league: league, ...summarizeTrainingState() });
}

export async function POST(request: Request) {
  const league = leagueFromRequest(request);
  const trainingState = getTrainingState();

  if (trainingState.running) {
    const state = summarizeTrainingState();
    const inProgressMessage = `Training already in progress for ${trainingState.league || "another league"}.`;
    const responseState = { ...state, error: state.error || inProgressMessage };
    return NextResponse.json(
      {
        ok: false,
        requested_league: league,
        ...responseState,
        message: inProgressMessage,
      },
      { status: 409 }
    );
  }

  let models: string[] | null = null;
  try {
    const body = (await request.json().catch(() => ({}))) as Record<string, unknown>;
    models = parseRequestedModels(body.models);
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        requested_league: league,
        error: error instanceof Error ? error.message : "Invalid model request payload.",
      },
      { status: 400 }
    );
  }

  try {
    startTrainingTask(league, models);
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        requested_league: league,
        error: error instanceof Error ? error.message : "Unable to start training.",
      },
      { status: 500 }
    );
  }

  return NextResponse.json({ ok: true, requested_league: league, ...summarizeTrainingState() });
}
