import { NextResponse } from "next/server";
import { runSqlJson } from "@/lib/db";
import { leagueFromRequest } from "@/lib/league";

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  const latest = runSqlJson("SELECT MAX(as_of_utc) AS as_of_utc FROM upcoming_game_forecasts", { league });
  const asOf = latest?.[0]?.as_of_utc;
  const rows = asOf
    ? runSqlJson(
        `SELECT game_id, game_date_utc, home_team, away_team, ensemble_prob_home_win, predicted_winner,
                spread_mean, spread_sd, bayes_ci_low, bayes_ci_high, uncertainty_flags_json
         FROM upcoming_game_forecasts
         WHERE as_of_utc = '${asOf}'
         ORDER BY game_date_utc ASC`,
        { league }
      )
    : [];
  return NextResponse.json({ league, as_of_utc: asOf, rows });
}
