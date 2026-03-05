import { NextResponse } from "next/server";
import { runSqlJson } from "@/lib/db";
import { leagueFromRequest } from "@/lib/league";
import { orderPredictionModels, parseModelWinProbabilities, predictionTrustNote } from "@/lib/predictions-report";

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  const latest = runSqlJson("SELECT MAX(as_of_utc) AS as_of_utc FROM upcoming_game_forecasts", { league });
  const asOf = latest?.[0]?.as_of_utc;
  const rawRows = asOf
    ? runSqlJson(
        `SELECT game_id, game_date_utc, home_team, away_team, ensemble_prob_home_win, predicted_winner,
                spread_mean, spread_sd, bayes_ci_low, bayes_ci_high, uncertainty_flags_json, per_model_probs_json
         FROM upcoming_game_forecasts
         WHERE as_of_utc = '${asOf}'
         ORDER BY ensemble_prob_home_win DESC, game_date_utc ASC, game_id ASC`,
        { league }
      )
    : [];

  const rows = rawRows.map((row) => {
    const modelWinProbabilities = {
      ensemble: Number(row.ensemble_prob_home_win),
      ...parseModelWinProbabilities(row.per_model_probs_json),
    };

    return {
      game_id: Number(row.game_id),
      game_date_utc: String(row.game_date_utc || ""),
      home_team: String(row.home_team || ""),
      away_team: String(row.away_team || ""),
      ensemble_prob_home_win: Number(row.ensemble_prob_home_win),
      predicted_winner: String(row.predicted_winner || ""),
      spread_mean: row.spread_mean == null ? undefined : Number(row.spread_mean),
      spread_sd: row.spread_sd == null ? undefined : Number(row.spread_sd),
      bayes_ci_low: row.bayes_ci_low == null ? undefined : Number(row.bayes_ci_low),
      bayes_ci_high: row.bayes_ci_high == null ? undefined : Number(row.bayes_ci_high),
      uncertainty_flags_json: row.uncertainty_flags_json == null ? undefined : String(row.uncertainty_flags_json),
      model_win_probabilities: modelWinProbabilities,
    };
  });

  const modelColumns = orderPredictionModels(
    rows.flatMap((row) => Object.keys(row.model_win_probabilities || {}))
  );

  const modelTrustNotes = Object.fromEntries(
    modelColumns.map((model) => [model, predictionTrustNote(model)])
  );

  return NextResponse.json({ league, as_of_utc: asOf, model_columns: modelColumns, model_trust_notes: modelTrustNotes, rows });
}
