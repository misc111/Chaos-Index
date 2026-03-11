import {
  canonicalizePredictionModelProbabilities,
  orderPredictionModels,
  parseModelWinProbabilities,
  predictionModelHeadline,
  predictionTrustNote,
} from "@/lib/predictions-report";
import { type LeagueCode } from "@/lib/league";
import { getLatestUpcomingAsOf, getPredictionRows } from "@/lib/server/repositories/forecasts";
import { getPairedMoneylineRowsForSnapshots, type RawPairedMoneylineRow } from "@/lib/server/repositories/odds";
import { loadServerModelFeatureMap } from "@/lib/server/repositories/model-feature-map";

function pairKey(homeTeam?: string | null, awayTeam?: string | null): string {
  return `${String(homeTeam || "").trim()}|${String(awayTeam || "").trim()}`;
}

export async function getPredictionsPayload(league: LeagueCode) {
  const featureMap = await loadServerModelFeatureMap(league);
  const asOf = getLatestUpcomingAsOf(league);
  const rawRows = asOf ? getPredictionRows(league, asOf) : [];
  const snapshotIds = Array.from(
    new Set(rawRows.map((row) => String(row.odds_snapshot_id || "").trim()).filter(Boolean))
  );
  const pairedMoneylineRows = getPairedMoneylineRowsForSnapshots(league, snapshotIds);
  const pairedMoneylineByGameId = new Map<string, RawPairedMoneylineRow>();
  const pairedMoneylineByTeamKey = new Map<string, RawPairedMoneylineRow>();

  for (const row of pairedMoneylineRows) {
    const snapshotId = String(row.odds_snapshot_id || "").trim();
    const gameId = Number(row.game_id);
    if (snapshotId && Number.isFinite(gameId)) {
      pairedMoneylineByGameId.set(`${snapshotId}::${gameId}`, row);
    }
    const teamKey = pairKey(row.home_team, row.away_team);
    if (snapshotId && teamKey !== "|") {
      pairedMoneylineByTeamKey.set(`${snapshotId}::${teamKey}`, row);
    }
  }

  const games = rawRows.map((row) => {
    const modelWinProbabilities = canonicalizePredictionModelProbabilities({
      ensemble: Number(row.ensemble_prob_home_win),
      ...parseModelWinProbabilities(row.per_model_probs_json),
    });
    const snapshotId = String(row.odds_snapshot_id || "").trim();
    const pairedMoneyline =
      (snapshotId
        ? pairedMoneylineByGameId.get(`${snapshotId}::${Number(row.game_id)}`) ||
          pairedMoneylineByTeamKey.get(`${snapshotId}::${pairKey(row.home_team, row.away_team)}`)
        : undefined);

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
      odds_as_of_utc: row.odds_as_of_utc == null ? undefined : String(row.odds_as_of_utc),
      home_moneyline: pairedMoneyline?.home_moneyline == null ? undefined : Number(pairedMoneyline.home_moneyline),
      away_moneyline: pairedMoneyline?.away_moneyline == null ? undefined : Number(pairedMoneyline.away_moneyline),
      moneyline_book: pairedMoneyline?.moneyline_book == null ? undefined : String(pairedMoneyline.moneyline_book),
      model_win_probabilities: modelWinProbabilities,
    };
  });

  const nextGameByTeam = new Map<string, { row: (typeof games)[number]; isHome: boolean }>();
  for (const row of games) {
    const homeTeam = row.home_team.trim();
    const awayTeam = row.away_team.trim();
    if (homeTeam && !nextGameByTeam.has(homeTeam)) {
      nextGameByTeam.set(homeTeam, { row, isHome: true });
    }
    if (awayTeam && !nextGameByTeam.has(awayTeam)) {
      nextGameByTeam.set(awayTeam, { row, isHome: false });
    }
  }

  const rows = Array.from(nextGameByTeam.values())
    .filter((entry) => entry.isHome)
    .map((entry) => entry.row)
    .sort(
      (left, right) =>
        right.ensemble_prob_home_win - left.ensemble_prob_home_win ||
        left.game_date_utc.localeCompare(right.game_date_utc) ||
        left.game_id - right.game_id
    );

  const modelColumns = orderPredictionModels(rows.flatMap((row) => Object.keys(row.model_win_probabilities || {})));
  const modelTrustNotes = Object.fromEntries(modelColumns.map((model) => [model, predictionTrustNote(model, league)]));
  const modelSummaries = Object.fromEntries(
    modelColumns.map((model) => {
      const activeFeatures = featureMap.models[model] || [];
      return [
        model,
        {
          headline: predictionModelHeadline(model, league, activeFeatures),
          trust_note: modelTrustNotes[model],
          active_feature_count: activeFeatures.length || undefined,
          active_features: activeFeatures,
        },
      ];
    })
  );

  return {
    league,
    as_of_utc: asOf,
    model_columns: modelColumns,
    model_trust_notes: modelTrustNotes,
    model_summaries: modelSummaries,
    model_feature_map_updated_at_utc: featureMap.updated_at_utc,
    rows,
  };
}
