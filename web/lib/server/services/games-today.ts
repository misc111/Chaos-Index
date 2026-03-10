import { getHistoricalReplayGames } from "@/lib/bet-history";
import { parseModelWinProbabilities, selectBettingModelProbability } from "@/lib/betting-model";
import { centralTodayDateKey, dateKeyForScheduledGame, shiftCentralDateKey } from "@/lib/games-today";
import { type LeagueCode } from "@/lib/league";
import { getGamesTodaySnapshotRows, getLatestUpcomingAsOf } from "@/lib/server/repositories/forecasts";
import { getPreferredBettingModelName } from "@/lib/server/services/betting-driver";
import {
  getMoneylineRowsForSnapshots,
  getOver190RowsForSnapshots,
  type RawMoneylineRow,
  type RawOver190Row,
} from "@/lib/server/repositories/odds";

function normalizeProbability(value: unknown): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0.5;
  return Math.max(0, Math.min(1, numeric));
}

export async function getGamesTodayPayload(league: LeagueCode) {
  const historicalReplay = getHistoricalReplayGames(league);
  const asOf = getLatestUpcomingAsOf(league);
  const todayKey = centralTodayDateKey();
  const snapshotFallbackWindowStart = shiftCentralDateKey(todayKey, -1);
  const preferredBettingModelName = getPreferredBettingModelName(league);

  if (!asOf) {
    return {
      league,
      as_of_utc: null,
      date_central: todayKey,
      historical_coverage_start_central: historicalReplay.coverage_start_central,
      strategy_configs: historicalReplay.strategy_configs,
      strategy_optimization: historicalReplay.strategy_optimization,
      historical_rows: historicalReplay.rows,
      rows: [],
    };
  }

  const rows = getGamesTodaySnapshotRows(league, asOf).map((row) => ({
      ...row,
      ...selectBettingModelProbability(
        normalizeProbability(row.home_win_probability),
        parseModelWinProbabilities(row.per_model_probs_json),
        preferredBettingModelName
      ),
    }))
    .filter((row) => {
      const dateKey = dateKeyForScheduledGame(row);
      return Boolean(dateKey && dateKey >= snapshotFallbackWindowStart);
    });

  const neededSnapshotIds = Array.from(
    new Set(rows.map((row) => String(row.odds_snapshot_id || "").trim()).filter(Boolean))
  );

  const moneylineRows = getMoneylineRowsForSnapshots(league, neededSnapshotIds);
  const over190Rows = getOver190RowsForSnapshots(league, neededSnapshotIds);

  const moneylineByGameId = new Map<string, RawMoneylineRow>();
  const moneylineByTeamKey = new Map<string, RawMoneylineRow>();
  for (const row of moneylineRows) {
    const snapshotId = String(row.odds_snapshot_id || "").trim();
    const gameId = Number(row.game_id);
    if (snapshotId && Number.isFinite(gameId)) {
      moneylineByGameId.set(`${snapshotId}::${gameId}`, row);
    }
    const homeTeam = String(row.home_team || "").trim();
    const awayTeam = String(row.away_team || "").trim();
    if (snapshotId && homeTeam && awayTeam) {
      moneylineByTeamKey.set(`${snapshotId}::${homeTeam}|${awayTeam}`, row);
    }
  }

  const over190ByGameId = new Map<string, RawOver190Row>();
  const over190ByTeamKey = new Map<string, RawOver190Row>();
  for (const row of over190Rows) {
    const snapshotId = String(row.odds_snapshot_id || "").trim();
    const gameId = Number(row.game_id);
    if (snapshotId && Number.isFinite(gameId)) {
      over190ByGameId.set(`${snapshotId}::${gameId}`, row);
    }
    const homeTeam = String(row.home_team || "").trim();
    const awayTeam = String(row.away_team || "").trim();
    if (snapshotId && homeTeam && awayTeam) {
      over190ByTeamKey.set(`${snapshotId}::${homeTeam}|${awayTeam}`, row);
    }
  }

  function buildHistoricalRow(row: (typeof historicalReplay.rows)[number]) {
    return {
      game_id: row.game_id,
      game_date_utc: row.date_central,
      home_team: row.home_team,
      away_team: row.away_team,
      home_win_probability: row.home_win_probability,
      betting_model_name: row.betting_model_name,
      model_win_probabilities: row.model_win_probabilities,
      forecast_as_of_utc: row.forecast_as_of_utc,
      odds_as_of_utc: row.odds_as_of_utc,
      start_time_utc: row.start_time_utc,
      home_moneyline: row.home_moneyline,
      away_moneyline: row.away_moneyline,
      home_moneyline_book: row.home_moneyline_book,
      away_moneyline_book: row.away_moneyline_book,
      over_190_price: row.over_190_price,
      over_190_point: row.over_190_point,
      over_190_book: row.over_190_book,
      replay_decisions: row.replay_decisions ?? null,
    };
  }

  const enrichedRows = rows.map((row) => {
    const snapshotId = String(row.odds_snapshot_id || "").trim();
    const moneylineMatch =
      (snapshotId
        ? moneylineByGameId.get(`${snapshotId}::${Number(row.game_id)}`) ||
          moneylineByTeamKey.get(`${snapshotId}::${row.home_team}|${row.away_team}`)
        : undefined);
    const over190Match =
      (snapshotId
        ? over190ByGameId.get(`${snapshotId}::${Number(row.game_id)}`) ||
          over190ByTeamKey.get(`${snapshotId}::${row.home_team}|${row.away_team}`)
        : undefined);
    return {
      game_id: row.game_id,
      game_date_utc: row.game_date_utc,
      home_team: row.home_team,
      away_team: row.away_team,
      home_win_probability: row.home_win_probability,
      betting_model_name: row.betting_model_name,
      model_win_probabilities: row.model_win_probabilities,
      forecast_as_of_utc: row.forecast_as_of_utc,
      start_time_utc: row.start_time_utc,
      odds_as_of_utc: row.odds_as_of_utc || null,
      home_moneyline: moneylineMatch?.home_moneyline ?? null,
      away_moneyline: moneylineMatch?.away_moneyline ?? null,
      home_moneyline_book: moneylineMatch?.home_moneyline_book ?? null,
      away_moneyline_book: moneylineMatch?.away_moneyline_book ?? null,
      over_190_price: over190Match?.over_190_price ?? null,
      over_190_point: over190Match?.over_190_point ?? null,
      over_190_book: over190Match?.over_190_book ?? null,
    };
  });

  return {
    league,
    as_of_utc: asOf,
    date_central: todayKey,
    historical_coverage_start_central: historicalReplay.coverage_start_central,
    strategy_configs: historicalReplay.strategy_configs,
    strategy_optimization: historicalReplay.strategy_optimization,
    historical_rows: historicalReplay.rows.map(buildHistoricalRow),
    rows: enrichedRows,
  };
}
