import { getHistoricalReplayGames } from "@/lib/bet-history";
import { centralDateKeyFromTimestamp, centralTodayDateKey, dateKeyForScheduledGame } from "@/lib/games-today";
import { type LeagueCode } from "@/lib/league";
import { getLatestUpcomingAsOf, getScheduledTodayRows } from "@/lib/server/repositories/forecasts";
import {
  getMoneylineRowsForSnapshots,
  getOddsSnapshots,
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

  if (!asOf) {
    return {
      league,
      as_of_utc: null,
      date_central: centralTodayDateKey(),
      historical_coverage_start_central: historicalReplay.coverage_start_central,
      historical_rows: historicalReplay.rows,
      rows: [],
    };
  }

  const rows = getScheduledTodayRows(league, asOf).map((row) => ({
    ...row,
    home_win_probability: normalizeProbability(row.home_win_probability),
  }));

  const snapshotRows = getOddsSnapshots(league);
  const latestOddsSnapshotByDate = new Map<string, { odds_snapshot_id: string; as_of_utc: string }>();
  for (const row of snapshotRows) {
    const snapshotId = String(row.odds_snapshot_id || "").trim();
    const asOfUtc = String(row.as_of_utc || "").trim();
    const snapshotDateKey = centralDateKeyFromTimestamp(asOfUtc);
    if (!snapshotId || !asOfUtc || !snapshotDateKey || latestOddsSnapshotByDate.has(snapshotDateKey)) {
      continue;
    }
    latestOddsSnapshotByDate.set(snapshotDateKey, { odds_snapshot_id: snapshotId, as_of_utc: asOfUtc });
  }

  const neededSnapshotIds = Array.from(
    new Set(
      [
        ...rows.map((row) => {
          const rowDateKey = dateKeyForScheduledGame(row);
          return rowDateKey ? latestOddsSnapshotByDate.get(rowDateKey)?.odds_snapshot_id || "" : "";
        }),
        ...historicalReplay.rows.map((row) => latestOddsSnapshotByDate.get(row.date_central)?.odds_snapshot_id || ""),
      ].filter(Boolean)
    )
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
    const daySnapshot = latestOddsSnapshotByDate.get(row.date_central);
    const snapshotId = daySnapshot?.odds_snapshot_id || "";
    const moneylineMatch =
      moneylineByGameId.get(`${snapshotId}::${Number(row.game_id)}`) ||
      moneylineByTeamKey.get(`${snapshotId}::${row.home_team}|${row.away_team}`);
    const over190Match =
      over190ByGameId.get(`${snapshotId}::${Number(row.game_id)}`) ||
      over190ByTeamKey.get(`${snapshotId}::${row.home_team}|${row.away_team}`);

    return {
      game_id: row.game_id,
      game_date_utc: row.date_central,
      home_team: row.home_team,
      away_team: row.away_team,
      home_win_probability: row.home_win_probability,
      forecast_as_of_utc: row.forecast_as_of_utc,
      odds_as_of_utc: daySnapshot?.as_of_utc || null,
      start_time_utc: row.start_time_utc,
      home_moneyline: moneylineMatch?.home_moneyline ?? null,
      away_moneyline: moneylineMatch?.away_moneyline ?? null,
      home_moneyline_book: moneylineMatch?.home_moneyline_book ?? null,
      away_moneyline_book: moneylineMatch?.away_moneyline_book ?? null,
      over_190_price: over190Match?.over_190_price ?? null,
      over_190_point: over190Match?.over_190_point ?? null,
      over_190_book: over190Match?.over_190_book ?? null,
      replay_decision: row.replay_decision ?? null,
    };
  }

  const enrichedRows = rows.map((row) => {
    const rowDateKey = dateKeyForScheduledGame(row);
    const daySnapshot = rowDateKey ? latestOddsSnapshotByDate.get(rowDateKey) : null;
    const snapshotId = daySnapshot?.odds_snapshot_id || "";
    const moneylineMatch =
      moneylineByGameId.get(`${snapshotId}::${Number(row.game_id)}`) ||
      moneylineByTeamKey.get(`${snapshotId}::${row.home_team}|${row.away_team}`);
    const over190Match =
      over190ByGameId.get(`${snapshotId}::${Number(row.game_id)}`) ||
      over190ByTeamKey.get(`${snapshotId}::${row.home_team}|${row.away_team}`);
    return {
      ...row,
      odds_as_of_utc: daySnapshot?.as_of_utc || null,
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
    date_central: centralTodayDateKey(),
    historical_coverage_start_central: historicalReplay.coverage_start_central,
    historical_rows: historicalReplay.rows.map(buildHistoricalRow),
    rows: enrichedRows,
  };
}
