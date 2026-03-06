import { computeBetDecision, settleBet } from "@/lib/betting";
import type {
  BetHistoryResponse,
  BetHistorySummary,
  HistoricalBetRow,
  HistoricalDailyPoint,
} from "@/lib/bet-history-types";
import { runSqlJson } from "@/lib/db";
import type { LeagueCode } from "@/lib/league";

type RawHistoricalGameRow = {
  game_id?: number | null;
  game_date_utc?: string | null;
  start_time_utc?: string | null;
  final_utc?: string | null;
  home_team?: string | null;
  away_team?: string | null;
  home_score?: number | null;
  away_score?: number | null;
  home_win?: number | null;
  forecast_as_of_utc?: string | null;
  home_win_probability?: number | null;
  odds_snapshot_id?: string | null;
  odds_as_of_utc?: string | null;
};

type RawMoneylineRow = {
  odds_snapshot_id?: string | null;
  game_id?: number | null;
  home_team?: string | null;
  away_team?: string | null;
  home_moneyline?: number | null;
  away_moneyline?: number | null;
  home_moneyline_book?: string | null;
  away_moneyline_book?: string | null;
};

export type HistoricalReplayGameRow = {
  game_id: number;
  date_central: string;
  start_time_utc: string | null;
  final_utc: string | null;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  home_win: number | null;
  forecast_as_of_utc: string;
  odds_as_of_utc: string;
  odds_snapshot_id: string;
  home_moneyline: number;
  away_moneyline: number;
  home_moneyline_book: string | null;
  away_moneyline_book: string | null;
  home_win_probability: number;
};

type HistoricalReplayDataset = {
  total_final_games: number;
  games_with_forecast: number;
  games_with_odds: number;
  earliest_final_date: string | null;
  coverage_start_central: string | null;
  coverage_end_central: string | null;
  rows: HistoricalReplayGameRow[];
};

function escapeSqlString(value: string): string {
  return value.replace(/'/g, "''");
}

function normalizeUtcTimestamp(value: string): string {
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z$/.test(value)) {
    return value.replace("Z", ":00Z");
  }
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(value)) {
    return `${value}:00Z`;
  }
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(value)) {
    return `${value}Z`;
  }
  return value;
}

function centralDateKeyFromTimestamp(value?: string | null): string | null {
  if (!value) return null;
  const parsed = new Date(normalizeUtcTimestamp(value));
  if (Number.isNaN(parsed.getTime())) return null;

  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Chicago",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(parsed);

  const year = parts.find((part) => part.type === "year")?.value;
  const month = parts.find((part) => part.type === "month")?.value;
  const day = parts.find((part) => part.type === "day")?.value;
  if (!year || !month || !day) return null;
  return `${year}-${month}-${day}`;
}

function dateKeyForHistoricalRow(row: Pick<RawHistoricalGameRow, "start_time_utc" | "game_date_utc" | "final_utc">): string | null {
  const fromStart = centralDateKeyFromTimestamp(row.start_time_utc);
  if (fromStart) return fromStart;

  const fromFinal = centralDateKeyFromTimestamp(row.final_utc);
  if (fromFinal) return fromFinal;

  const gameDate = String(row.game_date_utc || "").trim();
  if (/^\d{4}-\d{2}-\d{2}$/.test(gameDate)) return gameDate;

  return gameDate.length >= 10 ? gameDate.slice(0, 10) : null;
}

function normalizeProbability(value: unknown): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0.5;
  return Math.max(0, Math.min(1, numeric));
}

function numberOrNull(value: unknown): number | null {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function parseDateKey(dateKey: string): Date {
  return new Date(`${dateKey}T12:00:00Z`);
}

function addDays(dateKey: string, days: number): string {
  const date = parseDateKey(dateKey);
  date.setUTCDate(date.getUTCDate() + days);
  return date.toISOString().slice(0, 10);
}

function mondayOfWeek(dateKey: string): string {
  const date = parseDateKey(dateKey);
  const weekday = date.getUTCDay();
  const diff = weekday === 0 ? -6 : 1 - weekday;
  return addDays(dateKey, diff);
}

function snapshotGameKey(snapshotId: string, gameId: number): string {
  return `${snapshotId}::${gameId}`;
}

function snapshotTeamKey(snapshotId: string, homeTeam: string, awayTeam: string): string {
  return `${snapshotId}::${homeTeam}::${awayTeam}`;
}

function buildNote(
  totalFinalGames: number,
  analyzedGames: number,
  suggestedBets: number,
  earliestFinalDate: string | null,
  coverageStartDate: string | null
): string {
  if (totalFinalGames === 0) return "No finalized games are available yet.";
  if (analyzedGames === 0) {
    return "No finalized games have both a stored pregame forecast snapshot and a matching pregame odds snapshot yet.";
  }
  if (suggestedBets === 0) {
    return "The replay window is available, but none of the covered games cleared the current bet threshold.";
  }
  if (earliestFinalDate && coverageStartDate && coverageStartDate > earliestFinalDate) {
    return `Replay coverage starts on ${coverageStartDate} because older pregame forecast or odds snapshots are not stored in this database.`;
  }
  return "Replay results are based only on games with stored pregame forecast and odds snapshots.";
}

function historicalGamesSql(league: LeagueCode): string {
  const escapedLeague = escapeSqlString(league);
  const cutoffExpr = "COALESCE(g.start_time_utc, r.final_utc, r.game_date_utc || 'T23:59:59Z')";

  return `
    SELECT
      r.game_id,
      r.game_date_utc,
      g.start_time_utc,
      r.final_utc,
      r.home_team,
      r.away_team,
      r.home_score,
      r.away_score,
      r.home_win,
      (
        SELECT u.as_of_utc
        FROM upcoming_game_forecasts u
        WHERE u.game_id = r.game_id
          AND DATETIME(u.as_of_utc) <= DATETIME(${cutoffExpr})
        ORDER BY DATETIME(u.as_of_utc) DESC
        LIMIT 1
      ) AS forecast_as_of_utc,
      (
        SELECT u.ensemble_prob_home_win
        FROM upcoming_game_forecasts u
        WHERE u.game_id = r.game_id
          AND DATETIME(u.as_of_utc) <= DATETIME(${cutoffExpr})
        ORDER BY DATETIME(u.as_of_utc) DESC
        LIMIT 1
      ) AS home_win_probability,
      (
        SELECT s.odds_snapshot_id
        FROM odds_snapshots s
        WHERE s.league = '${escapedLeague}'
          AND DATETIME(s.as_of_utc) <= DATETIME(${cutoffExpr})
          AND EXISTS (
            SELECT 1
            FROM odds_market_lines l
            WHERE l.odds_snapshot_id = s.odds_snapshot_id
              AND l.market_key = 'h2h'
              AND (
                (l.game_id IS NOT NULL AND l.game_id = r.game_id)
                OR (l.home_team = r.home_team AND l.away_team = r.away_team)
              )
          )
        ORDER BY DATETIME(s.as_of_utc) DESC
        LIMIT 1
      ) AS odds_snapshot_id,
      (
        SELECT s.as_of_utc
        FROM odds_snapshots s
        WHERE s.league = '${escapedLeague}'
          AND DATETIME(s.as_of_utc) <= DATETIME(${cutoffExpr})
          AND EXISTS (
            SELECT 1
            FROM odds_market_lines l
            WHERE l.odds_snapshot_id = s.odds_snapshot_id
              AND l.market_key = 'h2h'
              AND (
                (l.game_id IS NOT NULL AND l.game_id = r.game_id)
                OR (l.home_team = r.home_team AND l.away_team = r.away_team)
              )
          )
        ORDER BY DATETIME(s.as_of_utc) DESC
        LIMIT 1
      ) AS odds_as_of_utc
    FROM results r
    LEFT JOIN games g ON g.game_id = r.game_id
    WHERE r.home_win IS NOT NULL
    ORDER BY DATETIME(${cutoffExpr}) ASC, r.game_id ASC
  `;
}

function moneylineSql(snapshotIds: string[]): string {
  const inList = snapshotIds.map((snapshotId) => `'${escapeSqlString(snapshotId)}'`).join(", ");

  return `
    WITH ranked AS (
      SELECT
        odds_snapshot_id,
        game_id,
        home_team,
        away_team,
        outcome_side,
        outcome_price,
        bookmaker_title,
        ROW_NUMBER() OVER (
          PARTITION BY
            odds_snapshot_id,
            COALESCE(CAST(game_id AS TEXT), home_team || '|' || away_team),
            outcome_side
          ORDER BY outcome_price DESC, DATETIME(bookmaker_last_update_utc) DESC, line_id DESC
        ) AS rn
      FROM odds_market_lines
      WHERE odds_snapshot_id IN (${inList})
        AND market_key = 'h2h'
        AND outcome_side IN ('home', 'away')
        AND outcome_price IS NOT NULL
    )
    SELECT
      odds_snapshot_id,
      game_id,
      home_team,
      away_team,
      MAX(CASE WHEN outcome_side = 'home' AND rn = 1 THEN outcome_price END) AS home_moneyline,
      MAX(CASE WHEN outcome_side = 'away' AND rn = 1 THEN outcome_price END) AS away_moneyline,
      MAX(CASE WHEN outcome_side = 'home' AND rn = 1 THEN bookmaker_title END) AS home_moneyline_book,
      MAX(CASE WHEN outcome_side = 'away' AND rn = 1 THEN bookmaker_title END) AS away_moneyline_book
    FROM ranked
    GROUP BY odds_snapshot_id, game_id, home_team, away_team
  `;
}

function buildHistoricalReplayDataset(league: LeagueCode): HistoricalReplayDataset {
  const rawGames = runSqlJson(historicalGamesSql(league), { league }) as RawHistoricalGameRow[];
  const uniqueSnapshotIds = Array.from(
    new Set(rawGames.map((row) => String(row.odds_snapshot_id || "").trim()).filter(Boolean))
  );
  const moneylineRows = uniqueSnapshotIds.length
    ? (runSqlJson(moneylineSql(uniqueSnapshotIds), { league }) as RawMoneylineRow[])
    : [];

  const moneylineByGame = new Map<string, RawMoneylineRow>();
  const moneylineByTeams = new Map<string, RawMoneylineRow>();

  for (const row of moneylineRows) {
    const snapshotId = String(row.odds_snapshot_id || "").trim();
    if (!snapshotId) continue;

    const gameId = numberOrNull(row.game_id);
    if (gameId !== null) {
      moneylineByGame.set(snapshotGameKey(snapshotId, gameId), row);
    }

    const homeTeam = String(row.home_team || "").trim();
    const awayTeam = String(row.away_team || "").trim();
    if (homeTeam && awayTeam) {
      moneylineByTeams.set(snapshotTeamKey(snapshotId, homeTeam, awayTeam), row);
    }
  }

  let gamesWithForecast = 0;
  let gamesWithOdds = 0;
  const earliestFinalDate = rawGames.length ? dateKeyForHistoricalRow(rawGames[0]) : null;
  let coverageStartDate: string | null = null;
  let coverageEndDate: string | null = null;

  const rows: HistoricalReplayGameRow[] = [];

  for (const row of rawGames) {
    const gameId = numberOrNull(row.game_id);
    const forecastAsOf = String(row.forecast_as_of_utc || "").trim();
    const oddsSnapshotId = String(row.odds_snapshot_id || "").trim();
    const oddsAsOf = String(row.odds_as_of_utc || "").trim();
    const homeTeam = String(row.home_team || "").trim();
    const awayTeam = String(row.away_team || "").trim();
    const dateCentral = dateKeyForHistoricalRow(row);

    if (forecastAsOf) gamesWithForecast += 1;
    if (oddsSnapshotId) gamesWithOdds += 1;

    if (gameId === null || !forecastAsOf || !oddsSnapshotId || !oddsAsOf || !homeTeam || !awayTeam || !dateCentral) {
      continue;
    }

    const moneylineMatch =
      moneylineByGame.get(snapshotGameKey(oddsSnapshotId, gameId)) ||
      moneylineByTeams.get(snapshotTeamKey(oddsSnapshotId, homeTeam, awayTeam));

    const homeMoneyline = numberOrNull(moneylineMatch?.home_moneyline);
    const awayMoneyline = numberOrNull(moneylineMatch?.away_moneyline);
    if (homeMoneyline === null || awayMoneyline === null) {
      continue;
    }

    if (!coverageStartDate || dateCentral < coverageStartDate) coverageStartDate = dateCentral;
    if (!coverageEndDate || dateCentral > coverageEndDate) coverageEndDate = dateCentral;

    rows.push({
      game_id: gameId,
      date_central: dateCentral,
      start_time_utc: row.start_time_utc ? String(row.start_time_utc) : null,
      final_utc: row.final_utc ? String(row.final_utc) : null,
      home_team: homeTeam,
      away_team: awayTeam,
      home_score: numberOrNull(row.home_score),
      away_score: numberOrNull(row.away_score),
      home_win: numberOrNull(row.home_win),
      forecast_as_of_utc: forecastAsOf,
      odds_as_of_utc: oddsAsOf,
      odds_snapshot_id: oddsSnapshotId,
      home_moneyline: homeMoneyline,
      away_moneyline: awayMoneyline,
      home_moneyline_book: moneylineMatch?.home_moneyline_book ? String(moneylineMatch.home_moneyline_book) : null,
      away_moneyline_book: moneylineMatch?.away_moneyline_book ? String(moneylineMatch.away_moneyline_book) : null,
      home_win_probability: normalizeProbability(row.home_win_probability),
    });
  }

  return {
    total_final_games: rawGames.length,
    games_with_forecast: gamesWithForecast,
    games_with_odds: gamesWithOdds,
    earliest_final_date: earliestFinalDate,
    coverage_start_central: coverageStartDate,
    coverage_end_central: coverageEndDate,
    rows,
  };
}

export function getHistoricalReplayGames(league: LeagueCode): {
  coverage_start_central: string | null;
  coverage_end_central: string | null;
  rows: HistoricalReplayGameRow[];
} {
  const dataset = buildHistoricalReplayDataset(league);
  return {
    coverage_start_central: dataset.coverage_start_central,
    coverage_end_central: dataset.coverage_end_central,
    rows: dataset.rows,
  };
}

export function getBetHistory(league: LeagueCode): BetHistoryResponse {
  const dataset = buildHistoricalReplayDataset(league);
  let analyzedGames = 0;
  let wins = 0;
  let losses = 0;
  let totalRisked = 0;
  let cumulativeProfit = 0;

  const bets: HistoricalBetRow[] = [];

  for (const row of dataset.rows) {
    analyzedGames += 1;

    const decision = computeBetDecision({
      home_team: row.home_team,
      away_team: row.away_team,
      home_win_probability: row.home_win_probability,
      home_moneyline: row.home_moneyline,
      away_moneyline: row.away_moneyline,
    });

    if (decision.stake <= 0 || decision.side === "none" || !decision.team || !Number.isFinite(decision.odds)) {
      continue;
    }

    const settlement = settleBet(decision, row.home_win);
    if (settlement.outcome === "no_bet") continue;

    totalRisked += decision.stake;
    cumulativeProfit += settlement.profit;
    if (settlement.outcome === "win") wins += 1;
    if (settlement.outcome === "loss") losses += 1;

    bets.push({
      game_id: row.game_id,
      date_central: row.date_central,
      week_start_central: mondayOfWeek(row.date_central),
      start_time_utc: row.start_time_utc,
      final_utc: row.final_utc,
      home_team: row.home_team,
      away_team: row.away_team,
      home_score: row.home_score,
      away_score: row.away_score,
      forecast_as_of_utc: row.forecast_as_of_utc,
      odds_as_of_utc: row.odds_as_of_utc,
      odds_snapshot_id: row.odds_snapshot_id,
      home_moneyline: row.home_moneyline,
      away_moneyline: row.away_moneyline,
      bet_label: decision.bet,
      reason: decision.reason,
      side: decision.side,
      team: decision.team,
      stake: decision.stake,
      odds: Number(decision.odds),
      expected_value: decision.expectedValue,
      edge: decision.edge,
      model_probability: decision.modelProbability,
      market_probability: decision.marketProbability,
      outcome: settlement.outcome,
      profit: settlement.profit,
      payout: settlement.payout,
      cumulative_profit: cumulativeProfit,
    });
  }

  const dailyTotals = new Map<string, { risked: number; daily_profit: number; bet_count: number }>();

  for (const bet of bets) {
    const current = dailyTotals.get(bet.date_central) || { risked: 0, daily_profit: 0, bet_count: 0 };
    current.risked += bet.stake;
    current.daily_profit += bet.profit;
    current.bet_count += 1;
    dailyTotals.set(bet.date_central, current);
  }

  const dailyPoints: HistoricalDailyPoint[] = [];
  let dailyCumulative = 0;
  for (const dateCentral of Array.from(dailyTotals.keys()).sort()) {
    const current = dailyTotals.get(dateCentral);
    if (!current) continue;
    dailyCumulative += current.daily_profit;
    dailyPoints.push({
      date_central: dateCentral,
      risked: current.risked,
      daily_profit: current.daily_profit,
      cumulative_profit: dailyCumulative,
      bet_count: current.bet_count,
    });
  }

  const summary: BetHistorySummary = {
    total_final_games: dataset.total_final_games,
    games_with_forecast: dataset.games_with_forecast,
    games_with_odds: dataset.games_with_odds,
    analyzed_games: analyzedGames,
    suggested_bets: bets.length,
    wins,
    losses,
    total_risked: totalRisked,
    total_profit: cumulativeProfit,
    roi: totalRisked > 0 ? cumulativeProfit / totalRisked : 0,
    coverage_start_central: dataset.coverage_start_central,
    coverage_end_central: dataset.coverage_end_central,
    note: buildNote(
      dataset.total_final_games,
      analyzedGames,
      bets.length,
      dataset.earliest_final_date,
      dataset.coverage_start_central
    ),
  };

  return {
    league,
    summary,
    daily_points: dailyPoints,
    bets,
  };
}
