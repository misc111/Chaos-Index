import {
  BET_SIZING_STYLES,
  BET_STRATEGIES,
  DEFAULT_BET_SIZING_STYLE,
  DEFAULT_BET_STRATEGY,
  type BetSizingStyle,
  type BetStrategy,
} from "@/lib/betting-strategy";
import {
  resolveBetStrategyConfigs,
  type BetStrategyOptimizationSummary,
  type OptimizableHistoricalBetRow,
  type ResolvedBetStrategyConfig,
} from "@/lib/betting-optimizer";
import {
  formatBetUnitLabel,
  settleBet,
  type BetDecision,
} from "@/lib/betting";
import type {
  BetHistorySizingBundle,
  BetHistoryStrategyBundle,
  BetHistoryResponse,
  BetHistorySummary,
  HistoricalBetRow,
  HistoricalDailyPoint,
} from "@/lib/bet-history-types";
import { runSqlJson } from "@/lib/db";
import type { LeagueCode } from "@/lib/league";
import {
  loadOrCreateHistoricalReplayDecisions,
  type HistoricalReplayDecisionSet,
  type HistoricalReplayDecisionSnapshot,
} from "@/lib/replay-bets";

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
  forecast_model_run_id?: string | null;
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

type RawOver190Row = {
  odds_snapshot_id?: string | null;
  game_id?: number | null;
  home_team?: string | null;
  away_team?: string | null;
  over_190_price?: number | null;
  over_190_point?: number | null;
  over_190_book?: string | null;
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
  forecast_model_run_id: string | null;
  odds_as_of_utc: string;
  odds_snapshot_id: string;
  home_moneyline: number;
  away_moneyline: number;
  home_moneyline_book: string | null;
  away_moneyline_book: string | null;
  over_190_price: number | null;
  over_190_point: number | null;
  over_190_book: string | null;
  home_win_probability: number;
  replay_decisions?: HistoricalReplayDecisionSet;
};

type HistoricalReplayDataset = {
  total_final_games: number;
  games_with_forecast: number;
  games_with_odds: number;
  earliest_final_date: string | null;
  coverage_start_central: string | null;
  coverage_end_central: string | null;
  strategy_configs: Record<BetStrategy, ResolvedBetStrategyConfig>;
  strategy_optimization: BetStrategyOptimizationSummary;
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
  if (value === null || value === undefined || value === "") return null;
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

function betDecisionFromReplaySnapshot(snapshot: HistoricalReplayDecisionSnapshot): BetDecision {
  return {
    bet: snapshot.bet_label,
    reason: snapshot.reason,
    side: snapshot.side,
    team: snapshot.team,
    stake: snapshot.stake,
    odds: snapshot.odds,
    modelProbability: snapshot.model_probability,
    marketProbability: snapshot.market_probability,
    edge: snapshot.edge,
    expectedValue: snapshot.expected_value,
  };
}

function historicalGamesSql(league: LeagueCode): string {
  const escapedLeague = escapeSqlString(league);

  // Finalized-game replay must use immutable prediction history. The live
  // `upcoming_game_forecasts` table is intentionally excluded here because
  // retrains can replace rows in place for the same `as_of_utc`.
  return `
    WITH finalized_games AS (
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
        COALESCE(g.start_time_utc, r.final_utc, r.game_date_utc || 'T23:59:59Z') AS replay_cutoff_utc
      FROM results r
      LEFT JOIN games g ON g.game_id = r.game_id
      WHERE r.home_win IS NOT NULL
    ),
    ranked_forecasts AS (
      SELECT
        fg.game_id,
        p.as_of_utc,
        p.prob_home_win,
        p.model_run_id,
        ROW_NUMBER() OVER (
          PARTITION BY fg.game_id
          ORDER BY
            DATETIME(p.as_of_utc) DESC,
            DATETIME(COALESCE(mr.created_at_utc, p.as_of_utc)) DESC,
            p.prediction_id DESC
        ) AS rn
      FROM finalized_games fg
      JOIN predictions p
        ON p.game_id = fg.game_id
       AND p.model_name = 'ensemble'
       AND DATETIME(p.as_of_utc) <= DATETIME(fg.replay_cutoff_utc)
      LEFT JOIN model_runs mr
        ON mr.model_run_id = p.model_run_id
      WHERE DATETIME(COALESCE(mr.created_at_utc, p.as_of_utc)) <= DATETIME(fg.replay_cutoff_utc)
    )
    SELECT
      fg.game_id,
      fg.game_date_utc,
      fg.start_time_utc,
      fg.final_utc,
      fg.home_team,
      fg.away_team,
      fg.home_score,
      fg.away_score,
      fg.home_win,
      rf.as_of_utc AS forecast_as_of_utc,
      rf.model_run_id AS forecast_model_run_id,
      rf.prob_home_win AS home_win_probability,
      (
        SELECT s.odds_snapshot_id
        FROM odds_snapshots s
        WHERE s.league = '${escapedLeague}'
          AND DATETIME(s.as_of_utc) <= DATETIME(fg.replay_cutoff_utc)
          AND EXISTS (
            SELECT 1
            FROM odds_market_lines l
            WHERE l.odds_snapshot_id = s.odds_snapshot_id
              AND l.market_key = 'h2h'
              AND (
                (l.game_id IS NOT NULL AND l.game_id = fg.game_id)
                OR (l.home_team = fg.home_team AND l.away_team = fg.away_team)
              )
          )
        ORDER BY DATETIME(s.as_of_utc) DESC
        LIMIT 1
      ) AS odds_snapshot_id,
      (
        SELECT s.as_of_utc
        FROM odds_snapshots s
        WHERE s.league = '${escapedLeague}'
          AND DATETIME(s.as_of_utc) <= DATETIME(fg.replay_cutoff_utc)
          AND EXISTS (
            SELECT 1
            FROM odds_market_lines l
            WHERE l.odds_snapshot_id = s.odds_snapshot_id
              AND l.market_key = 'h2h'
              AND (
                (l.game_id IS NOT NULL AND l.game_id = fg.game_id)
                OR (l.home_team = fg.home_team AND l.away_team = fg.away_team)
              )
          )
        ORDER BY DATETIME(s.as_of_utc) DESC
        LIMIT 1
      ) AS odds_as_of_utc
    FROM finalized_games fg
    LEFT JOIN ranked_forecasts rf
      ON rf.game_id = fg.game_id
     AND rf.rn = 1
    ORDER BY DATETIME(fg.replay_cutoff_utc) ASC, fg.game_id ASC
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

function over190Sql(snapshotIds: string[]): string {
  const inList = snapshotIds.map((snapshotId) => `'${escapeSqlString(snapshotId)}'`).join(", ");

  return `
    WITH ranked AS (
      SELECT
        odds_snapshot_id,
        game_id,
        home_team,
        away_team,
        outcome_price,
        outcome_point,
        bookmaker_title,
        ROW_NUMBER() OVER (
          PARTITION BY
            odds_snapshot_id,
            COALESCE(CAST(game_id AS TEXT), home_team || '|' || away_team)
          ORDER BY outcome_point ASC, DATETIME(bookmaker_last_update_utc) DESC, line_id DESC
        ) AS rn
      FROM odds_market_lines
      WHERE odds_snapshot_id IN (${inList})
        AND market_key = 'alternate_totals'
        AND outcome_side = 'over'
        AND outcome_point >= 190.0
        AND outcome_price IS NOT NULL
    )
    SELECT
      odds_snapshot_id,
      game_id,
      home_team,
      away_team,
      MAX(CASE WHEN rn = 1 THEN outcome_price END) AS over_190_price,
      MAX(CASE WHEN rn = 1 THEN outcome_point END) AS over_190_point,
      MAX(CASE WHEN rn = 1 THEN bookmaker_title END) AS over_190_book
    FROM ranked
    GROUP BY odds_snapshot_id, game_id, home_team, away_team
  `;
}

function toReplayableHistoricalGames(rows: HistoricalReplayGameRow[]) {
  return rows.map((row) => ({
    game_id: row.game_id,
    date_central: row.date_central,
    home_team: row.home_team,
    away_team: row.away_team,
    forecast_as_of_utc: row.forecast_as_of_utc,
    forecast_model_run_id: row.forecast_model_run_id,
    odds_as_of_utc: row.odds_as_of_utc,
    odds_snapshot_id: row.odds_snapshot_id,
    home_moneyline: row.home_moneyline,
    away_moneyline: row.away_moneyline,
    home_win_probability: row.home_win_probability,
  }));
}

function toOptimizableHistoricalBetRows(rows: HistoricalReplayGameRow[]): OptimizableHistoricalBetRow[] {
  return rows.map((row) => ({
    game_id: row.game_id,
    date_central: row.date_central,
    home_team: row.home_team,
    away_team: row.away_team,
    home_win_probability: row.home_win_probability,
    home_moneyline: row.home_moneyline,
    away_moneyline: row.away_moneyline,
    home_win: row.home_win,
  }));
}

function buildEmptyReplayDecisionSet(): HistoricalReplayDecisionSet {
  return BET_STRATEGIES.reduce((strategyAcc, strategy) => {
    strategyAcc[strategy] = BET_SIZING_STYLES.reduce((sizingAcc, sizingStyle) => {
      sizingAcc[sizingStyle] = null;
      return sizingAcc;
    }, {} as HistoricalReplayDecisionSet[BetStrategy]);
    return strategyAcc;
  }, {} as HistoricalReplayDecisionSet);
}

function buildHistoricalReplayDataset(league: LeagueCode): HistoricalReplayDataset {
  // Maintainer note: this is the shared "pregame replay" source for both
  // Bet History and Games Today past-date navigation. Keep the core fields
  // league-agnostic here; league-specific extras should stay optional.
  const rawGames = runSqlJson(historicalGamesSql(league), { league }) as RawHistoricalGameRow[];
  const uniqueSnapshotIds = Array.from(
    new Set(rawGames.map((row) => String(row.odds_snapshot_id || "").trim()).filter(Boolean))
  );
  const moneylineRows = uniqueSnapshotIds.length
    ? (runSqlJson(moneylineSql(uniqueSnapshotIds), { league }) as RawMoneylineRow[])
    : [];
  const over190Rows =
    uniqueSnapshotIds.length && league === "NBA"
      ? (runSqlJson(over190Sql(uniqueSnapshotIds), { league }) as RawOver190Row[])
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

  const over190ByGame = new Map<string, RawOver190Row>();
  const over190ByTeams = new Map<string, RawOver190Row>();

  for (const row of over190Rows) {
    const snapshotId = String(row.odds_snapshot_id || "").trim();
    if (!snapshotId) continue;

    const gameId = numberOrNull(row.game_id);
    if (gameId !== null) {
      over190ByGame.set(snapshotGameKey(snapshotId, gameId), row);
    }

    const homeTeam = String(row.home_team || "").trim();
    const awayTeam = String(row.away_team || "").trim();
    if (homeTeam && awayTeam) {
      over190ByTeams.set(snapshotTeamKey(snapshotId, homeTeam, awayTeam), row);
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
    const over190Match =
      over190ByGame.get(snapshotGameKey(oddsSnapshotId, gameId)) ||
      over190ByTeams.get(snapshotTeamKey(oddsSnapshotId, homeTeam, awayTeam));

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
      forecast_model_run_id: row.forecast_model_run_id ? String(row.forecast_model_run_id) : null,
      odds_as_of_utc: oddsAsOf,
      odds_snapshot_id: oddsSnapshotId,
      home_moneyline: homeMoneyline,
      away_moneyline: awayMoneyline,
      home_moneyline_book: moneylineMatch?.home_moneyline_book ? String(moneylineMatch.home_moneyline_book) : null,
      away_moneyline_book: moneylineMatch?.away_moneyline_book ? String(moneylineMatch.away_moneyline_book) : null,
      over_190_price: numberOrNull(over190Match?.over_190_price),
      over_190_point: numberOrNull(over190Match?.over_190_point),
      over_190_book: over190Match?.over_190_book ? String(over190Match.over_190_book) : null,
      home_win_probability: normalizeProbability(row.home_win_probability),
    });
  }

  const replayableRows = toReplayableHistoricalGames(rows);
  const { strategyConfigs, optimizationSummary } = resolveBetStrategyConfigs(toOptimizableHistoricalBetRows(rows));
  const replayDecisionsByProfile = new Map<string, Map<number, HistoricalReplayDecisionSnapshot>>();
  for (const strategy of BET_STRATEGIES) {
    for (const sizingStyle of BET_SIZING_STYLES) {
      const resolvedConfig = strategyConfigs[strategy];
      replayDecisionsByProfile.set(
        `${strategy}:${sizingStyle}`,
        loadOrCreateHistoricalReplayDecisions(
          replayableRows,
          league,
          strategy,
          sizingStyle,
          resolvedConfig,
          resolvedConfig.config_signature
        )
      );
    }
  }

  return {
    total_final_games: rawGames.length,
    games_with_forecast: gamesWithForecast,
    games_with_odds: gamesWithOdds,
    earliest_final_date: earliestFinalDate,
    coverage_start_central: coverageStartDate,
    coverage_end_central: coverageEndDate,
    strategy_configs: strategyConfigs,
    strategy_optimization: optimizationSummary,
    rows: rows.map((row) => ({
      ...row,
      replay_decisions: BET_STRATEGIES.reduce((strategyAcc, strategy) => {
        strategyAcc[strategy] = BET_SIZING_STYLES.reduce((sizingAcc, sizingStyle) => {
          sizingAcc[sizingStyle] = replayDecisionsByProfile.get(`${strategy}:${sizingStyle}`)?.get(row.game_id) ?? null;
          return sizingAcc;
        }, {} as HistoricalReplayDecisionSet[BetStrategy]);
        return strategyAcc;
      }, buildEmptyReplayDecisionSet()),
    })),
  };
}

export function getHistoricalReplayGames(league: LeagueCode): {
  coverage_start_central: string | null;
  coverage_end_central: string | null;
  strategy_configs: Record<BetStrategy, ResolvedBetStrategyConfig>;
  strategy_optimization: BetStrategyOptimizationSummary;
  rows: HistoricalReplayGameRow[];
} {
  const dataset = buildHistoricalReplayDataset(league);
  return {
    coverage_start_central: dataset.coverage_start_central,
    coverage_end_central: dataset.coverage_end_central,
    strategy_configs: dataset.strategy_configs,
    strategy_optimization: dataset.strategy_optimization,
    rows: dataset.rows,
  };
}

function buildBetHistoryStrategyBundle(
  dataset: HistoricalReplayDataset,
  strategy: BetStrategy,
  sizingStyle: BetSizingStyle
): BetHistoryStrategyBundle {
  let analyzedGames = 0;
  let wins = 0;
  let losses = 0;
  let totalRisked = 0;
  let cumulativeProfit = 0;

  const bets: HistoricalBetRow[] = [];

  for (const row of dataset.rows) {
    analyzedGames += 1;

    const replayDecision = row.replay_decisions?.[strategy]?.[sizingStyle];
    if (!replayDecision) {
      continue;
    }

    const decision = betDecisionFromReplaySnapshot(replayDecision);

    if (decision.stake <= 0 || decision.side === "none" || !decision.team || !Number.isFinite(decision.odds)) {
      continue;
    }

    const settlement = settleBet(decision, row.home_win);
    if (settlement.outcome === "no_bet") continue;

    totalRisked += replayDecision.stake;
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
      bet_label: formatBetUnitLabel(decision.team, replayDecision.stake),
      reason: replayDecision.reason,
      side: decision.side,
      team: decision.team,
      stake: replayDecision.stake,
      odds: Number(decision.odds),
      expected_value: replayDecision.expected_value,
      edge: replayDecision.edge,
      model_probability: replayDecision.model_probability,
      market_probability: replayDecision.market_probability,
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
    summary,
    daily_points: dailyPoints,
    bets,
  };
}

export function getBetHistory(league: LeagueCode): BetHistoryResponse {
  const dataset = buildHistoricalReplayDataset(league);

  return {
    league,
    default_strategy: DEFAULT_BET_STRATEGY,
    default_sizing_style: DEFAULT_BET_SIZING_STYLE,
    strategy_configs: dataset.strategy_configs,
    strategy_optimization: dataset.strategy_optimization,
    strategies: BET_STRATEGIES.reduce((strategyAcc, strategy) => {
      strategyAcc[strategy] = BET_SIZING_STYLES.reduce((sizingAcc, sizingStyle) => {
        sizingAcc[sizingStyle] = buildBetHistoryStrategyBundle(dataset, strategy, sizingStyle);
        return sizingAcc;
      }, {} as BetHistorySizingBundle);
      return strategyAcc;
    }, {} as Record<BetStrategy, BetHistorySizingBundle>),
  };
}
