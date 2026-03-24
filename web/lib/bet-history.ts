import {
  BET_STRATEGIES,
  DEFAULT_BET_STRATEGY,
  getDefaultBetStrategyForLeague,
  type BetStrategy,
} from "@/lib/betting-strategy";
import {
  resolveBetStrategyConfigs,
  type BetStrategyOptimizationSummary,
  type OptimizableHistoricalBetRow,
  type ResolvedBetStrategyConfig,
} from "@/lib/betting-optimizer";
import {
  HISTORICAL_BANKROLL_START_DATE_CENTRAL,
  HISTORICAL_BANKROLL_START_DOLLARS,
  formatBetRecommendationLabel,
  settleBet,
  type BetDecision,
} from "@/lib/betting";
import type { ModelWinProbabilities } from "@/lib/betting-model";
import type {
  BetHistoryStrategyBundle,
  BetHistoryResponse,
  BetHistorySummary,
  HistoricalBetRow,
  HistoricalDailyPoint,
} from "@/lib/bet-history-types";
import { runSqlJson } from "@/lib/db";
import type { LeagueCode } from "@/lib/league";
import {
  DIAGNOSTIC_FORECAST_SOURCE,
  STORED_FORECAST_SOURCE,
  effectiveReplayOddsAsOfSql,
  escapeSqlString,
  historicalForecastCandidatesUnionSql,
  historicalMoneylineSql,
} from "@/lib/server/services/replay-data";
import { getPreferredBettingModelName } from "@/lib/server/services/betting-driver";
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
  forecast_source?: string | null;
  home_win_probability?: number | null;
  odds_snapshot_id?: string | null;
  odds_as_of_utc?: string | null;
};

type RawHistoricalModelProbabilityRow = {
  game_id?: number | null;
  model_name?: string | null;
  prob_home_win?: number | null;
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
  league: LeagueCode;
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
  betting_model_name: string;
  model_win_probabilities: ModelWinProbabilities;
  replay_decisions?: HistoricalReplayDecisionSet;
};

type HistoricalReplayDataset = {
  total_final_games: number;
  games_with_forecast: number;
  games_with_odds: number;
  stored_forecast_games: number;
  diagnostic_forecast_games: number;
  earliest_final_date: string | null;
  coverage_start_central: string | null;
  coverage_end_central: string | null;
  strategy_configs: Record<BetStrategy, ResolvedBetStrategyConfig>;
  strategy_optimization: BetStrategyOptimizationSummary;
  rows: HistoricalReplayGameRow[];
};

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
  storedForecastGames: number,
  diagnosticForecastGames: number
): string {
  const usesDiagnosticBackfill = diagnosticForecastGames > 0;
  const coverageDescription = usesDiagnosticBackfill
    ? "Replay uses the latest stored pregame forecast before game time and backfills older gaps with historical diagnostics, paired with the latest pregame odds before game time."
    : "Replay uses the latest stored pregame forecast and latest matching pregame odds before game time.";

  if (totalFinalGames === 0) return "No finalized games are available yet.";
  if (analyzedGames === 0) {
    if (storedForecastGames > 0 || diagnosticForecastGames > 0) {
      return "Forecast history exists, but no finalized games have matching pregame odds snapshots yet.";
    }
    return "No finalized games have either stored pregame forecast history or historical diagnostic backfill yet.";
  }
  if (suggestedBets === 0) {
    return `${coverageDescription} None of the covered games cleared the current bet threshold.`;
  }
  return coverageDescription;
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

function historicalGamesSql(league: LeagueCode, modelName: string): string {
  const escapedLeague = escapeSqlString(league);
  const effectiveOddsAsOf = effectiveReplayOddsAsOfSql("s", "l");

  // Finalized-game replay should use the latest stored pregame prediction that
  // existed before game time. Historical diagnostics only fill holes where the
  // main prediction history is missing. The market side uses the latest
  // pregame odds snapshot available before the game starts.
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
    forecast_candidates AS (
      ${historicalForecastCandidatesUnionSql({ modelName })}
    ),
    ranked_forecasts AS (
      SELECT
        game_id,
        as_of_utc,
        prob_home_win,
        model_run_id,
        forecast_source,
        ROW_NUMBER() OVER (
          PARTITION BY game_id
          ORDER BY
            source_priority ASC,
            DATETIME(as_of_utc) DESC,
            DATETIME(created_at_utc) DESC,
            row_sort_id DESC
        ) AS rn
      FROM forecast_candidates
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
      rf.forecast_source AS forecast_source,
      rf.prob_home_win AS home_win_probability,
      (
        SELECT l.odds_snapshot_id
        FROM odds_market_lines l
        JOIN odds_snapshots s
          ON s.odds_snapshot_id = l.odds_snapshot_id
        WHERE rf.as_of_utc IS NOT NULL
          AND s.league = '${escapedLeague}'
          AND l.market_key = 'h2h'
          AND (
            (l.game_id IS NOT NULL AND l.game_id = fg.game_id)
            OR (l.home_team = fg.home_team AND l.away_team = fg.away_team)
          )
          AND DATETIME(${effectiveOddsAsOf}) <= DATETIME(fg.replay_cutoff_utc)
        ORDER BY DATETIME(${effectiveOddsAsOf}) DESC, DATETIME(s.as_of_utc) DESC, l.line_id DESC
        LIMIT 1
      ) AS odds_snapshot_id,
      (
        SELECT ${effectiveOddsAsOf}
        FROM odds_market_lines l
        JOIN odds_snapshots s
          ON s.odds_snapshot_id = l.odds_snapshot_id
        WHERE rf.as_of_utc IS NOT NULL
          AND s.league = '${escapedLeague}'
          AND l.market_key = 'h2h'
          AND (
            (l.game_id IS NOT NULL AND l.game_id = fg.game_id)
            OR (l.home_team = fg.home_team AND l.away_team = fg.away_team)
          )
          AND DATETIME(${effectiveOddsAsOf}) <= DATETIME(fg.replay_cutoff_utc)
        ORDER BY DATETIME(${effectiveOddsAsOf}) DESC, DATETIME(s.as_of_utc) DESC, l.line_id DESC
        LIMIT 1
      ) AS odds_as_of_utc
    FROM finalized_games fg
    LEFT JOIN ranked_forecasts rf
      ON rf.game_id = fg.game_id
     AND rf.rn = 1
    ORDER BY DATETIME(fg.replay_cutoff_utc) ASC, fg.game_id ASC
  `;
}

function historicalModelProbabilitiesSql(): string {
  return `
    WITH finalized_games AS (
      SELECT
        r.game_id,
        COALESCE(g.start_time_utc, r.final_utc, r.game_date_utc || 'T23:59:59Z') AS replay_cutoff_utc
      FROM results r
      LEFT JOIN games g ON g.game_id = r.game_id
      WHERE r.home_win IS NOT NULL
    ),
    forecast_candidates AS (
      ${historicalForecastCandidatesUnionSql()}
    ),
    ranked_forecasts AS (
      SELECT
        game_id,
        model_name,
        prob_home_win,
        ROW_NUMBER() OVER (
          PARTITION BY game_id, model_name
          ORDER BY
            source_priority ASC,
            DATETIME(as_of_utc) DESC,
            DATETIME(created_at_utc) DESC,
            row_sort_id DESC
        ) AS rn
      FROM forecast_candidates
    )
    SELECT game_id, model_name, prob_home_win
    FROM ranked_forecasts
    WHERE rn = 1
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
    league: row.league,
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
    betting_model_name: row.betting_model_name,
    model_win_probabilities: row.model_win_probabilities,
  }));
}

function toOptimizableHistoricalBetRows(rows: HistoricalReplayGameRow[]): OptimizableHistoricalBetRow[] {
  return rows.map((row) => ({
    game_id: row.game_id,
    league: row.league,
    date_central: row.date_central,
    home_team: row.home_team,
    away_team: row.away_team,
    home_win_probability: row.home_win_probability,
    home_moneyline: row.home_moneyline,
    away_moneyline: row.away_moneyline,
    home_win: row.home_win,
    betting_model_name: row.betting_model_name,
    model_win_probabilities: row.model_win_probabilities,
  }));
}

function buildEmptyReplayDecisionSet(): HistoricalReplayDecisionSet {
  return BET_STRATEGIES.reduce((strategyAcc, strategy) => {
    strategyAcc[strategy] = null;
    return strategyAcc;
  }, {} as HistoricalReplayDecisionSet);
}

function buildHistoricalReplayDataset(league: LeagueCode): HistoricalReplayDataset {
  // Maintainer note: this is the shared "pregame replay" source for both
  // Bet History and Games Today past-date navigation. Keep the core fields
  // league-agnostic here; league-specific extras should stay optional.
  const preferredBettingModelName = getPreferredBettingModelName(league, "historicalReplay");
  const rawGames = runSqlJson(historicalGamesSql(league, preferredBettingModelName), { league }) as RawHistoricalGameRow[];
  const modelProbabilityRows = runSqlJson(historicalModelProbabilitiesSql(), { league }) as RawHistoricalModelProbabilityRow[];
  const modelProbabilitiesByGame = new Map<number, ModelWinProbabilities>();
  for (const row of modelProbabilityRows) {
    const gameId = numberOrNull(row.game_id);
    const modelName = String(row.model_name || "").trim();
    const probability = numberOrNull(row.prob_home_win);
    if (gameId === null || !modelName) continue;
    const current = modelProbabilitiesByGame.get(gameId) || {};
    current[modelName] = probability;
    modelProbabilitiesByGame.set(gameId, current);
  }
  const uniqueSnapshotIds = Array.from(
    new Set(rawGames.map((row) => String(row.odds_snapshot_id || "").trim()).filter(Boolean))
  );
  const moneylineRows = uniqueSnapshotIds.length
    ? (runSqlJson(historicalMoneylineSql(uniqueSnapshotIds, { includeBookmakerTitles: true }), { league }) as RawMoneylineRow[])
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
  let storedForecastGames = 0;
  let diagnosticForecastGames = 0;
  const earliestFinalDate = rawGames.length ? dateKeyForHistoricalRow(rawGames[0]) : null;
  let coverageStartDate: string | null = null;
  let coverageEndDate: string | null = null;

  const rows: HistoricalReplayGameRow[] = [];

  for (const row of rawGames) {
    const gameId = numberOrNull(row.game_id);
    const forecastAsOf = String(row.forecast_as_of_utc || "").trim();
    const oddsSnapshotId = String(row.odds_snapshot_id || "").trim();
    const oddsAsOf = String(row.odds_as_of_utc || "").trim();
    const forecastSource = String(row.forecast_source || "").trim();
    const homeTeam = String(row.home_team || "").trim();
    const awayTeam = String(row.away_team || "").trim();
    const dateCentral = dateKeyForHistoricalRow(row);

    if (forecastAsOf) gamesWithForecast += 1;
    if (oddsSnapshotId) gamesWithOdds += 1;
    if (forecastAsOf && forecastSource === STORED_FORECAST_SOURCE) storedForecastGames += 1;
    if (forecastAsOf && forecastSource === DIAGNOSTIC_FORECAST_SOURCE) diagnosticForecastGames += 1;

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
      league,
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
      betting_model_name: preferredBettingModelName,
      model_win_probabilities: modelProbabilitiesByGame.get(gameId) || { [preferredBettingModelName]: normalizeProbability(row.home_win_probability) },
    });
  }

  const replayableRows = toReplayableHistoricalGames(rows);
  const { strategyConfigs, optimizationSummary } = resolveBetStrategyConfigs(toOptimizableHistoricalBetRows(rows));
  const replayDecisionsByProfile = new Map<string, Map<number, HistoricalReplayDecisionSnapshot>>();
  for (const strategy of BET_STRATEGIES) {
    const resolvedConfig = strategyConfigs[strategy];
    replayDecisionsByProfile.set(
      strategy,
      loadOrCreateHistoricalReplayDecisions(
        replayableRows,
        league,
        strategy,
        resolvedConfig,
        resolvedConfig.config_signature
      )
    );
  }

  return {
    total_final_games: rawGames.length,
    games_with_forecast: gamesWithForecast,
    games_with_odds: gamesWithOdds,
    stored_forecast_games: storedForecastGames,
    diagnostic_forecast_games: diagnosticForecastGames,
    earliest_final_date: earliestFinalDate,
    coverage_start_central: coverageStartDate,
    coverage_end_central: coverageEndDate,
    strategy_configs: strategyConfigs,
    strategy_optimization: optimizationSummary,
    rows: rows.map((row) => ({
      ...row,
      replay_decisions: BET_STRATEGIES.reduce((strategyAcc, strategy) => {
        strategyAcc[strategy] = replayDecisionsByProfile.get(strategy)?.get(row.game_id) ?? null;
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
  strategy: BetStrategy
): BetHistoryStrategyBundle {
  const replayRows = dataset.rows.filter((row) => row.date_central >= HISTORICAL_BANKROLL_START_DATE_CENTRAL);
  let analyzedGames = 0;
  let wins = 0;
  let losses = 0;
  let totalRisked = 0;
  let cumulativeProfit = 0;

  const bets: HistoricalBetRow[] = [];

  for (const row of replayRows) {
    analyzedGames += 1;

    const replayDecision = row.replay_decisions?.[strategy];
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
      bet_label: formatBetRecommendationLabel(decision.team, replayDecision.stake),
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
  let bankroll = HISTORICAL_BANKROLL_START_DOLLARS;
  for (const dateCentral of Array.from(dailyTotals.keys()).sort()) {
    const current = dailyTotals.get(dateCentral);
    if (!current) continue;
    dailyCumulative += current.daily_profit;
    bankroll += current.daily_profit;
    dailyPoints.push({
      date_central: dateCentral,
      risked: current.risked,
      daily_profit: current.daily_profit,
      cumulative_profit: dailyCumulative,
      cumulative_bankroll: bankroll,
      bet_count: current.bet_count,
    });
  }

  const replayCoverageStart =
    dataset.coverage_start_central && dataset.coverage_start_central > HISTORICAL_BANKROLL_START_DATE_CENTRAL
      ? dataset.coverage_start_central
      : HISTORICAL_BANKROLL_START_DATE_CENTRAL;
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
    starting_bankroll: HISTORICAL_BANKROLL_START_DOLLARS,
    current_bankroll: HISTORICAL_BANKROLL_START_DOLLARS + cumulativeProfit,
    bankroll_start_central: HISTORICAL_BANKROLL_START_DATE_CENTRAL,
    coverage_start_central: replayCoverageStart,
    coverage_end_central: dataset.coverage_end_central,
    note: buildNote(
      dataset.total_final_games,
      analyzedGames,
      bets.length,
      dataset.stored_forecast_games,
      dataset.diagnostic_forecast_games
    ),
  };

  return {
    summary,
    daily_points: dailyPoints,
    bets,
  };
}

function selectDefaultHistoricalStrategy(
  league: LeagueCode,
  strategies: Record<BetStrategy, BetHistoryStrategyBundle>
): BetStrategy {
  const leagueDefault = getDefaultBetStrategyForLeague(league) || DEFAULT_BET_STRATEGY;
  if (league !== "NBA") {
    return leagueDefault;
  }

  const conservativeProfit = strategies.capitalPreservation.summary.total_profit;
  const defaultProfit = strategies[leagueDefault].summary.total_profit;
  return conservativeProfit > defaultProfit ? "capitalPreservation" : leagueDefault;
}

export function getBetHistory(league: LeagueCode): BetHistoryResponse {
  const dataset = buildHistoricalReplayDataset(league);
  const strategies = BET_STRATEGIES.reduce((strategyAcc, strategy) => {
    strategyAcc[strategy] = buildBetHistoryStrategyBundle(dataset, strategy);
    return strategyAcc;
  }, {} as Record<BetStrategy, BetHistoryStrategyBundle>);
  const defaultStrategy = selectDefaultHistoricalStrategy(league, strategies);

  return {
    league,
    default_strategy: defaultStrategy,
    strategy_configs: dataset.strategy_configs,
    strategy_optimization: dataset.strategy_optimization,
    strategies,
  };
}
