export const STORED_FORECAST_SOURCE = "stored_prediction_history";
export const DIAGNOSTIC_FORECAST_SOURCE = "train_oof_history";

type HistoricalForecastCandidatesSqlOptions = {
  modelName?: string;
  diagnosticForecastSource?: string;
  storedForecastSource?: string;
};

type HistoricalMoneylineSqlOptions = {
  includeBookmakerTitles?: boolean;
};

type HistoricalFinalizedGamesCteSqlOptions = {
  resultsAlias?: string;
  gamesAlias?: string;
};

type HistoricalReplayOddsSelectionSqlOptions = {
  league: string;
  finalizedGameAlias?: string;
  lineAlias?: string;
  replayCutoffSql?: string;
  gameMatchSql?: string;
  requireConditionSql?: string;
};

const EFFECTIVE_ODDS_MARKET_LINES_VIEW_NAME = "odds_market_lines_effective_as_of";

export function escapeSqlString(value: string): string {
  return value.replace(/'/g, "''");
}

export function historicalForecastCandidatesUnionSql(options: HistoricalForecastCandidatesSqlOptions = {}): string {
  const escapedDiagnosticSource = escapeSqlString(options.diagnosticForecastSource ?? DIAGNOSTIC_FORECAST_SOURCE);
  const escapedStoredForecastSource = escapeSqlString(options.storedForecastSource ?? STORED_FORECAST_SOURCE);
  const modelFilter = options.modelName ? `AND p.model_name = '${escapeSqlString(options.modelName)}'` : "";

  return `
    SELECT
      fg.game_id,
      p.as_of_utc,
      p.prob_home_win,
      p.model_run_id,
      p.model_name,
      '${escapedStoredForecastSource}' AS forecast_source,
      0 AS source_priority,
      p.prediction_id AS row_sort_id,
      COALESCE(mr.created_at_utc, p.as_of_utc) AS created_at_utc
    FROM finalized_games fg
    JOIN predictions p
      ON p.game_id = fg.game_id
     ${modelFilter}
     AND DATETIME(p.as_of_utc) <= DATETIME(fg.replay_cutoff_utc)
    LEFT JOIN model_runs mr
      ON mr.model_run_id = p.model_run_id
    WHERE DATETIME(COALESCE(mr.created_at_utc, p.as_of_utc)) <= DATETIME(fg.replay_cutoff_utc)

    UNION ALL

    SELECT
      fg.game_id,
      p.as_of_utc,
      p.prob_home_win,
      p.model_run_id,
      p.model_name,
      '${escapedDiagnosticSource}' AS forecast_source,
      1 AS source_priority,
      p.diagnostic_id AS row_sort_id,
      p.as_of_utc AS created_at_utc
    FROM finalized_games fg
    JOIN prediction_diagnostics p
      ON p.game_id = fg.game_id
     ${modelFilter}
     AND COALESCE(json_extract(p.metadata_json, '$.source'), '') = '${escapedDiagnosticSource}'
     AND DATETIME(p.as_of_utc) <= DATETIME(fg.replay_cutoff_utc)
  `;
}

export function effectiveReplayOddsAsOfSql(lineAlias: string): string {
  return `${lineAlias}.effective_odds_as_of_utc`;
}

export function historicalFinalizedGamesCteSql(options: HistoricalFinalizedGamesCteSqlOptions = {}): string {
  const resultsAlias = options.resultsAlias ?? "r";
  const gamesAlias = options.gamesAlias ?? "g";

  return `finalized_games AS (
      SELECT
        ${resultsAlias}.game_id,
        ${resultsAlias}.game_date_utc,
        ${gamesAlias}.start_time_utc,
        ${resultsAlias}.final_utc,
        ${resultsAlias}.home_team,
        ${resultsAlias}.away_team,
        ${resultsAlias}.home_score,
        ${resultsAlias}.away_score,
        ${resultsAlias}.home_win,
        COALESCE(${gamesAlias}.start_time_utc, ${resultsAlias}.final_utc, ${resultsAlias}.game_date_utc || 'T23:59:59Z') AS replay_cutoff_utc
      FROM results ${resultsAlias}
      LEFT JOIN games ${gamesAlias} ON ${gamesAlias}.game_id = ${resultsAlias}.game_id
      WHERE ${resultsAlias}.home_win IS NOT NULL
    )`;
}

export function historicalReplayOddsSelectionSql(options: HistoricalReplayOddsSelectionSqlOptions): string {
  const finalizedGameAlias = options.finalizedGameAlias ?? "fg";
  const lineAlias = options.lineAlias ?? "l";
  const replayCutoffSql = options.replayCutoffSql ?? `DATETIME(${finalizedGameAlias}.replay_cutoff_utc)`;
  const gameMatchSql =
    options.gameMatchSql ??
    `(
            (${lineAlias}.game_id IS NOT NULL AND ${lineAlias}.game_id = ${finalizedGameAlias}.game_id)
            OR (${lineAlias}.home_team = ${finalizedGameAlias}.home_team AND ${lineAlias}.away_team = ${finalizedGameAlias}.away_team)
          )`;
  const requireConditionSql = options.requireConditionSql ? `\n          AND ${options.requireConditionSql}` : "";
  const escapedLeague = escapeSqlString(options.league);
  const effectiveOddsAsOf = effectiveReplayOddsAsOfSql(lineAlias);

  return `(
        SELECT ${lineAlias}.odds_snapshot_id
        FROM ${EFFECTIVE_ODDS_MARKET_LINES_VIEW_NAME} ${lineAlias}
        WHERE ${lineAlias}.league = '${escapedLeague}'
          AND ${lineAlias}.market_key = 'h2h'
          AND ${gameMatchSql}${requireConditionSql}
          AND DATETIME(${effectiveOddsAsOf}) <= ${replayCutoffSql}
        ORDER BY DATETIME(${effectiveOddsAsOf}) DESC, DATETIME(${lineAlias}.snapshot_as_of_utc) DESC, ${lineAlias}.line_id DESC
        LIMIT 1
      ) AS odds_snapshot_id,
      (
        SELECT ${effectiveOddsAsOf}
        FROM ${EFFECTIVE_ODDS_MARKET_LINES_VIEW_NAME} ${lineAlias}
        WHERE ${lineAlias}.league = '${escapedLeague}'
          AND ${lineAlias}.market_key = 'h2h'
          AND ${gameMatchSql}${requireConditionSql}
          AND DATETIME(${effectiveOddsAsOf}) <= ${replayCutoffSql}
        ORDER BY DATETIME(${effectiveOddsAsOf}) DESC, DATETIME(${lineAlias}.snapshot_as_of_utc) DESC, ${lineAlias}.line_id DESC
        LIMIT 1
      ) AS odds_as_of_utc`;
}

export function historicalMoneylineSql(snapshotIds: string[], options: HistoricalMoneylineSqlOptions = {}): string {
  const inList = snapshotIds.map((snapshotId) => `'${escapeSqlString(snapshotId)}'`).join(", ") || "''";
  const includeBookmakerTitles = options.includeBookmakerTitles ?? false;

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
      MAX(CASE WHEN outcome_side = 'away' AND rn = 1 THEN outcome_price END) AS away_moneyline
      ${
        includeBookmakerTitles
          ? `,
      MAX(CASE WHEN outcome_side = 'home' AND rn = 1 THEN bookmaker_title END) AS home_moneyline_book,
      MAX(CASE WHEN outcome_side = 'away' AND rn = 1 THEN bookmaker_title END) AS away_moneyline_book`
          : ""
      }
    FROM ranked
    GROUP BY odds_snapshot_id, game_id, home_team, away_team
  `;
}
