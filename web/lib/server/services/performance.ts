import { DEFAULT_BET_STRATEGY } from "@/lib/betting-strategy";
import { runSqlJson } from "@/lib/db";
import { type LeagueCode } from "@/lib/league";
import { canonicalizePredictionModel } from "@/lib/predictions-report";
import {
  buildModelReplayRuns,
  type ModelReplayCandidateRow,
  type ModelReplayRunMetadata,
} from "@/lib/model-version-replay";
import type {
  ChangePointRow,
  ModelReplayRunRow,
  ModelRunSummaryRow,
  PerformanceResponse,
  PerformanceScoreRow,
  TableRow,
} from "@/lib/types";
import type { ModelWinProbabilities } from "@/lib/betting-model";

type RawScoreRow = {
  model_name?: string | null;
  game_date_utc?: string | null;
  log_loss?: number | null;
};

type RawRunSummaryRow = {
  model_name?: string | null;
  model_run_id?: string | null;
  run_type?: string | null;
  created_at_utc?: string | null;
  snapshot_id?: string | null;
  feature_set_version?: string | null;
  first_game_date_utc?: string | null;
  last_game_date_utc?: string | null;
  n_games?: number | null;
  avg_log_loss?: number | null;
  avg_brier?: number | null;
  accuracy?: number | null;
  version_rank?: number | null;
  is_latest_version?: number | null;
};

type RawReplayGameRow = {
  game_id?: number | null;
  game_date_utc?: string | null;
  start_time_utc?: string | null;
  final_utc?: string | null;
  home_team?: string | null;
  away_team?: string | null;
  home_score?: number | null;
  away_score?: number | null;
  home_win?: number | null;
  odds_snapshot_id?: string | null;
  odds_as_of_utc?: string | null;
};

type RawForecastRow = {
  game_id?: number | null;
  model_name?: string | null;
  model_run_id?: string | null;
  forecast_as_of_utc?: string | null;
  prob_home_win?: number | null;
};

type RawMoneylineRow = {
  odds_snapshot_id?: string | null;
  game_id?: number | null;
  home_team?: string | null;
  away_team?: string | null;
  home_moneyline?: number | null;
  away_moneyline?: number | null;
};

type RawRunMetadataRow = {
  model_run_id?: string | null;
  model_name?: string | null;
  run_type?: string | null;
  created_at_utc?: string | null;
  snapshot_id?: string | null;
  feature_set_version?: string | null;
  artifact_path?: string | null;
  params_json?: string | null;
  metrics_json?: string | null;
  feature_columns_json?: string | null;
  feature_metadata_json?: string | null;
};

type ReplayBaseGame = {
  game_id: number;
  date_central: string;
  start_time_utc: string | null;
  final_utc: string | null;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  home_win: number | null;
  odds_snapshot_id: string;
  odds_as_of_utc: string;
  home_moneyline: number;
  away_moneyline: number;
};

function escapeSqlString(value: string): string {
  return value.replace(/'/g, "''");
}

function numberOrNull(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function parseJsonRecord(value?: string | null): TableRow | null {
  if (!value) return null;
  try {
    const parsed = JSON.parse(value) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as TableRow;
    }
  } catch {}
  return null;
}

function parseJsonStringArray(value?: string | null): string[] {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value) as unknown;
    if (Array.isArray(parsed)) {
      return parsed.map((entry) => String(entry || "").trim()).filter(Boolean);
    }
  } catch {}
  return [];
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

function dateKeyForReplayRow(row: Pick<RawReplayGameRow, "start_time_utc" | "game_date_utc" | "final_utc">): string | null {
  const fromStart = centralDateKeyFromTimestamp(row.start_time_utc);
  if (fromStart) return fromStart;

  const fromFinal = centralDateKeyFromTimestamp(row.final_utc);
  if (fromFinal) return fromFinal;

  const gameDate = String(row.game_date_utc || "").trim();
  if (/^\d{4}-\d{2}-\d{2}$/.test(gameDate)) return gameDate;
  return gameDate.length >= 10 ? gameDate.slice(0, 10) : null;
}

function snapshotGameKey(snapshotId: string, gameId: number): string {
  return `${snapshotId}::${gameId}`;
}

function snapshotTeamKey(snapshotId: string, homeTeam: string, awayTeam: string): string {
  return `${snapshotId}::${homeTeam}::${awayTeam}`;
}

function historicalReplayGamesSql(league: LeagueCode): string {
  const escapedLeague = escapeSqlString(league);

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
    ORDER BY DATETIME(fg.replay_cutoff_utc) ASC, fg.game_id ASC
  `;
}

function historicalForecastCandidatesSql(): string {
  const diagnosticSource = "train_oof_history";

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
      SELECT
        fg.game_id,
        p.as_of_utc,
        p.prob_home_win,
        p.model_run_id,
        p.model_name,
        0 AS source_priority,
        p.prediction_id AS row_sort_id,
        COALESCE(mr.created_at_utc, p.as_of_utc) AS created_at_utc
      FROM finalized_games fg
      JOIN predictions p
        ON p.game_id = fg.game_id
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
        1 AS source_priority,
        p.diagnostic_id AS row_sort_id,
        p.as_of_utc AS created_at_utc
      FROM finalized_games fg
      JOIN prediction_diagnostics p
        ON p.game_id = fg.game_id
       AND COALESCE(json_extract(p.metadata_json, '$.source'), '') = '${diagnosticSource}'
       AND DATETIME(p.as_of_utc) <= DATETIME(fg.replay_cutoff_utc)
    ),
    ranked_forecasts AS (
      SELECT
        game_id,
        CASE
          WHEN model_name = 'glm_logit' THEN 'glm_ridge'
          ELSE model_name
        END AS model_name,
        model_run_id,
        as_of_utc,
        prob_home_win,
        ROW_NUMBER() OVER (
          PARTITION BY
            game_id,
            CASE
              WHEN model_name = 'glm_logit' THEN 'glm_ridge'
              ELSE model_name
            END
          ORDER BY
            source_priority ASC,
            DATETIME(as_of_utc) DESC,
            DATETIME(created_at_utc) DESC,
            row_sort_id DESC
        ) AS rn
      FROM forecast_candidates
    )
    SELECT
      game_id,
      model_name,
      model_run_id,
      as_of_utc AS forecast_as_of_utc,
      prob_home_win
    FROM ranked_forecasts
    WHERE rn = 1
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
    FROM ranked
    GROUP BY odds_snapshot_id, game_id, home_team, away_team
  `;
}

function queryScores(league: LeagueCode): PerformanceScoreRow[] {
  return (runSqlJson(
    `SELECT
       CASE
         WHEN model_name = 'glm_logit' THEN 'glm_ridge'
         ELSE model_name
       END AS model_name,
       game_date_utc,
       AVG(log_loss) AS log_loss
     FROM model_scores
     GROUP BY
       CASE
         WHEN model_name = 'glm_logit' THEN 'glm_ridge'
         ELSE model_name
       END,
       game_date_utc
     ORDER BY game_date_utc ASC`,
    { league }
  ) as RawScoreRow[]).map((row) => ({
    model_name: canonicalizePredictionModel(String(row.model_name || "")),
    game_date_utc: String(row.game_date_utc || ""),
    log_loss: numberOrNull(row.log_loss) ?? 0,
  }));
}

function queryRunSummaries(league: LeagueCode): ModelRunSummaryRow[] {
  return (runSqlJson(
    `WITH canonical_scores AS (
       SELECT
         CASE
           WHEN model_name = 'glm_logit' THEN 'glm_ridge'
           ELSE model_name
         END AS model_name,
         model_run_id,
         game_date_utc,
         log_loss,
         brier,
         accuracy
       FROM model_scores
       WHERE model_run_id IS NOT NULL
         AND TRIM(model_run_id) <> ''
     ),
     canonical_runs AS (
       SELECT
         model_run_id,
         CASE
           WHEN model_name = 'glm_logit' THEN 'glm_ridge'
           ELSE model_name
         END AS model_name,
         run_type,
         created_at_utc,
         snapshot_id,
         feature_set_version
       FROM model_runs
     ),
     summarized AS (
       SELECT
         cs.model_name,
         cs.model_run_id,
         cr.run_type,
         cr.created_at_utc,
         cr.snapshot_id,
         cr.feature_set_version,
         MIN(cs.game_date_utc) AS first_game_date_utc,
         MAX(cs.game_date_utc) AS last_game_date_utc,
         COUNT(*) AS n_games,
         AVG(cs.log_loss) AS avg_log_loss,
         AVG(cs.brier) AS avg_brier,
         AVG(cs.accuracy) AS accuracy
       FROM canonical_scores cs
       LEFT JOIN canonical_runs cr
         ON cr.model_run_id = cs.model_run_id
       GROUP BY
         cs.model_name,
         cs.model_run_id,
         cr.run_type,
         cr.created_at_utc,
         cr.snapshot_id,
         cr.feature_set_version
     ),
     ranked AS (
       SELECT
         *,
         ROW_NUMBER() OVER (
           PARTITION BY model_name
           ORDER BY
             DATETIME(COALESCE(created_at_utc, last_game_date_utc || 'T23:59:59Z')) DESC,
             model_run_id DESC
         ) AS version_rank
       FROM summarized
     )
     SELECT
       model_name,
       model_run_id,
       run_type,
       created_at_utc,
       snapshot_id,
       feature_set_version,
       first_game_date_utc,
       last_game_date_utc,
       n_games,
       avg_log_loss,
       avg_brier,
       accuracy,
       version_rank,
       CASE WHEN version_rank = 1 THEN 1 ELSE 0 END AS is_latest_version
     FROM ranked
     ORDER BY
       DATETIME(COALESCE(created_at_utc, last_game_date_utc || 'T23:59:59Z')) DESC,
       model_name ASC,
       avg_log_loss ASC`,
    { league }
  ) as RawRunSummaryRow[]).map((row) => ({
    model_name: canonicalizePredictionModel(String(row.model_name || "")),
    model_run_id: String(row.model_run_id || ""),
    run_type: row.run_type ? String(row.run_type) : null,
    created_at_utc: row.created_at_utc ? String(row.created_at_utc) : null,
    snapshot_id: row.snapshot_id ? String(row.snapshot_id) : null,
    feature_set_version: row.feature_set_version ? String(row.feature_set_version) : null,
    first_game_date_utc: row.first_game_date_utc ? String(row.first_game_date_utc) : null,
    last_game_date_utc: row.last_game_date_utc ? String(row.last_game_date_utc) : null,
    n_games: numberOrNull(row.n_games) ?? 0,
    avg_log_loss: numberOrNull(row.avg_log_loss) ?? 0,
    avg_brier: numberOrNull(row.avg_brier) ?? 0,
    accuracy: numberOrNull(row.accuracy) ?? 0,
    version_rank: numberOrNull(row.version_rank) ?? 0,
    is_latest_version: numberOrNull(row.is_latest_version) ?? 0,
  }));
}

function queryChangePoints(league: LeagueCode): ChangePointRow[] {
  return (runSqlJson(
    `WITH latest AS (
       SELECT MAX(as_of_utc) AS as_of_utc
       FROM change_points
     )
     SELECT model_name, metric_name, method, statistic, threshold, details_json, as_of_utc
     FROM change_points
     WHERE as_of_utc = (SELECT as_of_utc FROM latest)
     ORDER BY statistic DESC, model_name ASC
     LIMIT 120`,
    { league }
  ) as ChangePointRow[]).map((row) => ({
    ...row,
    model_name: canonicalizePredictionModel(String(row.model_name || "")),
  }));
}

function queryReplayBaseGames(league: LeagueCode): ReplayBaseGame[] {
  const rawGames = runSqlJson(historicalReplayGamesSql(league), { league }) as RawReplayGameRow[];
  const snapshotIds = Array.from(new Set(rawGames.map((row) => String(row.odds_snapshot_id || "").trim()).filter(Boolean)));
  const moneylineRows = snapshotIds.length
    ? (runSqlJson(moneylineSql(snapshotIds), { league }) as RawMoneylineRow[])
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

  const baseGames: ReplayBaseGame[] = [];
  for (const row of rawGames) {
    const gameId = numberOrNull(row.game_id);
    const oddsSnapshotId = String(row.odds_snapshot_id || "").trim();
    const oddsAsOf = String(row.odds_as_of_utc || "").trim();
    const homeTeam = String(row.home_team || "").trim();
    const awayTeam = String(row.away_team || "").trim();
    const dateCentral = dateKeyForReplayRow(row);

    if (gameId === null || !oddsSnapshotId || !oddsAsOf || !homeTeam || !awayTeam || !dateCentral) {
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

    baseGames.push({
      game_id: gameId,
      date_central: dateCentral,
      start_time_utc: row.start_time_utc ? String(row.start_time_utc) : null,
      final_utc: row.final_utc ? String(row.final_utc) : null,
      home_team: homeTeam,
      away_team: awayTeam,
      home_score: numberOrNull(row.home_score),
      away_score: numberOrNull(row.away_score),
      home_win: numberOrNull(row.home_win),
      odds_snapshot_id: oddsSnapshotId,
      odds_as_of_utc: oddsAsOf,
      home_moneyline: homeMoneyline,
      away_moneyline: awayMoneyline,
    });
  }

  return baseGames;
}

function queryReplayForecastRows(league: LeagueCode): RawForecastRow[] {
  return runSqlJson(historicalForecastCandidatesSql(), { league }) as RawForecastRow[];
}

function queryRunMetadata(league: LeagueCode, runIds: string[]): Map<string, ModelReplayRunMetadata> {
  if (!runIds.length) return new Map();

  const inList = runIds.map((runId) => `'${escapeSqlString(runId)}'`).join(", ");
  const rows = runSqlJson(
    `SELECT
       mr.model_run_id,
       CASE
         WHEN mr.model_name = 'glm_logit' THEN 'glm_ridge'
         ELSE mr.model_name
       END AS model_name,
       mr.run_type,
       mr.created_at_utc,
       mr.snapshot_id,
       mr.feature_set_version,
       mr.artifact_path,
       mr.params_json,
       mr.metrics_json,
       fs.feature_columns_json,
       fs.metadata_json AS feature_metadata_json
     FROM model_runs mr
     LEFT JOIN feature_sets fs
       ON fs.feature_set_version = mr.feature_set_version
     WHERE mr.model_run_id IN (${inList})`,
    { league }
  ) as RawRunMetadataRow[];

  const out = new Map<string, ModelReplayRunMetadata>();
  for (const row of rows) {
    const modelRunId = String(row.model_run_id || "").trim();
    if (!modelRunId) continue;
    out.set(modelRunId, {
      model_name: canonicalizePredictionModel(String(row.model_name || "")),
      model_run_id: modelRunId,
      run_type: row.run_type ? String(row.run_type) : null,
      created_at_utc: row.created_at_utc ? String(row.created_at_utc) : null,
      snapshot_id: row.snapshot_id ? String(row.snapshot_id) : null,
      feature_set_version: row.feature_set_version ? String(row.feature_set_version) : null,
      artifact_path: row.artifact_path ? String(row.artifact_path) : null,
      feature_columns: parseJsonStringArray(row.feature_columns_json),
      feature_metadata: parseJsonRecord(row.feature_metadata_json),
      params: parseJsonRecord(row.params_json),
      metrics: parseJsonRecord(row.metrics_json),
    });
  }

  return out;
}

function buildReplayCandidates(league: LeagueCode, runSummaries: ModelRunSummaryRow[]): ModelReplayRunRow[] {
  const baseGames = queryReplayBaseGames(league);
  if (!baseGames.length) return [];

  const forecastRows = queryReplayForecastRows(league);
  if (!forecastRows.length) return [];

  const baseGamesById = new Map(baseGames.map((row) => [row.game_id, row]));
  const modelProbabilitiesByGame = new Map<number, ModelWinProbabilities>();
  for (const row of forecastRows) {
    const gameId = numberOrNull(row.game_id);
    const modelName = canonicalizePredictionModel(String(row.model_name || "").trim());
    const probability = numberOrNull(row.prob_home_win);
    if (gameId === null || !modelName) continue;

    const current = modelProbabilitiesByGame.get(gameId) || {};
    current[modelName] = probability;
    modelProbabilitiesByGame.set(gameId, current);
  }

  const runSummaryById = new Map(runSummaries.map((row) => [row.model_run_id, row]));
  const candidates: ModelReplayCandidateRow[] = [];

  for (const row of forecastRows) {
    const gameId = numberOrNull(row.game_id);
    const modelRunId = String(row.model_run_id || "").trim();
    const modelName = canonicalizePredictionModel(String(row.model_name || "").trim());
    const forecastAsOf = String(row.forecast_as_of_utc || "").trim();
    const homeWinProbability = numberOrNull(row.prob_home_win);
    if (gameId === null || !modelRunId || !modelName || !forecastAsOf || homeWinProbability === null) continue;

    const baseGame = baseGamesById.get(gameId);
    if (!baseGame) continue;

    candidates.push({
      game_id: gameId,
      date_central: baseGame.date_central,
      start_time_utc: baseGame.start_time_utc,
      final_utc: baseGame.final_utc,
      home_team: baseGame.home_team,
      away_team: baseGame.away_team,
      home_score: baseGame.home_score,
      away_score: baseGame.away_score,
      home_win: baseGame.home_win,
      forecast_as_of_utc: forecastAsOf,
      home_moneyline: baseGame.home_moneyline,
      away_moneyline: baseGame.away_moneyline,
      model_name: modelName,
      model_run_id: modelRunId,
      home_win_probability: homeWinProbability,
      model_win_probabilities: modelProbabilitiesByGame.get(gameId) || { [modelName]: homeWinProbability },
    });
  }

  const runIds = Array.from(new Set(candidates.map((row) => row.model_run_id)));
  const runMetadataById = queryRunMetadata(league, runIds);
  return buildModelReplayRuns(candidates, runMetadataById, runSummaryById);
}

export function getPerformancePayload(league: LeagueCode): PerformanceResponse {
  const scores = queryScores(league);
  const run_summaries = queryRunSummaries(league);
  const change_points = queryChangePoints(league);
  const replay_runs = buildReplayCandidates(league, run_summaries);

  return {
    league,
    scores,
    run_summaries,
    change_points,
    replay_runs,
    default_replay_strategy: DEFAULT_BET_STRATEGY,
    comparison_replay_strategy: "aggressive",
  };
}
