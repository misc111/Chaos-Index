import {
  type BetStrategyRuleConfig,
  type BetStrategy,
  DEFAULT_BET_STRATEGY,
} from "@/lib/betting-strategy";
import { REFERENCE_STAKE_DOLLARS, computeBetDecisionsForSlate, type BetDecision, type ExpectedSide } from "@/lib/betting";
import type { ModelWinProbabilities } from "@/lib/betting-model";
import { execSql, runSqlJson } from "@/lib/db";
import type { LeagueCode } from "@/lib/league";

const REPLAY_DECISION_VERSION = "historical_replay_v8";
const REPLAY_MATERIALIZATION_VERSION = "historical_prediction_history_v5";
const REPLAY_DECISION_TABLE = "historical_bet_decisions_by_profile";
const DEFAULT_REPLAY_VARIANT = "default";

export type ReplayableHistoricalGame = {
  game_id: number;
  league?: LeagueCode | null;
  date_central: string;
  home_team: string;
  away_team: string;
  forecast_as_of_utc: string;
  forecast_model_run_id: string | null;
  odds_as_of_utc: string;
  odds_snapshot_id: string;
  home_moneyline: number;
  away_moneyline: number;
  home_win_probability: number;
  betting_model_name?: string | null;
  model_win_probabilities?: ModelWinProbabilities | null;
};

type RawHistoricalReplayDecisionRow = {
  strategy?: string | null;
  sizing_style?: string | null;
  game_id?: number | null;
  date_central?: string | null;
  forecast_as_of_utc?: string | null;
  forecast_model_run_id?: string | null;
  odds_as_of_utc?: string | null;
  odds_snapshot_id?: string | null;
  home_team?: string | null;
  away_team?: string | null;
  home_win_probability?: number | null;
  home_moneyline?: number | null;
  away_moneyline?: number | null;
  bet_label?: string | null;
  reason?: string | null;
  side?: string | null;
  team?: string | null;
  stake?: number | null;
  odds?: number | null;
  model_probability?: number | null;
  market_probability?: number | null;
  edge?: number | null;
  expected_value?: number | null;
  stake_unit_dollars?: number | null;
  strategy_config_signature?: string | null;
  decision_logic_version?: string | null;
  materialization_version?: string | null;
  created_at_utc?: string | null;
};

export type HistoricalReplayDecisionSnapshot = {
  strategy: BetStrategy;
  game_id: number;
  date_central: string;
  forecast_as_of_utc: string;
  forecast_model_run_id: string | null;
  odds_as_of_utc: string;
  odds_snapshot_id: string;
  home_team: string;
  away_team: string;
  home_win_probability: number;
  home_moneyline: number;
  away_moneyline: number;
  bet_label: string;
  reason: string;
  side: ExpectedSide;
  team: string | null;
  stake: number;
  odds: number | null;
  model_probability: number | null;
  market_probability: number | null;
  edge: number | null;
  expected_value: number | null;
  reference_stake_dollars: number;
  strategy_config_signature: string;
  decision_logic_version: string;
  materialization_version: string | null;
  created_at_utc: string;
};

export type HistoricalReplayDecisionSet = Record<BetStrategy, HistoricalReplayDecisionSnapshot | null>;

type RawSqliteTableInfoRow = {
  name?: string | null;
};

function escapeSqlString(value: string): string {
  return value.replace(/'/g, "''");
}

function sqlText(value?: string | null): string {
  if (value === null || value === undefined) return "NULL";
  return `'${escapeSqlString(value)}'`;
}

function sqlNumber(value?: number | null): string {
  return typeof value === "number" && Number.isFinite(value) ? `${value}` : "NULL";
}

function numberOrNull(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function normalizeSide(value: unknown): ExpectedSide {
  const side = String(value || "").trim();
  if (side === "home" || side === "away") return side;
  return "none";
}

function normalizeStrategy(value: unknown): BetStrategy {
  const strategy = String(value || "").trim();
  if (strategy === "riskAdjusted" || strategy === "aggressive" || strategy === "capitalPreservation") {
    return strategy;
  }
  if (strategy === "balanced") return "riskAdjusted";
  if (strategy === "aggressiveEv" || strategy === "riskLoving") return "aggressive";
  if (strategy === "riskAverse") return "capitalPreservation";
  return DEFAULT_BET_STRATEGY;
}

function ensureHistoricalReplayDecisionTable(league: LeagueCode): void {
  execSql(
    `
    CREATE TABLE IF NOT EXISTS ${REPLAY_DECISION_TABLE} (
      strategy TEXT NOT NULL,
      sizing_style TEXT NOT NULL DEFAULT '${DEFAULT_REPLAY_VARIANT}',
      game_id INTEGER NOT NULL,
      date_central TEXT NOT NULL,
      forecast_as_of_utc TEXT NOT NULL,
      forecast_model_run_id TEXT,
      odds_as_of_utc TEXT NOT NULL,
      odds_snapshot_id TEXT NOT NULL,
      home_team TEXT NOT NULL,
      away_team TEXT NOT NULL,
      home_win_probability REAL NOT NULL,
      home_moneyline REAL NOT NULL,
      away_moneyline REAL NOT NULL,
      bet_label TEXT NOT NULL,
      reason TEXT NOT NULL,
      side TEXT NOT NULL,
      team TEXT,
      stake REAL NOT NULL,
      odds REAL,
      model_probability REAL,
      market_probability REAL,
      edge REAL,
      expected_value REAL,
      stake_unit_dollars REAL NOT NULL DEFAULT 100,
      strategy_config_signature TEXT,
      decision_logic_version TEXT NOT NULL,
      materialization_version TEXT,
      created_at_utc TEXT NOT NULL,
      PRIMARY KEY (strategy, sizing_style, game_id)
    );
    `,
    { league }
  );

  const columns = new Set(
    (runSqlJson(`PRAGMA table_info(${REPLAY_DECISION_TABLE})`, { league }) as RawSqliteTableInfoRow[])
      .map((row) => String(row.name || "").trim())
      .filter(Boolean)
  );

  const alterStatements: string[] = [];
  if (!columns.has("sizing_style")) {
    alterStatements.push(
      `ALTER TABLE ${REPLAY_DECISION_TABLE} ADD COLUMN sizing_style TEXT NOT NULL DEFAULT '${DEFAULT_REPLAY_VARIANT}';`
    );
  }
  if (!columns.has("forecast_model_run_id")) {
    alterStatements.push(`ALTER TABLE ${REPLAY_DECISION_TABLE} ADD COLUMN forecast_model_run_id TEXT;`);
  }
  if (!columns.has("materialization_version")) {
    alterStatements.push(`ALTER TABLE ${REPLAY_DECISION_TABLE} ADD COLUMN materialization_version TEXT;`);
  }
  if (!columns.has("stake_unit_dollars")) {
    alterStatements.push(
      `ALTER TABLE ${REPLAY_DECISION_TABLE} ADD COLUMN stake_unit_dollars REAL NOT NULL DEFAULT 100;`
    );
  }
  if (!columns.has("strategy_config_signature")) {
    alterStatements.push(`ALTER TABLE ${REPLAY_DECISION_TABLE} ADD COLUMN strategy_config_signature TEXT;`);
  }

  if (alterStatements.length) {
    execSql(alterStatements.join("\n"), { league });
  }

  execSql(
    `
    DELETE FROM ${REPLAY_DECISION_TABLE}
    WHERE sizing_style <> '${DEFAULT_REPLAY_VARIANT}'
      AND rowid NOT IN (
        SELECT MAX(rowid)
        FROM ${REPLAY_DECISION_TABLE}
        WHERE sizing_style <> '${DEFAULT_REPLAY_VARIANT}'
        GROUP BY strategy, game_id
      );

    DELETE FROM ${REPLAY_DECISION_TABLE}
    WHERE sizing_style <> '${DEFAULT_REPLAY_VARIANT}'
      AND EXISTS (
        SELECT 1
        FROM ${REPLAY_DECISION_TABLE} AS preferred
        WHERE preferred.strategy = ${REPLAY_DECISION_TABLE}.strategy
          AND preferred.game_id = ${REPLAY_DECISION_TABLE}.game_id
          AND preferred.sizing_style = '${DEFAULT_REPLAY_VARIANT}'
      );

    UPDATE ${REPLAY_DECISION_TABLE}
    SET sizing_style = '${DEFAULT_REPLAY_VARIANT}'
    WHERE sizing_style <> '${DEFAULT_REPLAY_VARIANT}';
    `,
    { league }
  );

  execSql(
    `
    CREATE INDEX IF NOT EXISTS idx_historical_bet_decisions_by_profile_date
      ON ${REPLAY_DECISION_TABLE}(strategy, sizing_style, date_central);
    `,
    { league }
  );
}

function loadStoredHistoricalReplayDecisions(
  gameIds: number[],
  league: LeagueCode,
  strategy: BetStrategy
): Map<number, HistoricalReplayDecisionSnapshot> {
  if (!gameIds.length) return new Map();

  const inList = Array.from(new Set(gameIds.filter((gameId) => Number.isFinite(gameId)))).join(", ");
  if (!inList) return new Map();

  const rows = runSqlJson(
    `
    SELECT
      strategy,
      sizing_style,
      game_id,
      date_central,
      forecast_as_of_utc,
      forecast_model_run_id,
      odds_as_of_utc,
      odds_snapshot_id,
      home_team,
      away_team,
      home_win_probability,
      home_moneyline,
      away_moneyline,
      bet_label,
      reason,
      side,
      team,
      stake,
      odds,
      model_probability,
      market_probability,
      edge,
      expected_value,
      stake_unit_dollars,
      strategy_config_signature,
      decision_logic_version,
      materialization_version,
      created_at_utc
    FROM ${REPLAY_DECISION_TABLE}
    WHERE strategy = '${escapeSqlString(strategy)}'
      AND sizing_style = '${DEFAULT_REPLAY_VARIANT}'
      AND game_id IN (${inList})
    `,
    { league }
  ) as RawHistoricalReplayDecisionRow[];

  const snapshots = new Map<number, HistoricalReplayDecisionSnapshot>();
  for (const row of rows) {
    const gameId = numberOrNull(row.game_id);
    if (gameId === null) continue;

    snapshots.set(gameId, {
      strategy: normalizeStrategy(row.strategy),
      game_id: gameId,
      date_central: String(row.date_central || ""),
      forecast_as_of_utc: String(row.forecast_as_of_utc || ""),
      forecast_model_run_id: row.forecast_model_run_id ? String(row.forecast_model_run_id) : null,
      odds_as_of_utc: String(row.odds_as_of_utc || ""),
      odds_snapshot_id: String(row.odds_snapshot_id || ""),
      home_team: String(row.home_team || ""),
      away_team: String(row.away_team || ""),
      home_win_probability: numberOrNull(row.home_win_probability) ?? 0.5,
      home_moneyline: numberOrNull(row.home_moneyline) ?? 0,
      away_moneyline: numberOrNull(row.away_moneyline) ?? 0,
      bet_label: String(row.bet_label || "$0"),
      reason: String(row.reason || "Price fair"),
      side: normalizeSide(row.side),
      team: row.team ? String(row.team) : null,
      stake: numberOrNull(row.stake) ?? 0,
      odds: numberOrNull(row.odds),
      model_probability: numberOrNull(row.model_probability),
      market_probability: numberOrNull(row.market_probability),
      edge: numberOrNull(row.edge),
      expected_value: numberOrNull(row.expected_value),
      reference_stake_dollars: numberOrNull(row.stake_unit_dollars) ?? 100,
      strategy_config_signature: String(row.strategy_config_signature || ""),
      decision_logic_version: String(row.decision_logic_version || REPLAY_DECISION_VERSION),
      materialization_version: row.materialization_version ? String(row.materialization_version) : null,
      created_at_utc: String(row.created_at_utc || ""),
    });
  }

  return snapshots;
}

function buildHistoricalReplayDecision(
  row: ReplayableHistoricalGame,
  decision: BetDecision,
  strategy: BetStrategy,
  strategyConfigSignature: string
): HistoricalReplayDecisionSnapshot {
  return {
    strategy,
    game_id: row.game_id,
    date_central: row.date_central,
    forecast_as_of_utc: row.forecast_as_of_utc,
    forecast_model_run_id: row.forecast_model_run_id,
    odds_as_of_utc: row.odds_as_of_utc,
    odds_snapshot_id: row.odds_snapshot_id,
    home_team: row.home_team,
    away_team: row.away_team,
    home_win_probability: row.home_win_probability,
    home_moneyline: row.home_moneyline,
    away_moneyline: row.away_moneyline,
    bet_label: decision.bet,
    reason: decision.reason,
    side: decision.side,
    team: decision.team,
    stake: decision.stake,
    odds: decision.odds,
    model_probability: decision.modelProbability,
    market_probability: decision.marketProbability,
    edge: decision.edge,
    expected_value: decision.expectedValue,
    reference_stake_dollars: REFERENCE_STAKE_DOLLARS,
    strategy_config_signature: strategyConfigSignature,
    decision_logic_version: REPLAY_DECISION_VERSION,
    materialization_version: REPLAY_MATERIALIZATION_VERSION,
    created_at_utc: new Date().toISOString(),
  };
}

function upsertHistoricalReplayDecisions(rows: HistoricalReplayDecisionSnapshot[], league: LeagueCode): void {
  if (!rows.length) return;

  const valuesSql = rows
    .map(
      (row) => `(
        ${sqlText(row.strategy)},
        ${sqlText(DEFAULT_REPLAY_VARIANT)},
        ${row.game_id},
        ${sqlText(row.date_central)},
        ${sqlText(row.forecast_as_of_utc)},
        ${sqlText(row.forecast_model_run_id)},
        ${sqlText(row.odds_as_of_utc)},
        ${sqlText(row.odds_snapshot_id)},
        ${sqlText(row.home_team)},
        ${sqlText(row.away_team)},
        ${sqlNumber(row.home_win_probability)},
        ${sqlNumber(row.home_moneyline)},
        ${sqlNumber(row.away_moneyline)},
        ${sqlText(row.bet_label)},
        ${sqlText(row.reason)},
        ${sqlText(row.side)},
        ${sqlText(row.team)},
        ${sqlNumber(row.stake)},
        ${sqlNumber(row.odds)},
        ${sqlNumber(row.model_probability)},
        ${sqlNumber(row.market_probability)},
        ${sqlNumber(row.edge)},
        ${sqlNumber(row.expected_value)},
        ${sqlNumber(row.reference_stake_dollars)},
        ${sqlText(row.strategy_config_signature)},
        ${sqlText(row.decision_logic_version)},
        ${sqlText(row.materialization_version)},
        ${sqlText(row.created_at_utc)}
      )`
    )
    .join(",\n");

  execSql(
    `
    INSERT OR REPLACE INTO ${REPLAY_DECISION_TABLE} (
      strategy,
      sizing_style,
      game_id,
      date_central,
      forecast_as_of_utc,
      forecast_model_run_id,
      odds_as_of_utc,
      odds_snapshot_id,
      home_team,
      away_team,
      home_win_probability,
      home_moneyline,
      away_moneyline,
      bet_label,
      reason,
      side,
      team,
      stake,
      odds,
      model_probability,
      market_probability,
      edge,
      expected_value,
      stake_unit_dollars,
      strategy_config_signature,
      decision_logic_version,
      materialization_version,
      created_at_utc
    ) VALUES ${valuesSql}
    `,
    { league }
  );
}

export function loadOrCreateHistoricalReplayDecisions(
  rows: ReplayableHistoricalGame[],
  league: LeagueCode,
  strategy: BetStrategy = DEFAULT_BET_STRATEGY,
  strategyConfig: BetStrategyRuleConfig,
  strategyConfigSignature: string
): Map<number, HistoricalReplayDecisionSnapshot> {
  if (!rows.length) return new Map();

  let snapshots = new Map<number, HistoricalReplayDecisionSnapshot>();
  let canPersist = true;

  try {
    ensureHistoricalReplayDecisionTable(league);
    snapshots = loadStoredHistoricalReplayDecisions(
      rows.map((row) => row.game_id),
      league,
      strategy
    );
  } catch {
    canPersist = false;
  }

  const pendingRows = rows
    .filter((row) => {
      const snapshot = snapshots.get(row.game_id);
      // Materialization version gates one-time repair of legacy rows while
      // still freezing the current replay decision against future retrains.
      return (
        !snapshot ||
        snapshot.decision_logic_version !== REPLAY_DECISION_VERSION ||
        snapshot.materialization_version !== REPLAY_MATERIALIZATION_VERSION ||
        snapshot.reference_stake_dollars !== REFERENCE_STAKE_DOLLARS ||
        snapshot.strategy_config_signature !== strategyConfigSignature
      );
    });

  const rowsToMaterialize: HistoricalReplayDecisionSnapshot[] = [];
  const pendingRowsByDate = new Map<string, ReplayableHistoricalGame[]>();
  for (const row of pendingRows) {
    const current = pendingRowsByDate.get(row.date_central) || [];
    current.push(row);
    pendingRowsByDate.set(row.date_central, current);
  }

  for (const dayRows of pendingRowsByDate.values()) {
    const decisions = computeBetDecisionsForSlate(
      dayRows.map((row) => ({
        home_team: row.home_team,
        away_team: row.away_team,
        home_win_probability: row.home_win_probability,
        home_moneyline: row.home_moneyline,
        away_moneyline: row.away_moneyline,
        betting_model_name: row.betting_model_name,
        model_win_probabilities: row.model_win_probabilities,
      })),
      strategy,
      strategyConfig
    );

    dayRows.forEach((row, index) => {
      rowsToMaterialize.push(
        buildHistoricalReplayDecision(row, decisions[index], strategy, strategyConfigSignature)
      );
    });
  }

  if (!rowsToMaterialize.length) {
    return snapshots;
  }

  if (canPersist) {
    try {
      upsertHistoricalReplayDecisions(rowsToMaterialize, league);
    } catch {
      // Falling back to in-memory snapshots still keeps the current request stable.
    }
  }

  for (const row of rowsToMaterialize) {
    snapshots.set(row.game_id, row);
  }

  return snapshots;
}
