import { computeBetDecision, type ExpectedSide } from "@/lib/betting";
import { execSql, runSqlJson } from "@/lib/db";
import type { LeagueCode } from "@/lib/league";

const REPLAY_DECISION_VERSION = "historical_replay_v1";

export type ReplayableHistoricalGame = {
  game_id: number;
  date_central: string;
  home_team: string;
  away_team: string;
  forecast_as_of_utc: string;
  odds_as_of_utc: string;
  odds_snapshot_id: string;
  home_moneyline: number;
  away_moneyline: number;
  home_win_probability: number;
};

type RawHistoricalReplayDecisionRow = {
  game_id?: number | null;
  date_central?: string | null;
  forecast_as_of_utc?: string | null;
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
  decision_logic_version?: string | null;
  created_at_utc?: string | null;
};

export type HistoricalReplayDecisionSnapshot = {
  game_id: number;
  date_central: string;
  forecast_as_of_utc: string;
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
  decision_logic_version: string;
  created_at_utc: string;
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

function ensureHistoricalReplayDecisionTable(league: LeagueCode): void {
  // Historical bet rows should stop drifting once a final game's pregame
  // recommendation has been materialized for the first time.
  execSql(
    `
    CREATE TABLE IF NOT EXISTS historical_bet_decisions (
      game_id INTEGER PRIMARY KEY,
      date_central TEXT NOT NULL,
      forecast_as_of_utc TEXT NOT NULL,
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
      decision_logic_version TEXT NOT NULL,
      created_at_utc TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_historical_bet_decisions_date
      ON historical_bet_decisions(date_central);
    `,
    { league }
  );
}

function loadStoredHistoricalReplayDecisions(
  gameIds: number[],
  league: LeagueCode
): Map<number, HistoricalReplayDecisionSnapshot> {
  if (!gameIds.length) return new Map();

  const inList = Array.from(new Set(gameIds.filter((gameId) => Number.isFinite(gameId)))).join(", ");
  if (!inList) return new Map();

  const rows = runSqlJson(
    `
    SELECT
      game_id,
      date_central,
      forecast_as_of_utc,
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
      decision_logic_version,
      created_at_utc
    FROM historical_bet_decisions
    WHERE game_id IN (${inList})
    `,
    { league }
  ) as RawHistoricalReplayDecisionRow[];

  const snapshots = new Map<number, HistoricalReplayDecisionSnapshot>();
  for (const row of rows) {
    const gameId = numberOrNull(row.game_id);
    if (gameId === null) continue;

    snapshots.set(gameId, {
      game_id: gameId,
      date_central: String(row.date_central || ""),
      forecast_as_of_utc: String(row.forecast_as_of_utc || ""),
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
      decision_logic_version: String(row.decision_logic_version || REPLAY_DECISION_VERSION),
      created_at_utc: String(row.created_at_utc || ""),
    });
  }

  return snapshots;
}

function buildHistoricalReplayDecision(row: ReplayableHistoricalGame): HistoricalReplayDecisionSnapshot {
  const decision = computeBetDecision({
    home_team: row.home_team,
    away_team: row.away_team,
    home_win_probability: row.home_win_probability,
    home_moneyline: row.home_moneyline,
    away_moneyline: row.away_moneyline,
  });

  return {
    game_id: row.game_id,
    date_central: row.date_central,
    forecast_as_of_utc: row.forecast_as_of_utc,
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
    decision_logic_version: REPLAY_DECISION_VERSION,
    created_at_utc: new Date().toISOString(),
  };
}

function insertHistoricalReplayDecisions(rows: HistoricalReplayDecisionSnapshot[], league: LeagueCode): void {
  if (!rows.length) return;

  const valuesSql = rows
    .map(
      (row) => `(
        ${row.game_id},
        ${sqlText(row.date_central)},
        ${sqlText(row.forecast_as_of_utc)},
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
        ${sqlText(row.decision_logic_version)},
        ${sqlText(row.created_at_utc)}
      )`
    )
    .join(",\n");

  execSql(
    `
    INSERT OR IGNORE INTO historical_bet_decisions (
      game_id,
      date_central,
      forecast_as_of_utc,
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
      decision_logic_version,
      created_at_utc
    ) VALUES ${valuesSql}
    `,
    { league }
  );
}

export function loadOrCreateHistoricalReplayDecisions(
  rows: ReplayableHistoricalGame[],
  league: LeagueCode
): Map<number, HistoricalReplayDecisionSnapshot> {
  if (!rows.length) return new Map();

  let snapshots = new Map<number, HistoricalReplayDecisionSnapshot>();
  let canPersist = true;

  try {
    ensureHistoricalReplayDecisionTable(league);
    snapshots = loadStoredHistoricalReplayDecisions(
      rows.map((row) => row.game_id),
      league
    );
  } catch {
    canPersist = false;
  }

  const missing = rows.filter((row) => !snapshots.has(row.game_id)).map(buildHistoricalReplayDecision);
  if (!missing.length) {
    return snapshots;
  }

  if (canPersist) {
    try {
      insertHistoricalReplayDecisions(missing, league);
    } catch {
      // Falling back to in-memory snapshots still keeps the current request stable.
    }
  }

  for (const row of missing) {
    snapshots.set(row.game_id, row);
  }

  return snapshots;
}
