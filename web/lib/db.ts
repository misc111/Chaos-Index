import { execFileSync } from "node:child_process";
import { LEAGUE_RUNTIME } from "@/lib/generated/league-registry";
import { type LeagueCode, normalizeLeague } from "@/lib/league";
import { resolveDbPathForLeague } from "@/lib/server/manifests";

const envLeague = normalizeLeague(process.env.LEAGUE);
const SQLITE_JSON_MAX_BUFFER_BYTES = 64 * 1024 * 1024;

function dbPathForLeague(league: LeagueCode): string {
  if (process.env.SPORTS_DB_PATH) {
    return process.env.SPORTS_DB_PATH;
  }
  const runtime = LEAGUE_RUNTIME[league];
  return process.env[runtime.dbEnvVar] || resolveDbPathForLeague(league);
}

export function runSqlJson<T extends Record<string, unknown> = Record<string, unknown>>(
  sql: string,
  opts?: { league?: LeagueCode }
): T[] {
  const dbPath = dbPathForLeague(opts?.league || envLeague);
  try {
    const out = execFileSync("sqlite3", ["-json", dbPath, sql], {
      encoding: "utf8",
      maxBuffer: SQLITE_JSON_MAX_BUFFER_BYTES,
    });
    return out.trim() ? (JSON.parse(out) as T[]) : [];
  } catch {
    return [];
  }
}

export function execSql(sql: string, opts?: { league?: LeagueCode }): string {
  const dbPath = dbPathForLeague(opts?.league || envLeague);
  return execFileSync("sqlite3", [dbPath, sql], { encoding: "utf8" });
}

const dbPath = dbPathForLeague(envLeague);
export { dbPath };
