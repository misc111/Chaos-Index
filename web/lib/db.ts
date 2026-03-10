import { execFileSync } from "node:child_process";
import { type LeagueCode, normalizeLeague } from "@/lib/league";
import { resolveDbPathForLeague } from "@/lib/server/manifests";

const envLeague = normalizeLeague(process.env.LEAGUE);

function dbPathForLeague(league: LeagueCode): string {
  return process.env.SPORTS_DB_PATH
    ? process.env.SPORTS_DB_PATH
    : league === "NBA"
      ? process.env.NBA_DB_PATH || resolveDbPathForLeague("NBA")
      : league === "NHL"
        ? process.env.NHL_DB_PATH || resolveDbPathForLeague("NHL")
        : process.env.NCAAM_DB_PATH || resolveDbPathForLeague("NCAAM");
}

export function runSqlJson<T extends Record<string, unknown> = Record<string, unknown>>(
  sql: string,
  opts?: { league?: LeagueCode }
): T[] {
  const dbPath = dbPathForLeague(opts?.league || envLeague);
  try {
    const out = execFileSync("sqlite3", ["-json", dbPath, sql], { encoding: "utf8" });
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
