import { execFileSync } from "node:child_process";
import path from "node:path";
import { type LeagueCode, normalizeLeague } from "@/lib/league";

const defaultNhlDbPath = path.resolve(process.cwd(), "..", "data", "processed", "nhl_forecast.db");
const defaultNbaDbPath = path.resolve(process.cwd(), "..", "data", "processed", "nba_forecast.db");
const envLeague = normalizeLeague(process.env.LEAGUE);

function dbPathForLeague(league: LeagueCode): string {
  return process.env.SPORTS_DB_PATH
    ? process.env.SPORTS_DB_PATH
    : league === "NBA"
      ? process.env.NBA_DB_PATH || defaultNbaDbPath
      : process.env.NHL_DB_PATH || defaultNhlDbPath;
}

export function runSqlJson(sql: string, opts?: { league?: LeagueCode }): any[] {
  const dbPath = dbPathForLeague(opts?.league || envLeague);
  try {
    const out = execFileSync("sqlite3", ["-json", dbPath, sql], { encoding: "utf8" });
    return out.trim() ? JSON.parse(out) : [];
  } catch (err) {
    return [];
  }
}

export function execSql(sql: string, opts?: { league?: LeagueCode }): string {
  const dbPath = dbPathForLeague(opts?.league || envLeague);
  return execFileSync("sqlite3", [dbPath, sql], { encoding: "utf8" });
}

const dbPath = dbPathForLeague(envLeague);
export { dbPath };
