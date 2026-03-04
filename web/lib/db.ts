import { execFileSync } from "node:child_process";
import path from "node:path";

const defaultNhlDbPath = path.resolve(process.cwd(), "..", "data", "processed", "nhl_forecast.db");
const defaultNbaDbPath = path.resolve(process.cwd(), "..", "data", "processed", "nba_forecast.db");
const league = (process.env.LEAGUE || "NHL").toUpperCase();
const dbPath =
  process.env.SPORTS_DB_PATH ||
  (league === "NBA" ? process.env.NBA_DB_PATH || defaultNbaDbPath : process.env.NHL_DB_PATH || defaultNhlDbPath);

export function runSqlJson(sql: string): any[] {
  try {
    const out = execFileSync("sqlite3", ["-json", dbPath, sql], { encoding: "utf8" });
    return out.trim() ? JSON.parse(out) : [];
  } catch (err) {
    return [];
  }
}

export { dbPath };
