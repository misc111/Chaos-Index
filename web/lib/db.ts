import { execFileSync } from "node:child_process";
import path from "node:path";

const defaultDbPath = path.resolve(process.cwd(), "..", "data", "processed", "nhl_forecast.db");
const dbPath = process.env.NHL_DB_PATH || defaultDbPath;

export function runSqlJson(sql: string): any[] {
  try {
    const out = execFileSync("sqlite3", ["-json", dbPath, sql], { encoding: "utf8" });
    return out.trim() ? JSON.parse(out) : [];
  } catch (err) {
    return [];
  }
}

export { dbPath };
