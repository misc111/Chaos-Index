import { NextResponse } from "next/server";
import { runSqlJson } from "@/lib/db";
import { leagueFromRequest } from "@/lib/league";

type RawTodayGameRow = {
  game_id: number;
  game_date_utc?: string | null;
  home_team: string;
  away_team: string;
  home_win_probability: number;
  start_time_utc?: string | null;
};

type RawOddsLine = {
  game_id?: number | null;
  home_team?: string | null;
  away_team?: string | null;
  market_key?: string | null;
  outcome_side?: string | null;
  outcome_point?: number | null;
  outcome_price?: number | null;
  bookmaker_title?: string | null;
};

type RawTeamNameRow = {
  team_abbrev?: string | null;
  team_name?: string | null;
};

function escapeSqlString(value: string): string {
  return value.replace(/'/g, "''");
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

function centralTodayDateKey(): string {
  const now = new Date();
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Chicago",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(now);

  const year = parts.find((part) => part.type === "year")?.value;
  const month = parts.find((part) => part.type === "month")?.value;
  const day = parts.find((part) => part.type === "day")?.value;
  if (!year || !month || !day) {
    return now.toISOString().slice(0, 10);
  }
  return `${year}-${month}-${day}`;
}

function dateKeyForRow(row: Pick<RawTodayGameRow, "start_time_utc" | "game_date_utc">): string | null {
  const byStartTime = centralDateKeyFromTimestamp(row.start_time_utc);
  if (byStartTime) return byStartTime;

  const fallback = String(row.game_date_utc || "").trim();
  if (!fallback) return null;
  if (/^\d{4}-\d{2}-\d{2}$/.test(fallback)) return fallback;

  const byGameDateTimestamp = centralDateKeyFromTimestamp(fallback);
  if (byGameDateTimestamp) return byGameDateTimestamp;

  return fallback.length >= 10 ? fallback.slice(0, 10) : null;
}

function normalizeProbability(value: unknown): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0.5;
  return Math.max(0, Math.min(1, numeric));
}

function numericOrNull(value: unknown): number | null {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function pairKey(homeTeam?: string | null, awayTeam?: string | null): string {
  return `${String(homeTeam || "").trim()}|${String(awayTeam || "").trim()}`;
}

function pickBestPrice(rows: RawOddsLine[], side: string): { price: number | null; book: string | null } {
  let bestPrice: number | null = null;
  let bestBook: string | null = null;

  for (const row of rows) {
    if (String(row.outcome_side || "") !== side) continue;
    const price = numericOrNull(row.outcome_price);
    if (price === null) continue;
    if (bestPrice === null || price > bestPrice) {
      bestPrice = price;
      bestBook = row.bookmaker_title == null ? null : String(row.bookmaker_title);
    }
  }

  return { price: bestPrice, book: bestBook };
}

function weightedCenter(candidates: Array<{ point: number; weight: number }>): number {
  const totalWeight = candidates.reduce((sum, candidate) => sum + candidate.weight, 0);
  if (!totalWeight) return 0;
  const weightedSum = candidates.reduce((sum, candidate) => sum + candidate.point * candidate.weight, 0);
  return weightedSum / totalWeight;
}

function pickConsensusPoint(rows: RawOddsLine[], marketKey: "spreads" | "totals") {
  const grouped = new Map<
    string,
    { point: number; rows: RawOddsLine[]; books: Set<string>; sides: Set<string>; weight: number }
  >();

  for (const row of rows) {
    const pointRaw = numericOrNull(row.outcome_point);
    if (pointRaw === null) continue;
    const point = marketKey === "spreads" ? Math.abs(pointRaw) : pointRaw;
    const key = point.toFixed(1);
    const bucket = grouped.get(key) || {
      point,
      rows: [],
      books: new Set<string>(),
      sides: new Set<string>(),
      weight: 0,
    };

    bucket.rows.push(row);
    bucket.weight += 1;
    const book = String(row.bookmaker_title || "").trim();
    if (book) bucket.books.add(book);
    const side = String(row.outcome_side || "").trim();
    if (side) bucket.sides.add(side);
    grouped.set(key, bucket);
  }

  const candidates = Array.from(grouped.values());
  if (!candidates.length) return null;

  const center = weightedCenter(candidates.map((candidate) => ({ point: candidate.point, weight: candidate.weight })));
  candidates.sort(
    (left, right) =>
      right.books.size - left.books.size ||
      right.sides.size - left.sides.size ||
      Math.abs(left.point - center) - Math.abs(right.point - center) ||
      left.point - right.point
  );
  return candidates[0];
}

function pickMoneylineBoard(rows: RawOddsLine[]) {
  const away = pickBestPrice(rows, "away");
  const home = pickBestPrice(rows, "home");
  const books = new Set(
    rows
      .map((row) => String(row.bookmaker_title || "").trim())
      .filter((value) => value.length > 0)
  );
  return {
    away_price: away.price,
    home_price: home.price,
    away_book: away.book,
    home_book: home.book,
    books_count: books.size,
  };
}

function pickSpreadBoard(rows: RawOddsLine[]) {
  const selected = pickConsensusPoint(rows, "spreads");
  if (!selected) {
    return {
      point: null,
      away_price: null,
      home_price: null,
      away_book: null,
      home_book: null,
      books_count: 0,
    };
  }

  const away = pickBestPrice(selected.rows, "away");
  const home = pickBestPrice(selected.rows, "home");
  return {
    point: selected.point,
    away_price: away.price,
    home_price: home.price,
    away_book: away.book,
    home_book: home.book,
    books_count: selected.books.size,
  };
}

function pickTotalBoard(rows: RawOddsLine[]) {
  const selected = pickConsensusPoint(rows, "totals");
  if (!selected) {
    return {
      point: null,
      over_price: null,
      under_price: null,
      over_book: null,
      under_book: null,
      books_count: 0,
    };
  }

  const over = pickBestPrice(selected.rows, "over");
  const under = pickBestPrice(selected.rows, "under");
  return {
    point: selected.point,
    over_price: over.price,
    under_price: under.price,
    over_book: over.book,
    under_book: under.book,
    books_count: selected.books.size,
  };
}

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  const latest = runSqlJson("SELECT MAX(as_of_utc) AS as_of_utc FROM upcoming_game_forecasts", { league });
  const asOf = latest?.[0]?.as_of_utc;

  if (typeof asOf !== "string" || !asOf.trim()) {
    return NextResponse.json({ league, as_of_utc: null, odds_as_of_utc: null, date_central: centralTodayDateKey(), rows: [] });
  }

  const escapedAsOf = escapeSqlString(asOf);
  const rawRows = runSqlJson(
    `
    SELECT
      u.game_id,
      u.game_date_utc,
      u.home_team,
      u.away_team,
      u.ensemble_prob_home_win AS home_win_probability,
      g.start_time_utc
    FROM upcoming_game_forecasts u
    LEFT JOIN games g ON g.game_id = u.game_id
    WHERE u.as_of_utc = '${escapedAsOf}'
      AND COALESCE(g.status_final, 0) = 0
    ORDER BY
      CASE WHEN g.start_time_utc IS NULL THEN 1 ELSE 0 END,
      DATETIME(g.start_time_utc) ASC,
      u.game_date_utc ASC,
      u.game_id ASC
    `,
    { league }
  ) as RawTodayGameRow[];

  const todayKey = centralTodayDateKey();
  const rows = rawRows
    .map((row) => ({
      ...row,
      home_win_probability: normalizeProbability(row.home_win_probability),
    }))
    .filter((row) => dateKeyForRow(row) === todayKey);

  const latestOddsSnapshot = runSqlJson(
    `
    SELECT odds_snapshot_id, as_of_utc
    FROM odds_snapshots
    WHERE league = '${escapeSqlString(league)}'
    ORDER BY DATETIME(as_of_utc) DESC
    LIMIT 1
    `,
    { league }
  );

  const oddsSnapshotId =
    typeof latestOddsSnapshot?.[0]?.odds_snapshot_id === "string" ? latestOddsSnapshot[0].odds_snapshot_id : "";
  const oddsAsOfUtc = typeof latestOddsSnapshot?.[0]?.as_of_utc === "string" ? latestOddsSnapshot[0].as_of_utc : null;
  const escapedSnapshotId = oddsSnapshotId ? escapeSqlString(oddsSnapshotId) : "";

  const gameIds = rows.map((row) => Number(row.game_id)).filter((value) => Number.isFinite(value));
  const gameIdClause = gameIds.length ? `AND game_id IN (${gameIds.join(",")})` : "AND 1 = 0";

  const marketLines = escapedSnapshotId
    ? (runSqlJson(
        `
        SELECT
          game_id,
          home_team,
          away_team,
          market_key,
          outcome_side,
          outcome_point,
          outcome_price,
          bookmaker_title
        FROM odds_market_lines
        WHERE odds_snapshot_id = '${escapedSnapshotId}'
          AND market_key IN ('h2h', 'spreads', 'totals')
          AND outcome_price IS NOT NULL
          ${gameIdClause}
        ORDER BY game_id ASC, market_key ASC, bookmaker_title ASC
        `,
        { league }
      ) as RawOddsLine[])
    : [];

  const linesByGameId = new Map<number, RawOddsLine[]>();
  const linesByPairKey = new Map<string, RawOddsLine[]>();

  for (const line of marketLines) {
    const gameId = Number(line.game_id);
    if (Number.isFinite(gameId)) {
      const current = linesByGameId.get(gameId) || [];
      current.push(line);
      linesByGameId.set(gameId, current);
    }

    const key = pairKey(line.home_team, line.away_team);
    if (key !== "|") {
      const current = linesByPairKey.get(key) || [];
      current.push(line);
      linesByPairKey.set(key, current);
    }
  }

  const latestTeamsSnapshot = runSqlJson(
    `
    SELECT team_abbrev, team_name
    FROM teams
    WHERE league = '${escapeSqlString(league)}'
      AND as_of_utc = (
        SELECT MAX(as_of_utc)
        FROM teams
        WHERE league = '${escapeSqlString(league)}'
      )
    `,
    { league }
  ) as RawTeamNameRow[];

  const teamNames = new Map<string, string>();
  for (const row of latestTeamsSnapshot) {
    const abbrev = String(row.team_abbrev || "").trim();
    const name = String(row.team_name || "").trim();
    if (abbrev && name) {
      teamNames.set(abbrev, name);
    }
  }

  const enrichedRows = rows.map((row) => {
    const matchedLines =
      linesByGameId.get(Number(row.game_id)) ||
      linesByPairKey.get(pairKey(row.home_team, row.away_team)) ||
      [];
    const moneylineRows = matchedLines.filter((line) => line.market_key === "h2h");
    const spreadRows = matchedLines.filter((line) => line.market_key === "spreads");
    const totalRows = matchedLines.filter((line) => line.market_key === "totals");

    return {
      game_id: Number(row.game_id),
      game_date_utc: row.game_date_utc || null,
      start_time_utc: row.start_time_utc || null,
      home_team: row.home_team,
      away_team: row.away_team,
      home_team_name: teamNames.get(row.home_team) || row.home_team,
      away_team_name: teamNames.get(row.away_team) || row.away_team,
      home_win_probability: row.home_win_probability,
      moneyline: pickMoneylineBoard(moneylineRows),
      spread: pickSpreadBoard(spreadRows),
      total: pickTotalBoard(totalRows),
    };
  });

  return NextResponse.json({
    league,
    as_of_utc: asOf,
    odds_as_of_utc: oddsAsOfUtc,
    date_central: todayKey,
    rows: enrichedRows,
  });
}
