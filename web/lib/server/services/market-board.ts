import { getHistoricalReplayGames } from "@/lib/bet-history";
import { centralTodayDateKey, dateKeyForScheduledGame } from "@/lib/games-today";
import { type LeagueCode } from "@/lib/league";
import { getLatestUpcomingAsOf, getLatestTeamNames, getScheduledTodayRows } from "@/lib/server/repositories/forecasts";
import { getLatestOddsSnapshot, getMarketLinesForSnapshot, type RawOddsLine } from "@/lib/server/repositories/odds";

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
  const grouped = new Map<string, { point: number; rows: RawOddsLine[]; books: Set<string>; sides: Set<string>; weight: number }>();
  for (const row of rows) {
    const pointRaw = numericOrNull(row.outcome_point);
    if (pointRaw === null) continue;
    const point = marketKey === "spreads" ? Math.abs(pointRaw) : pointRaw;
    const key = point.toFixed(1);
    const bucket = grouped.get(key) || { point, rows: [], books: new Set<string>(), sides: new Set<string>(), weight: 0 };
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
  const books = new Set(rows.map((row) => String(row.bookmaker_title || "").trim()).filter((value) => value.length > 0));
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
    return { point: null, away_price: null, home_price: null, away_book: null, home_book: null, books_count: 0 };
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
    return { point: null, over_price: null, under_price: null, over_book: null, under_book: null, books_count: 0 };
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

export async function getMarketBoardPayload(league: LeagueCode) {
  const historicalReplay = getHistoricalReplayGames(league);
  const asOf = getLatestUpcomingAsOf(league);
  if (!asOf) {
    return {
      league,
      as_of_utc: null,
      odds_as_of_utc: null,
      date_central: centralTodayDateKey(),
      strategy_configs: historicalReplay.strategy_configs,
      rows: [],
    };
  }

  const todayKey = centralTodayDateKey();
  const rows = getScheduledTodayRows(league, asOf)
    .map((row) => ({ ...row, home_win_probability: normalizeProbability(row.home_win_probability) }))
    .filter((row) => dateKeyForScheduledGame(row) === todayKey);

  const latestOddsSnapshot = getLatestOddsSnapshot(league);
  const oddsSnapshotId = typeof latestOddsSnapshot?.odds_snapshot_id === "string" ? latestOddsSnapshot.odds_snapshot_id : "";
  const oddsAsOfUtc = typeof latestOddsSnapshot?.as_of_utc === "string" ? latestOddsSnapshot.as_of_utc : null;
  const gameIds = rows.map((row) => Number(row.game_id)).filter((value) => Number.isFinite(value));
  const marketLines = getMarketLinesForSnapshot(league, oddsSnapshotId, gameIds);

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

  const teamNames = new Map<string, string>();
  for (const row of getLatestTeamNames(league)) {
    const abbrev = String(row.team_abbrev || "").trim();
    const name = String(row.team_name || "").trim();
    if (abbrev && name) {
      teamNames.set(abbrev, name);
    }
  }

  const enrichedRows = rows.map((row) => {
    const matchedLines = linesByGameId.get(Number(row.game_id)) || linesByPairKey.get(pairKey(row.home_team, row.away_team)) || [];
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

  return {
    league,
    as_of_utc: asOf,
    odds_as_of_utc: oddsAsOfUtc,
    date_central: todayKey,
    strategy_configs: historicalReplay.strategy_configs,
    rows: enrichedRows,
  };
}
