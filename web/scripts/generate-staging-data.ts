import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { LeagueCode } from "../lib/league";

type JsonRouteHandler = (request: Request) => Promise<Response>;
type JsonRecord = Record<string, unknown>;

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const appRoot = path.resolve(scriptDir, "..");
// Maintainer note: this script is the bridge from the live local dashboard
// data model to the committed GitHub Pages staging snapshot. If a local
// dashboard/API change should be visible on staging, rerun this script and
// commit the resulting files under web/public/staging-data/.
const outputRoot = path.join(appRoot, "public", "staging-data");
const generatedAtUtc = new Date().toISOString();

const routes: Array<{ fileName: string; modulePath: string; routePath: string }> = [
  { fileName: "actual-vs-expected.json", modulePath: "app/api/actual-vs-expected/route.ts", routePath: "/api/actual-vs-expected" },
  { fileName: "bet-history.json", modulePath: "app/api/bet-history/route.ts", routePath: "/api/bet-history" },
  { fileName: "games-today.json", modulePath: "app/api/games-today/route.ts", routePath: "/api/games-today" },
  { fileName: "market-board.json", modulePath: "app/api/market-board/route.ts", routePath: "/api/market-board" },
  { fileName: "metrics.json", modulePath: "app/api/metrics/route.ts", routePath: "/api/metrics" },
  { fileName: "performance.json", modulePath: "app/api/performance/route.ts", routePath: "/api/performance" },
  { fileName: "predictions.json", modulePath: "app/api/predictions/route.ts", routePath: "/api/predictions" },
  { fileName: "validation.json", modulePath: "app/api/validation/route.ts", routePath: "/api/validation" },
];

function requestForLeague(routePath: string, league: LeagueCode): Request {
  return new Request(`http://staging.local${routePath}?league=${league}`);
}

function sanitizePublicPayload(fileName: string, payload: unknown, league: LeagueCode): unknown {
  if (fileName !== "validation.json") {
    return payload;
  }

  const raw = (payload || {}) as JsonRecord;
  return {
    league,
    significance: [],
    sections: {
      public_note: [
        {
          message: "Detailed validation artifacts remain private in the public GitHub Pages staging build.",
        },
      ],
    },
    as_of_utc: raw.as_of_utc ?? null,
  };
}

async function writeJson(filePath: string, payload: unknown): Promise<void> {
  await fs.writeFile(filePath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}

async function loadRouteHandler(modulePath: string): Promise<JsonRouteHandler> {
  const routeEntryPoint = new URL(`../${modulePath}`, import.meta.url);
  const routeModule = (await import(routeEntryPoint.href)) as { GET?: JsonRouteHandler };
  if (typeof routeModule.GET !== "function") {
    throw new Error(`Route module ${modulePath} does not export GET`);
  }
  return routeModule.GET;
}

async function generateLeagueSnapshot(league: LeagueCode): Promise<void> {
  const leagueDir = path.join(outputRoot, league.toLowerCase());
  await fs.mkdir(leagueDir, { recursive: true });

  for (const route of routes) {
    const handler = await loadRouteHandler(route.modulePath);
    const response = await handler(requestForLeague(route.routePath, league));
    const payload = await response.json();
    await writeJson(path.join(leagueDir, route.fileName), sanitizePublicPayload(route.fileName, payload, league));
  }

  await writeJson(path.join(leagueDir, "meta.json"), {
    generated_at_utc: generatedAtUtc,
    league,
    mode: "static-staging-snapshot",
    note: "Generated locally from the current dashboard data sources for GitHub Pages staging.",
    files: routes.map((route) => route.fileName),
  });
}

async function main(): Promise<void> {
  await generateLeagueSnapshot("NHL");
  await generateLeagueSnapshot("NBA");
  // Maintainer note: Pages publishes these committed artifacts directly.
  // Regenerating without committing leaves the local dashboard and staging out of sync.
  await writeJson(path.join(outputRoot, "manifest.json"), {
    generated_at_utc: generatedAtUtc,
    leagues: ["NHL", "NBA"],
  });
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
