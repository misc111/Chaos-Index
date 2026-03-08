export type JsonRouteHandler = (request: Request) => Promise<Response>;

export const STAGING_ROUTE_LOADERS: Record<string, () => Promise<JsonRouteHandler>> = {
  "app/api/actual-vs-expected/route.ts": async () => (await import("../app/api/actual-vs-expected/route.ts")).GET,
  "app/api/bet-history/route.ts": async () => (await import("../app/api/bet-history/route.ts")).GET,
  "app/api/games-today/route.ts": async () => (await import("../app/api/games-today/route.ts")).GET,
  "app/api/market-board/route.ts": async () => (await import("../app/api/market-board/route.ts")).GET,
  "app/api/metrics/route.ts": async () => (await import("../app/api/metrics/route.ts")).GET,
  "app/api/performance/route.ts": async () => (await import("../app/api/performance/route.ts")).GET,
  "app/api/predictions/route.ts": async () => (await import("../app/api/predictions/route.ts")).GET,
  "app/api/validation/route.ts": async () => (await import("../app/api/validation/route.ts")).GET,
};
