export type JsonRouteHandler = (request: Request) => Promise<Response>;

async function loadGetHandler(path: string): Promise<JsonRouteHandler> {
  const mod = await import(path);
  const handler = (mod as { GET?: JsonRouteHandler }).GET || (mod as { default?: { GET?: JsonRouteHandler } }).default?.GET;
  if (!handler) {
    throw new Error(`Route module ${path} did not export GET`);
  }
  return handler;
}

export const STAGING_ROUTE_LOADERS: Record<string, () => Promise<JsonRouteHandler>> = {
  "app/api/actual-vs-expected/route.ts": async () => loadGetHandler("../app/api/actual-vs-expected/route.ts"),
  "app/api/bet-history/route.ts": async () => loadGetHandler("../app/api/bet-history/route.ts"),
  "app/api/games-today/route.ts": async () => loadGetHandler("../app/api/games-today/route.ts"),
  "app/api/market-board/route.ts": async () => loadGetHandler("../app/api/market-board/route.ts"),
  "app/api/metrics/route.ts": async () => loadGetHandler("../app/api/metrics/route.ts"),
  "app/api/performance/route.ts": async () => loadGetHandler("../app/api/performance/route.ts"),
  "app/api/predictions/route.ts": async () => loadGetHandler("../app/api/predictions/route.ts"),
  "app/api/validation/route.ts": async () => loadGetHandler("../app/api/validation/route.ts"),
};
