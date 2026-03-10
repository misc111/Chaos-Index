import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export type JsonRouteHandler = (request: Request) => Promise<Response>;

type RouteModule = {
  GET?: JsonRouteHandler;
  default?: {
    GET?: JsonRouteHandler;
  };
};

function resolveGetHandler(path: string, mod: RouteModule): JsonRouteHandler {
  const handler = mod.GET || mod.default?.GET;
  if (!handler) {
    throw new Error(`Route module ${path} did not export GET`);
  }
  return handler;
}

async function loadRouteModule(routePath: string): Promise<RouteModule> {
  const scriptDir = path.dirname(fileURLToPath(import.meta.url));
  const modulePath = path.resolve(scriptDir, "..", routePath);
  return import(pathToFileURL(modulePath).href);
}

export const STAGING_ROUTE_LOADERS: Record<string, () => Promise<JsonRouteHandler>> = {
  "app/api/actual-vs-expected/route.ts": async () =>
    resolveGetHandler("app/api/actual-vs-expected/route.ts", await loadRouteModule("app/api/actual-vs-expected/route.ts")),
  "app/api/bet-history/route.ts": async () =>
    resolveGetHandler("app/api/bet-history/route.ts", await loadRouteModule("app/api/bet-history/route.ts")),
  "app/api/games-today/route.ts": async () =>
    resolveGetHandler("app/api/games-today/route.ts", await loadRouteModule("app/api/games-today/route.ts")),
  "app/api/market-board/route.ts": async () =>
    resolveGetHandler("app/api/market-board/route.ts", await loadRouteModule("app/api/market-board/route.ts")),
  "app/api/metrics/route.ts": async () =>
    resolveGetHandler("app/api/metrics/route.ts", await loadRouteModule("app/api/metrics/route.ts")),
  "app/api/performance/route.ts": async () =>
    resolveGetHandler("app/api/performance/route.ts", await loadRouteModule("app/api/performance/route.ts")),
  "app/api/predictions/route.ts": async () =>
    resolveGetHandler("app/api/predictions/route.ts", await loadRouteModule("app/api/predictions/route.ts")),
  "app/api/validation/route.ts": async () =>
    resolveGetHandler("app/api/validation/route.ts", await loadRouteModule("app/api/validation/route.ts")),
};
