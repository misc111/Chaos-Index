import * as actualVsExpectedRoute from "../app/api/actual-vs-expected/route";
import * as betHistoryRoute from "../app/api/bet-history/route";
import * as gamesTodayRoute from "../app/api/games-today/route";
import * as marketBoardRoute from "../app/api/market-board/route";
import * as metricsRoute from "../app/api/metrics/route";
import * as performanceRoute from "../app/api/performance/route";
import * as predictionsRoute from "../app/api/predictions/route";
import * as validationRoute from "../app/api/validation/route";

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

export const STAGING_ROUTE_LOADERS: Record<string, () => Promise<JsonRouteHandler>> = {
  "app/api/actual-vs-expected/route.ts": async () =>
    resolveGetHandler("app/api/actual-vs-expected/route.ts", actualVsExpectedRoute),
  "app/api/bet-history/route.ts": async () => resolveGetHandler("app/api/bet-history/route.ts", betHistoryRoute),
  "app/api/games-today/route.ts": async () => resolveGetHandler("app/api/games-today/route.ts", gamesTodayRoute),
  "app/api/market-board/route.ts": async () => resolveGetHandler("app/api/market-board/route.ts", marketBoardRoute),
  "app/api/metrics/route.ts": async () => resolveGetHandler("app/api/metrics/route.ts", metricsRoute),
  "app/api/performance/route.ts": async () => resolveGetHandler("app/api/performance/route.ts", performanceRoute),
  "app/api/predictions/route.ts": async () => resolveGetHandler("app/api/predictions/route.ts", predictionsRoute),
  "app/api/validation/route.ts": async () => resolveGetHandler("app/api/validation/route.ts", validationRoute),
};
