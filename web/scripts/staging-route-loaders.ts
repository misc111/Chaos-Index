import { GET as actualVsExpectedGet } from "../app/api/actual-vs-expected/route.ts";
import { GET as betHistoryGet } from "../app/api/bet-history/route.ts";
import { GET as gamesTodayGet } from "../app/api/games-today/route.ts";
import { GET as marketBoardGet } from "../app/api/market-board/route.ts";
import { GET as metricsGet } from "../app/api/metrics/route.ts";
import { GET as performanceGet } from "../app/api/performance/route.ts";
import { GET as predictionsGet } from "../app/api/predictions/route.ts";
import { GET as validationGet } from "../app/api/validation/route.ts";

export type JsonRouteHandler = (request: Request) => Promise<Response>;

export const STAGING_ROUTE_LOADERS: Record<string, JsonRouteHandler> = {
  "app/api/actual-vs-expected/route.ts": actualVsExpectedGet,
  "app/api/bet-history/route.ts": betHistoryGet,
  "app/api/games-today/route.ts": gamesTodayGet,
  "app/api/market-board/route.ts": marketBoardGet,
  "app/api/metrics/route.ts": metricsGet,
  "app/api/performance/route.ts": performanceGet,
  "app/api/predictions/route.ts": predictionsGet,
  "app/api/validation/route.ts": validationGet,
};
