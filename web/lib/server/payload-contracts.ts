/**
 * Public dashboard/staging payload contracts.
 *
 * These shapes are consumed by both the live dashboard routes and the committed
 * staging snapshots under `web/public/staging-data/`. Route services should
 * return these contracts directly so staging-data generation stays aligned.
 */
import type { DashboardRouteKey } from "@/lib/generated/dashboard-routes";
import type { BetHistoryResponse } from "@/lib/bet-history-types";
import type {
  ActualVsExpectedResponse,
  GamesTodayResponse,
  MarketBoardResponse,
  MetricsResponse,
  PerformanceResponse,
  PredictionsResponse,
  ValidationResponse,
} from "@/lib/types";

export type DashboardPayloadByKey = {
  actualVsExpected: ActualVsExpectedResponse;
  betHistory: BetHistoryResponse;
  gamesToday: GamesTodayResponse;
  marketBoard: MarketBoardResponse;
  metrics: MetricsResponse;
  performance: PerformanceResponse;
  predictions: PredictionsResponse;
  validation: ValidationResponse;
};

type _AssertRoutePayloadCoverage = DashboardRouteKey extends keyof DashboardPayloadByKey ? true : never;
type _AssertPayloadRouteCoverage = keyof DashboardPayloadByKey extends DashboardRouteKey ? true : never;

const _routePayloadCoverage: _AssertRoutePayloadCoverage = true;
const _payloadRouteCoverage: _AssertPayloadRouteCoverage = true;

void _routePayloadCoverage;
void _payloadRouteCoverage;

export type DashboardRouteContract = DashboardRouteKey;
