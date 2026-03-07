/**
 * Public dashboard/staging payload contracts.
 *
 * These shapes are consumed by both the live dashboard routes and the committed
 * staging snapshots under `web/public/staging-data/`. Route services should
 * return these contracts directly so staging-data generation stays aligned.
 */
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

export type DashboardRouteContract = keyof DashboardPayloadByKey;
