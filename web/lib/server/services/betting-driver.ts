import { runSqlJson } from "@/lib/db";
import type { LeagueCode } from "@/lib/league";
import { ensureHistoricalReplayDecisionTable } from "@/lib/replay-bets";

type RawBettableModelScoreRow = {
  game_id?: number | null;
  home_win?: number | null;
  model_name?: string | null;
  prob_home_win?: number | null;
};

const DEFAULT_BETTING_MODEL = "ensemble";
const BETTABLE_DRIVER_LOOKBACK_DAYS = 60;
const REPLAY_DECISION_TABLE = "historical_bet_decisions_by_profile_v2";
const FROZEN_FORECAST_SOURCE = "train_upcoming";

// Temporary live-betting driver reranker:
// We are intentionally selecting the betting driver from recent *bettable*
// games instead of blindly trusting the global ensemble for every league. This
// was added after the March 8, 2026 audit showed that the model leading the
// broad leaderboard was not the same model performing best on the smaller slice
// of games that actually cleared the betting gates. This should not become
// permanent folklore. Future Codex threads should explicitly revisit whether
// this reranker is still needed once each league has a healthier post-change
// replay sample. A good removal test is: at least 50 settled risk-adjusted
// bets for the league after this change, plus the ensemble once again matching
// or beating the reranked driver on recent bettable-game log loss and Brier.
// If those checks pass, prefer deleting this reranker and returning to the
// simpler ensemble-first policy.
const TEMPORARY_MIN_BETTABLE_DRIVER_GAMES = 10;

function clampProbability(value: number): number {
  return Math.min(1 - 1e-9, Math.max(1e-9, value));
}

function modelDriverSql(lookbackDays: number): string {
  return `
    WITH latest_day AS (
      SELECT MAX(date_central) AS max_date
      FROM ${REPLAY_DECISION_TABLE}
      WHERE strategy = 'riskAdjusted'
        AND stake > 0
    ),
    target_games AS (
      SELECT DISTINCT
        d.game_id,
        r.home_win,
        COALESCE(g.start_time_utc, r.final_utc, r.game_date_utc || 'T23:59:59Z') AS replay_cutoff_utc
      FROM ${REPLAY_DECISION_TABLE} d
      JOIN latest_day ld
        ON ld.max_date IS NOT NULL
      JOIN results r
        ON r.game_id = d.game_id
      LEFT JOIN games g
        ON g.game_id = d.game_id
      WHERE d.strategy = 'riskAdjusted'
        AND d.stake > 0
        AND r.home_win IS NOT NULL
        AND d.date_central >= DATE(ld.max_date, '-${lookbackDays - 1} days')
    ),
    ranked AS (
      SELECT
        tg.game_id,
        tg.home_win,
        p.model_name,
        p.prob_home_win,
        ROW_NUMBER() OVER (
          PARTITION BY tg.game_id, p.model_name
          ORDER BY
            DATETIME(p.as_of_utc) DESC,
            DATETIME(COALESCE(mr.created_at_utc, p.as_of_utc)) DESC,
            p.prediction_id DESC
        ) AS rn
      FROM target_games tg
      JOIN predictions p
        ON p.game_id = tg.game_id
       AND COALESCE(json_extract(p.metadata_json, '$.source'), '') = '${FROZEN_FORECAST_SOURCE}'
       AND DATETIME(p.as_of_utc) <= DATETIME(tg.replay_cutoff_utc)
      LEFT JOIN model_runs mr
        ON mr.model_run_id = p.model_run_id
      WHERE DATETIME(COALESCE(mr.created_at_utc, p.as_of_utc)) <= DATETIME(tg.replay_cutoff_utc)
    )
    SELECT game_id, home_win, model_name, prob_home_win
    FROM ranked
    WHERE rn = 1
  `;
}

export function getPreferredBettingModelName(league: LeagueCode): string {
  ensureHistoricalReplayDecisionTable(league);
  const rows = runSqlJson(modelDriverSql(BETTABLE_DRIVER_LOOKBACK_DAYS), { league }) as RawBettableModelScoreRow[];
  const metricsByModel = new Map<string, { n: number; logLoss: number; brier: number; accuracy: number }>();

  for (const row of rows) {
    const modelName = String(row.model_name || "").trim();
    const probability = Number(row.prob_home_win);
    const outcome = Number(row.home_win);
    if (!modelName || !Number.isFinite(probability) || (outcome !== 0 && outcome !== 1)) {
      continue;
    }

    const p = clampProbability(probability);
    const current = metricsByModel.get(modelName) || { n: 0, logLoss: 0, brier: 0, accuracy: 0 };
    current.n += 1;
    current.logLoss += -(outcome * Math.log(p) + (1 - outcome) * Math.log(1 - p));
    current.brier += (p - outcome) ** 2;
    current.accuracy += Number((p >= 0.5) === (outcome === 1));
    metricsByModel.set(modelName, current);
  }

  const candidates = Array.from(metricsByModel.entries())
    .filter(([, metrics]) => metrics.n >= TEMPORARY_MIN_BETTABLE_DRIVER_GAMES)
    .map(([modelName, metrics]) => ({
      modelName,
      n: metrics.n,
      logLoss: metrics.logLoss / metrics.n,
      brier: metrics.brier / metrics.n,
      accuracy: metrics.accuracy / metrics.n,
    }))
    .sort(
      (left, right) =>
        left.logLoss - right.logLoss || left.brier - right.brier || right.accuracy - left.accuracy || left.modelName.localeCompare(right.modelName)
    );

  return candidates[0]?.modelName || DEFAULT_BETTING_MODEL;
}
