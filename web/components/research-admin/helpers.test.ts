import assert from "node:assert/strict";
import test from "node:test";

import type { ResearchAdminResponse, ResearchChampionSummary, ResearchPromotionSummary } from "@/lib/types";
import { buildAdminCounts, buildChampionTimeline, describeChampion, summarizeRunSummary } from "./helpers";

test("buildChampionTimeline keeps the active champion, includes older promotions, and ignores rejected decisions", () => {
  const champion: ResearchChampionSummary = {
    league: "NBA",
    profile_key: "default",
    model_name: "glm_ridge",
    promoted_at_utc: "2026-03-27T12:00:00Z",
    source_run_id: "run-7",
    source_brief_id: "brief-2",
  };
  const decisions: ResearchPromotionSummary[] = [
    {
      promoted: false,
      candidate_model_name: "glm_poisson",
      incumbent_model_name: "glm_ridge",
      reason_summary: "Failed bankroll gate",
      created_at_utc: "2026-03-27T13:00:00Z",
    },
    {
      promoted: true,
      candidate_model_name: "glm_ridge",
      incumbent_model_name: "glm_elastic_net",
      reason_summary: "Beat incumbent on bankroll and drawdown.",
      created_at_utc: "2026-03-27T12:00:00Z",
    },
    {
      promoted: true,
      candidate_model_name: "glm_bayes",
      incumbent_model_name: "glm_elastic_net",
      reason_summary: "Historical approval",
      created_at_utc: "2026-03-20T12:00:00Z",
    },
  ];

  const timeline = buildChampionTimeline(champion, decisions);
  assert.equal(timeline.length, 2);
  assert.equal(timeline[0]?.model_name, "glm_ridge");
  assert.equal(timeline[0]?.is_active, true);
  assert.equal(timeline[0]?.reason_summary, "Beat incumbent on bankroll and drawdown.");
  assert.equal(timeline[1]?.model_name, "glm_bayes");
});

test("summarizeRunSummary surfaces a readable headline from stored metrics", () => {
  assert.equal(
    summarizeRunSummary({
      best_candidate_model: "glm_ridge",
      bankroll: 142.73,
      max_drawdown: 38.2,
      profitable_folds: 3,
    }),
    "glm_ridge · bankroll 143 · drawdown 38 · profitable folds 3"
  );
});

test("buildAdminCounts and describeChampion summarize the admin header state", () => {
  const payload: ResearchAdminResponse = {
    league: "NBA",
    as_of_utc: "2026-03-27T12:15:00Z",
    champion: {
      league: "NBA",
      profile_key: "default",
      model_name: "glm_ridge",
      promoted_at_utc: "2026-03-27T12:00:00Z",
      source_run_id: "run-7",
      source_brief_id: "brief-2",
      descriptor: { candidate_count: 4 },
    },
    briefs: [
      {
        brief_id: "brief-1",
        league: "NBA",
        profile_key: "default",
        brief_key: "default",
        title: "Default desk brief",
        status: "active",
        updated_at_utc: "2026-03-27T00:00:00Z",
      },
    ],
    runs: [],
    decisions: [
      { promoted: true, candidate_model_name: "glm_ridge", created_at_utc: "2026-03-27T12:00:00Z" },
      { promoted: false, candidate_model_name: "glm_poisson", created_at_utc: "2026-03-27T13:00:00Z" },
    ],
  };

  assert.deepEqual(buildAdminCounts(payload), {
    briefs: 1,
    runs: 0,
    decisions: 2,
    promotions: 1,
  });
  assert.equal(describeChampion(payload.champion), "Run run-7 · 4 candidate models reviewed");
});
