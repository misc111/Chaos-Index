import assert from "node:assert/strict";
import test from "node:test";

import { getPreferredBettingModelName } from "./betting-driver";

test("NBA prefers glm_elastic_net as the betting driver", () => {
  assert.equal(getPreferredBettingModelName("NBA"), "glm_elastic_net");
  assert.equal(getPreferredBettingModelName("NBA", "historicalReplay"), "glm_elastic_net");
});

test("other leagues keep the ensemble betting driver", () => {
  assert.equal(getPreferredBettingModelName("NHL"), "ensemble");
  assert.equal(getPreferredBettingModelName("NCAAM"), "ensemble");
});
