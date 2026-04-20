import assert from "node:assert/strict";
import test from "node:test";

import { CORE_MODEL_KEYS, EXPERIMENTAL_MODEL_KEYS } from "@/lib/generated/model-manifest";
import { defaultTrainingModelsForLeague, parseRequestedModels } from "./process-tasks";

test("default training models stay on the core actuarial lane", () => {
  const defaults = defaultTrainingModelsForLeague("MLB");
  const defaultSet = new Set<string>(defaults);

  assert.deepEqual(defaults, [...CORE_MODEL_KEYS]);
  assert.deepEqual(defaultTrainingModelsForLeague("NBA"), [...CORE_MODEL_KEYS]);
  for (const experimentalModel of EXPERIMENTAL_MODEL_KEYS) {
    assert.equal(defaultSet.has(experimentalModel), false);
  }
});

test("parseRequestedModels still allows explicit challenger opt-in", () => {
  assert.deepEqual(parseRequestedModels(["glm", "rf", "glm", "nn"]), ["glm_ridge", "rf", "nn_mlp"]);
});
