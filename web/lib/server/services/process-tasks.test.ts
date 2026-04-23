import assert from "node:assert/strict";
import test from "node:test";

import { DEFAULT_TRAINING_MODELS, EXPERIMENTAL_MODEL_KEYS, THEORY_EXTENSION_MODEL_KEYS } from "@/lib/generated/model-manifest";
import { defaultTrainingModelsForLeague, parseRequestedModels } from "./process-tasks";

test("default training models stay on the core actuarial lane", () => {
  const defaults = defaultTrainingModelsForLeague("MLB");
  const defaultSet = new Set<string>(defaults);

  assert.deepEqual(defaults, [...DEFAULT_TRAINING_MODELS]);
  assert.deepEqual(defaultTrainingModelsForLeague("NBA"), [...DEFAULT_TRAINING_MODELS]);
  for (const experimentalModel of EXPERIMENTAL_MODEL_KEYS) {
    assert.equal(defaultSet.has(experimentalModel), false);
  }
  for (const extensionModel of THEORY_EXTENSION_MODEL_KEYS) {
    assert.equal(defaultSet.has(extensionModel), false);
  }
});

test("parseRequestedModels still allows explicit challenger opt-in", () => {
  assert.deepEqual(parseRequestedModels(["glm", "rf", "glm", "nn"]), ["glm_ridge", "rf", "nn_mlp"]);
});
