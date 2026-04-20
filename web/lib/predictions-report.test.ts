import assert from "node:assert/strict";
import test from "node:test";

import { displayPredictionModel, predictionModelHeadline, predictionTrustNote } from "./predictions-report";

test("MLB trust notes describe baseball drivers instead of goalie or xG context", () => {
  const note = predictionTrustNote("glm_ridge", "MLB");

  assert.match(note, /starting pitching/i);
  assert.match(note, /bullpen/i);
  assert.doesNotMatch(note, /goalie/i);
  assert.doesNotMatch(note, /xg/i);
});

test("MLB headlines describe baseball-specific logistic context", () => {
  const headline = predictionModelHeadline("glm_lasso", "MLB", ["diff_starting_pitcher_quality"]);

  assert.match(String(headline), /starting pitcher/i);
  assert.match(String(headline), /bullpen/i);
  assert.doesNotMatch(String(headline), /goalie/i);
  assert.doesNotMatch(String(headline), /xg/i);
});

test("MLB poisson headline uses run-rate language", () => {
  const headline = predictionModelHeadline("goals_poisson", "MLB");

  assert.match(String(headline), /run-rate/i);
  assert.doesNotMatch(String(headline), /goal-rate/i);
});

test("experimental challengers are labeled as challengers in operator copy", () => {
  assert.match(displayPredictionModel("rf"), /challenger/i);
  assert.match(predictionTrustNote("rf", "MLB"), /experimental challenger/i);
  assert.match(String(predictionModelHeadline("bayes_bt_state_space", "MLB")), /experimental/i);
});
