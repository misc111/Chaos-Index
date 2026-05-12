import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import test from "node:test";

import { getTeamIconDefinition, MLB_TEAM_ICON_CODES, MLB_TEAM_ICON_FILE_CODES } from "./team-icons";

const MLB_TEAM_ICON_DIR = new URL("../public/team-icons/mlb/", import.meta.url);

test("MLB team icon registry resolves to the complete committed PNG set", () => {
  assert.equal(MLB_TEAM_ICON_CODES.length, 30);

  const expectedFiles = MLB_TEAM_ICON_CODES.map((code) => `${MLB_TEAM_ICON_FILE_CODES[code]}.png`).sort();
  const actualFiles = readdirSync(MLB_TEAM_ICON_DIR)
    .filter((fileName) => fileName.endsWith(".png"))
    .sort();

  assert.deepEqual(actualFiles, expectedFiles);

  for (const code of MLB_TEAM_ICON_CODES) {
    const icon = getTeamIconDefinition("MLB", code);
    assert.equal(icon.src, `/team-icons/mlb/${MLB_TEAM_ICON_FILE_CODES[code]}.png`);
  }
});
