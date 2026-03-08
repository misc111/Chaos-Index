import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const appRoot = path.resolve(scriptDir, "..");
const teamIconsRoot = path.join(appRoot, "public", "team-icons");

const NBA_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams";
const NHL_TEAMS_URL = "https://api-web.nhle.com/v1/standings/now";

const NBA_TEAM_CODES = [
  "ATL",
  "BKN",
  "BOS",
  "CHA",
  "CHI",
  "CLE",
  "DAL",
  "DEN",
  "DET",
  "GS",
  "HOU",
  "IND",
  "LAC",
  "LAL",
  "MEM",
  "MIA",
  "MIL",
  "MIN",
  "NO",
  "NY",
  "OKC",
  "ORL",
  "PHI",
  "PHX",
  "POR",
  "SA",
  "SAC",
  "TOR",
  "UTAH",
  "WSH",
];

const NHL_TEAM_CODES = [
  "ANA",
  "BOS",
  "BUF",
  "CAR",
  "CBJ",
  "CGY",
  "CHI",
  "COL",
  "DAL",
  "DET",
  "EDM",
  "FLA",
  "LAK",
  "MIN",
  "MTL",
  "NJD",
  "NSH",
  "NYI",
  "NYR",
  "OTT",
  "PHI",
  "PIT",
  "SEA",
  "SJS",
  "STL",
  "TBL",
  "TOR",
  "UTA",
  "VAN",
  "VGK",
  "WPG",
  "WSH",
];

async function fetchJson(url) {
  const response = await fetch(url, {
    headers: {
      "user-agent": "SportsModeling Team Icon Sync",
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status}`);
  }

  return response.json();
}

async function downloadFile(url, destination) {
  const response = await fetch(url, {
    headers: {
      "user-agent": "SportsModeling Team Icon Sync",
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to download ${url}: ${response.status}`);
  }

  const buffer = Buffer.from(await response.arrayBuffer());
  await fs.writeFile(destination, buffer);
}

function pickNbaLogo(team) {
  const logos = Array.isArray(team.logos) ? team.logos : [];
  return (
    logos.find((entry) => Array.isArray(entry.rel) && entry.rel.includes("full") && entry.rel.includes("default"))?.href ||
    logos[0]?.href ||
    null
  );
}

function buildExpectedUrlMap(entries, expectedCodes, league) {
  const byCode = new Map(entries.map((entry) => [entry.code, entry.url]));

  for (const code of expectedCodes) {
    if (!byCode.has(code)) {
      throw new Error(`Missing ${league} logo source for ${code}`);
    }
  }

  return byCode;
}

async function recreateDir(directory) {
  await fs.rm(directory, { recursive: true, force: true });
  await fs.mkdir(directory, { recursive: true });
}

async function syncNbaIcons() {
  const payload = await fetchJson(NBA_TEAMS_URL);
  const rawTeams = ((((payload?.sports || [])[0] || {}).leagues || [])[0] || {}).teams || [];
  const entries = rawTeams
    .map((row) => row?.team || {})
    .map((team) => ({
      code: String(team.abbreviation || "").trim().toUpperCase(),
      url: pickNbaLogo(team),
    }))
    .filter((entry) => entry.code && entry.url);

  const byCode = buildExpectedUrlMap(entries, NBA_TEAM_CODES, "NBA");
  const outputDir = path.join(teamIconsRoot, "nba");
  await recreateDir(outputDir);

  for (const code of NBA_TEAM_CODES) {
    await downloadFile(byCode.get(code), path.join(outputDir, `${code.toLowerCase()}.png`));
  }
}

async function syncNhlIcons() {
  const payload = await fetchJson(NHL_TEAMS_URL);
  const entries = (payload?.standings || [])
    .map((row) => ({
      code: String(row?.teamAbbrev?.default || "").trim().toUpperCase(),
      url: String(row?.teamLogo || "").trim(),
    }))
    .filter((entry) => entry.code && entry.url);

  const byCode = buildExpectedUrlMap(entries, NHL_TEAM_CODES, "NHL");
  const outputDir = path.join(teamIconsRoot, "nhl");
  await recreateDir(outputDir);

  for (const code of NHL_TEAM_CODES) {
    await downloadFile(byCode.get(code), path.join(outputDir, `${code.toLowerCase()}.svg`));
  }
}

await syncNbaIcons();
await syncNhlIcons();

console.log(`Synced ${NBA_TEAM_CODES.length} NBA logos and ${NHL_TEAM_CODES.length} NHL logos.`);
