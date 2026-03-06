import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const appRoot = path.resolve(scriptDir, "..");

function detectRepoName() {
  if (process.env.GITHUB_REPOSITORY) {
    const parts = process.env.GITHUB_REPOSITORY.split("/");
    return parts[parts.length - 1] || "";
  }

  try {
    const remote = execFileSync("git", ["remote", "get-url", "origin"], {
      cwd: path.resolve(appRoot, ".."),
      encoding: "utf8",
    }).trim();
    const cleaned = remote.replace(/\.git$/, "");
    const lastSlash = cleaned.lastIndexOf("/");
    const lastColon = cleaned.lastIndexOf(":");
    const splitAt = Math.max(lastSlash, lastColon);
    return splitAt >= 0 ? cleaned.slice(splitAt + 1) : cleaned;
  } catch {
    return "";
  }
}

const repoName = process.env.PAGES_REPO_NAME || detectRepoName();
const basePath = process.env.PAGES_BASE_PATH || (repoName ? `/${repoName}` : "");
const nextBin = path.resolve(appRoot, "node_modules", ".bin", process.platform === "win32" ? "next.cmd" : "next");
const snapshotManifest = path.join(appRoot, "public", "staging-data", "manifest.json");
const apiDir = path.join(appRoot, "app", "api");
const parkedApiDir = path.join(appRoot, ".pages-build", "api");

if (!fs.existsSync(snapshotManifest)) {
  console.error("Missing web/public/staging-data/manifest.json. Run `npm run generate:staging-data` before building Pages.");
  process.exit(1);
}

fs.mkdirSync(path.dirname(parkedApiDir), { recursive: true });

if (fs.existsSync(parkedApiDir)) {
  fs.rmSync(parkedApiDir, { recursive: true, force: true });
}

let apiWasParked = false;
let exitCode = 1;

try {
  if (fs.existsSync(apiDir)) {
    fs.renameSync(apiDir, parkedApiDir);
    apiWasParked = true;
  }

  const result = spawnSync(nextBin, ["build"], {
    cwd: appRoot,
    env: {
      ...process.env,
      STATIC_EXPORT: "1",
      NEXT_PUBLIC_STATIC_STAGING: "1",
      PAGES_BASE_PATH: basePath,
      NEXT_PUBLIC_BASE_PATH: basePath,
    },
    stdio: "inherit",
  });

  if (typeof result.status === "number") {
    exitCode = result.status;
  }
} finally {
  if (apiWasParked && fs.existsSync(parkedApiDir)) {
    fs.renameSync(parkedApiDir, apiDir);
  }
}

process.exit(exitCode);
