# NHL + NBA Agent Instructions

## Scope Contract
- This project supports NHL and NBA forecasting.
- Interpret user questions in the configured league context by default (`config.data.league`).
- If no config context is available, default to NHL for ambiguous wording.
- If a bug, regression, or data drift issue is found in one supported league, investigate the same failure mode in the other supported league before closing the task.
- Cross-league investigation must cover the analogous pipeline stages that could share the bug: storage tables, model outputs, API payloads, dashboard views, and committed staging snapshots.
- If the same bug exists in the other league, fix it in the shared or league-specific path as appropriate and regenerate any affected staging snapshot files for every impacted league.

## Team Interpretation
- Treat city names, nicknames, mascots, and common shorthand as NHL/NBA clubs.
- Resolve clear references directly.
- If wording is ambiguous across leagues and context is missing, ask for a short league clarification.

## Supported Query Styles
- Casual next-game probability phrasing should map to the team forecast.
- Casual multi-game phrasing should be supported, e.g. `next 3 games`, `next three`, `next couple`, `next few`.
- Championship questions should be supported:
  - NHL: Stanley Cup
  - NBA: NBA Finals
- Championship answers should be probabilistic heuristic estimates with explicit caveats.

## Out-of-Scope Requests
- If a user asks about leagues outside NHL/NBA (MLB, MLS, NFL, etc.), respond that this project currently supports NHL and NBA forecasting and ask for an NHL/NBA-framed question.

## Dashboard And Staging Sync
- Treat the local Next.js dashboard and the GitHub Pages staging site as two separate delivery targets.
- If you change dashboard code or API payloads in a way that affects the shipped staging experience, also update the staging snapshot inputs under `web/public/staging-data/`.
- The required staging sync path is:
  - `cd web && npm run generate:staging-data`
  - commit the updated files in `web/public/staging-data/`
  - if needed, verify with `cd web && npm run build:pages`
- Do not assume pushing dashboard code alone updates staging; GitHub Pages serves the committed snapshot files, not live SQLite data.

## Data Refresh Contract
- When the user asks to refresh data only, pull in data without rebuilding features, or refresh without training, treat that as the repository-level data-only pipeline.
- The canonical repository trigger for the executable data-only pipeline is `make data_refresh`.
- A data-only refresh always covers both leagues, even if the user names only one team or one league in the same message.
- Run the data-only refresh steps in this exact order, sequentially, with no league parallelism and no step reordering:
  - `make fetch CONFIG=configs/nhl.yaml`
  - `make fetch CONFIG=configs/nba.yaml`
  - `python3 -m src.cli fetch-odds --config configs/nhl.yaml`
  - `python3 -m src.cli fetch-odds --config configs/nba.yaml`
- Data-only refreshes stop after data ingestion:
  - do not rebuild features
  - do not train
  - do not regenerate `web/public/staging-data/`
  - do not build Pages just because a data-only refresh ran
- The dedicated `fetch-odds` step is mandatory for data-only refreshes so the ingest cycle ends with the freshest odds snapshot for each league.

## Hard Refresh Contract
- Treat the exact phrase `do a hard refresh` as a repository-level command alias for a full deterministic refresh/train/publish cycle across both supported leagues without rebuilding features.
- The canonical repository trigger for the executable refresh pipeline is `make hard_refresh`. After it succeeds, continue with the required commit/push/workflow-watch closeout steps below.
- A hard refresh always covers both leagues, even if the user names only one team or one league in the same message.
- Run the hard refresh steps in this exact order, sequentially, with no league parallelism and no step reordering:
  - `make init-db CONFIG=configs/nhl.yaml`
  - `make init-db CONFIG=configs/nba.yaml`
  - `make fetch CONFIG=configs/nhl.yaml`
  - `make fetch CONFIG=configs/nba.yaml`
  - `python3 -m src.cli fetch-odds --config configs/nhl.yaml`
  - `python3 -m src.cli fetch-odds --config configs/nba.yaml`
  - `make train CONFIG=configs/nhl.yaml`
  - `make train CONFIG=configs/nba.yaml`
  - `cd web && npm run generate:staging-data`
  - if dashboard or staging-build behavior changed, `cd web && npm run build:pages`
  - commit all resulting tracked changes
  - `git push origin main`
  - watch the `Publish Sanitized Staging Site` GitHub Actions workflow for the pushed `HEAD`
- The dedicated `fetch-odds` step is mandatory for hard refreshes. `fetch` already persists an odds snapshot, but hard refreshes must end data collection with an explicit final odds pull for each league before training.
- Hard refreshes reuse the current processed feature snapshot for each league. They do not run `make features`.
- Hard refreshes must use the repository defaults for model coverage. Do not narrow `MODELS=` unless the user explicitly asks for a partial rebuild.
- Hard refreshes must be fail-fast and deterministic in behavior:
  - do not skip a league because its files look unchanged
  - do not parallelize NHL and NBA runs
  - do not silently rerun steps in a different order
  - do not pass `APPROVE_FEATURE_CHANGES=1` unless the user explicitly asks to approve a feature-contract update
  - if any required step fails, stop, report the failing command, and do not push partial results
- When closing a successful hard refresh, report which commit was pushed, whether staging-data changed, and the final GitHub Actions workflow URL plus success/failure status.

## Git Workflow
- Keep this repository on `main`. Do not create or push feature branches unless the user explicitly asks for one.
- After completing repository edits, commit on `main` and push `main` to `origin` by default so the web app and staging publish can update. Only skip the push if the user explicitly says not to push yet.
- Do not wait for a separate "push" request once the requested edits are complete; pushing is the default close-out step for this repo.
- After every push to GitHub, watch the `Publish Sanitized Staging Site` GitHub Actions workflow before closing out the task.
- The required workflow-watch step is:
  - `gh run list --workflow "Publish Sanitized Staging Site" --limit 5 --json databaseId,headSha,status,conclusion,url,displayTitle`
  - identify the run for the pushed `HEAD`
  - `gh run watch <databaseId> --interval 5`
  - confirm the final workflow URL and whether it succeeded
