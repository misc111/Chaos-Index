# Historical Odds Cache Design

## Goal

Add a reproducible historical-odds cache workflow for the supported leagues, starting with the NBA use case that is currently blocking the full profit-first research tournament. The cache should be safe to delete and regenerate, and DB import should remain a separate explicit step.

## User Problem

The research desk can only run a short-horizon NBA tournament because the local priced-odds history starts in March 2026 while the games table reaches back to October 2025. We need a workflow that can download historical odds in manageable chunks, store them as raw cache artifacts, and rebuild a manifest that `import-history` can ingest later.

## Approach

Create a new cache-first command that:

1. Reads the selected league's games from the existing SQLite DB.
2. Infers or accepts a historical date window.
3. Splits that window into fixed-size chunks.
4. Exports a per-chunk games file.
5. Calls the existing `write_historical_odds_bundle(...)` helper for each chunk.
6. Rebuilds a top-level `data/raw/historical/<league>/manifest.json` that points at all chunk manifests.

The command does not import into the DB. Users can delete `data/raw/historical/<league>/` whenever they want to free space, then regenerate it later and run `import-history`.

## Storage Layout

Use the league's existing research source directory:

- `data/raw/historical/nba/backfill_2025-10-02_2025-10-31/`
- `data/raw/historical/nba/backfill_2025-11-01_2025-11-30/`
- ...
- `data/raw/historical/nba/manifest.json`

Each chunk directory contains:

- one exported games file for that chunk
- one odds CSV produced by `write_historical_odds_bundle(...)`
- one chunk-local `manifest.json`

The top-level manifest aggregates all chunk manifests using relative paths so `import-history` can read the whole cache with one path.

## Command Surface

Add a new CLI command and Make target for cache generation only.

Suggested CLI:

- `python3 -m src.cli backfill-historical-odds --config configs/nba.yaml`
- `python3 -m src.cli backfill-historical-odds --config configs/nba.yaml --start-date 2025-10-02 --end-date 2026-04-04 --chunk-days 30`

Then import separately:

- `python3 -m src.cli import-history --config configs/nba.yaml --source-manifest data/raw/historical/nba/manifest.json`

## Design Choices

- Keep cache generation and DB import separate.
- Reuse existing odds bundle and manifest import code instead of building a second import path.
- Skip already-downloaded chunks when their chunk manifest exists.
- Use the existing games table as the source of truth for which events need odds.
- Group chunks by central-date windows so they line up with the ESPN scoreboard historical date keys more naturally than naive UTC-date slicing.

## Risks And Guardrails

- Historical odds endpoints may fail or return sparse coverage for some dates. Chunking keeps retries local.
- Team normalization can drift; the service should reuse the league team fetcher and existing alias logic.
- A chunk may span dates with no games. Those chunks should be skipped instead of producing empty bundles.

## Success Criteria

- We can delete `data/raw/historical/nba/`, rerun the backfill command, and regenerate a valid top-level manifest.
- `import-history` can ingest the regenerated manifest without code changes.
- The resulting odds coverage can be extended far enough to unblock the full NBA research tournament.
