# Historical Odds Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a cache-first historical-odds backfill command that downloads chunked odds bundles, rebuilds the league manifest, and leaves DB import as a separate explicit step.

**Architecture:** Add a small service that reads games from the league DB, slices a central-date window into chunks, writes manifest-backed odds bundles via the existing odds helper, and then synthesizes a top-level historical manifest. Wire that service into the CLI and Makefile without changing the existing `import-history` path.

**Tech Stack:** Python, pandas, SQLite, existing ESPN odds client, pytest

---

### Task 1: Add Historical Odds Cache Service

**Files:**
- Create: `src/services/historical_odds_backfill.py`
- Test: `tests/test_historical_odds_backfill.py`

- [ ] **Step 1: Write failing service tests**
- [ ] **Step 2: Run them and verify they fail**
- [ ] **Step 3: Implement chunking, games export, chunk bundle creation, and top-level manifest rebuild**
- [ ] **Step 4: Run tests and verify they pass**

### Task 2: Wire The CLI

**Files:**
- Modify: `src/commands/data.py`
- Modify: `src/registry/commands.py`
- Modify: `Makefile`

- [ ] **Step 1: Add command args and registry entry**
- [ ] **Step 2: Add command handler**
- [ ] **Step 3: Add Make target and optional arg plumbing**
- [ ] **Step 4: Verify parser/help path**

### Task 3: Regenerate Manifests And Verify

**Files:**
- Modify: `configs/generated/command_manifest.json`
- Modify: `docs/generated/commands.md`

- [ ] **Step 1: Run targeted pytest for new service and import compatibility**
- [ ] **Step 2: Regenerate command manifests/docs**
- [ ] **Step 3: Run registry/type checks as needed**
- [ ] **Step 4: Smoke the new backfill command in dry real usage if feasible**
