---
name: mlb-daily-evidence-loop
description: Run the SportsModeling MLB-first daily evidence loop: dirty-worktree preflight, safe MLB data refresh, fresh frozen-ledger scoring, focused verification, and concise daily report. Use for daily MLB predictiveness automation, current-season forecast health checks, or manual MLB evidence-loop runs; do not use for broad refactors, routine retraining, publish closeout, or retired league work.
---

# MLB Daily Evidence Loop

Use this skill to run the repeatable daily MLB evidence loop without broad refactors, retired league work, surprise publishing, or surprise retraining.

## Safety Contract

- Start with `git status --short --untracked-files=all`.
- If unrelated user edits or generated outputs would be overwritten, stop and report exact dirty paths.
- Do not run commands that commit, push, reset, delete, or backfill frozen prediction ledgers unless the user explicitly asks.
- Do not run `make hard_refresh` for routine daily automation; it performs publish closeout behavior after the pipeline. Use its dry run only when checking the integrated plan.
- Do not retrain during routine daily automation. Use `make daily_score` for the default fresh-scoring path, and reserve `make run_daily RETRAIN=1` or `make hard_refresh` for explicit retraining work.
- If live fetch is blocked by an environment/network failure, use `make daily_score SCORE_ONLY=1` to refresh current-season predictiveness from existing frozen ledgers instead of waiting on cache replay.
- `make daily_score` refreshes static staging snapshots after scoring; this is not a publish closeout, but it should be reported as generated output.
- Keep scope MLB-only. Redirect legacy league needs to git history unless explicitly requested.

## Workflow

1. Read `AGENTS.md` and `LEARNINGS.md` enough to capture current rules and open risks.
2. Confirm the routine daily plan with `make daily_score DRY_RUN=1`.
3. Run the safest relevant MLB lane:
   - Default daily scoring: `make daily_score`.
   - Score-only fallback after fetch/network blockage: `make daily_score SCORE_ONLY=1`.
   - Data-only refresh: `make data_refresh`.
   - Explicit retraining checkpoint: `make run_daily RETRAIN=1` for local retraining without publish closeout, or `make hard_refresh` only when the user wants the integrated clean-worktree publish path.
4. If dashboard or API payloads changed outside the routine scoring lane, regenerate staging with `cd web && npm run generate:staging-data`; the routine scoring lane already does this after scoring. Verify the staging contract with the existing project script or tests.
5. Run the fastest focused verification that covers the touched contract, usually targeted `pytest` tests plus relevant `web` unit, lint, or typecheck commands.
6. Report commands run, pass/fail, current-season predictiveness status, scored live predictions count, changed generated files, whether retraining was skipped or explicitly requested, and the next blocker if the result is not measurable.

## Output Shape

Return a concise daily summary:

- `Commands`: commands run and results.
- `Predictiveness`: measurable/not measurable, scored predictions count, and notable metrics when available.
- `Changed files`: generated or source files touched.
- `Verification`: tests/checks run and gaps.
- `Blockers`: exact next blocker, or `none`.
