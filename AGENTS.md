# MLB-First Rebuild Instructions

## Instruction Scope
- Treat this file as the repository contract for the MLB-first actuarial rebuild.
- The repository is being rebuilt around MLB betting and actuarial GLM-family methods.
- Legacy NHL/NBA code may still exist during migration, but it is not the governing product contract.

## Coordinator Mode
- The user has explicitly requested coordinator-style delegation for this rebuild.
- Use subagents aggressively for independent workstreams, but keep write scopes controlled.
- Read `SUBAGENTS.md` before choosing subagent model/routing strategy.
- The coordinator owns decomposition, synthesis, conflict handling, acceptance, and final user-facing judgment.
- Keep overlapping write scopes with one owner whenever possible.

## Theory Governance
- The statistical governing sources for the rebuild are:
  - `statistical_theory/09_GLM_Generalized_Linear_Models_for_Insurance_Rating.pdf`
  - `statistical_theory/10_Holmes_Casotto_Penalized_Regression_and_Lasso_Credibility_2025_Revision.pdf`
- Every major modeling, validation, and governance decision must be traceable to those monographs or explicitly labeled as one of:
  - direct theory implementation
  - theory-compatible engineering support
  - betting overlay
  - dashboard/reporting layer
- Do not blur theory-backed modeling decisions with betting-product overlays.

## Scope Contract
- The active product scope is MLB statistical ensemble betting for:
  - moneyline
  - runline
  - totals
- The repo contract is MLB-first across docs, configs, commands, storage, features, models, validation, dashboard, and staging outputs.
- Questions or implementations that depend on other leagues should be treated as migration debt unless the user explicitly asks for legacy comparison.

## Product Principles
- Pregame-only features and immutable pregame prediction ledgers are mandatory.
- Market awareness should be implemented through explicit vig-free market transforms and offset/complement logic, not by collapsing the system into market-copying.
- The core theory lane is GLM-family, penalized GLM, and lasso-credibility driven.
- Non-CAS model families may survive only as clearly labeled experimental challengers.
- Promotion of any champion model or ensemble requires written evidence.

## Theory-Backed Model Program
- The default modeling program must materially cover the monograph-supported families:
  - vanilla GLMs
  - ridge / lasso / elastic net
  - market-offset and prior-offset lasso credibility
  - GLMM
  - DGLM
  - GAM
  - MARS-style hinge discovery with GLM reuse where sensible
  - theory-compatible ensembles
- All key diagnostics and validation families materially supported by the monographs must be implemented and surfaced.

## Repository Refresh Contract
- During the rebuild, repository-level `data_refresh` and `hard_refresh` flows should be treated as MLB-first orchestration commands.
- If the implementation still carries legacy multi-league plumbing, do not document it as the primary contract.
- Fail fast on required-step errors. Do not silently skip stale or missing MLB outputs.

## Dashboard And Staging Sync
- Treat the local Next.js dashboard and the committed static staging snapshots as separate delivery targets.
- If dashboard code or API payloads change in a way that affects shipped staging, also update `web/public/staging-data/mlb/`.
- Do not assume live SQLite changes update staging automatically.

## Git Workflow
- Keep this repository on `main` unless the user explicitly asks for another branch.
- For this repo, pushing `main` remains the default close-out once an integrated sprint checkpoint is actually ready.
- Do not push partial refresh output from a broken MLB pipeline.

## Sprint Expectations
- Sprint 0 is complete. Do not describe the repo as pre-implementation scaffolding only.
- The active checkpoint spans:
  - Sprint 1 MLB data foundation
  - Sprint 2 MLB-first feature architecture and leakage enforcement
  - Sprint 8 dashboard/staging contract surgery for shipped MLB payloads
- Later sprints remain incomplete, especially the full GLM factory, validation gold standard, ensemble promotion package, and release hardening.
- Each sprint closeout should report:
  - what changed
  - what tests passed
  - what remains
  - blockers
  - exact files touched
  - updated risks and assumptions

## File Linking
- When referencing local files in Codex responses, use clickable Markdown links with absolute paths.
- For paths without spaces, use `[label](/absolute/path/to/file:line)`.
- For paths with spaces, wrap the target in angle brackets so the renderer keeps the full path together: `[label](</absolute/path with spaces/to/file:line>)`.
- This repository lives under `/Users/davidiruegas/Library/Application Support/SportsModeling`, so links to files in this repo usually need the angle-bracket form.
- Example: [AGENTS.md](</Users/davidiruegas/Library/Application Support/SportsModeling/AGENTS.md:1>)
