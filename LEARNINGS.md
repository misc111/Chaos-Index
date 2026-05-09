# LEARNINGS.md

Durable project memory for the MLB-first rebuild. This file captures lessons
learned through trial, bugs, refactors, validation runs, and dashboard staging
work. It is not a changelog and not a replacement for `AGENTS.md`.

## How To Use This File

- Read this before planning or writing project code.
- Add lessons only when they are likely to prevent repeated mistakes.
- Keep entries concise: date, source, lesson, and action.
- Do not store secrets, raw transcripts, or one-off command output.
- If a lesson becomes a permanent repo rule, promote the distilled rule to
  `AGENTS.md`; keep the historical lesson here.
- If a lesson becomes stale, mark it superseded instead of deleting it.

## Community Pattern Adopted

The useful community pattern is separation of concerns:

- `AGENTS.md` is the short always-loaded project contract and routing file.
- `LEARNINGS.md` is the durable memory of mistakes, corrections, and hard-won
  project knowledge.
- Detailed architecture, plans, validation evidence, and raw logs belong in
  focused docs or artifacts, with links from learnings when useful.
- Repeated learnings should become tests, automation, or short agent rules.
- Stale memory is dangerous; entries need dates, sources, and a current action.

## Lessons From 2026-04-28 To 2026-05-04

### 2026-05-01 - MLB-first means the shipped surface, not just the docs

Source: `252de50`, `docs/mlb/target_architecture.md`, registry and staging tests.

Lesson: Retired league artifacts, registry rows, generated manifests, and static
dashboard payloads can quietly keep old product scope alive. MLB-first only holds
when commands, configs, model registries, reports, dashboard routes, and staging
snapshots all agree.

Action: New work should route through MLB-only registries and update guard tests
when touching league scope. Recover retired NBA/NHL behavior from git history
only if the user explicitly asks for that scope.

### 2026-05-01 - Theory labels must travel with every artifact

Source: `b397ab2`, `src/governance/evidence.py`,
`src/evaluation/validation_governance.py`, `docs/mlb/validation_requirements.md`.

Lesson: The repo now has four distinct lanes: direct theory implementation,
theory-compatible engineering support, betting overlay, and dashboard/reporting
layer. If payloads or validation rows omit those labels, betting product metrics
can get mistaken for actuarial model evidence.

Action: Any new validation task, tournament result, promotion packet, dashboard
payload, or staging snapshot that makes model-quality claims should carry an
explicit governance lane.

### 2026-05-01 - Static staging is a separate delivery target

Source: `b397ab2`, `a6d2fe7`, `web/public/staging-data/mlb/`.

Lesson: The Next.js dashboard and committed staging snapshots do not update each
other automatically. API contract changes can look correct locally while shipped
static payloads remain stale.

Action: When dashboard code, API payload shape, or MLB snapshot semantics change,
update `web/public/staging-data/mlb/` and `web/public/staging-data/manifest.json`
in the same workstream.

### 2026-05-02 - Daily predictiveness needs contract tests, not eyeballing

Source: `a6d2fe7`, `tests/test_current_season_predictiveness.py`,
`src/services/current_season_predictiveness_*`.

Lesson: Current-season predictiveness touches validation artifacts, bet history,
market boards, staging snapshots, and UI payloads. The outputs are too large and
too cross-cutting for manual inspection to be the safety net.

Action: Keep the current-season predictiveness contract in tests and generated
metadata. Prefer narrow contract assertions over hand-reviewing huge JSON diffs.

### 2026-05-04 - Validation gates beat calibration cosmetics

Source: `23e5fa2`, `docs/mlb/weather_context_broad_platt_gate_autopsy.md`.

Lesson: A Platt calibrator can improve ECE while compressing probabilities into a
narrow band that carries weak lift. The weather-context ridge family looked better
after calibration, but still failed calibration slope/intercept, CV-validation
drift, and practical signal checks.

Action: Do not loosen gates to promote a calibrated variant. Promotion requires
the full validation story: calibration, lift, drift, and holdout evidence.

### 2026-05-04 - Structured feature slates must prove retained context

Source: `23e5fa2`, `src/research/nested_tournament_features.py`,
`tests/test_mlb_nested_tournament.py`.

Lesson: The `weather_context_broad_platt_calibrated` name overstated the actual
retained slate. In the reviewed run it effectively retained rest/travel aliases,
not real run-environment features.

Action: Guard structured variants at construction time. Weather/run-environment
slates must retain enough requested width and at least one real context feature
before entering the tournament.

### 2026-05-04 - Promotion packets need written evidence, not just winners

Source: `1b41e9e`, `src/services/model_compare.py`,
`tests/test_research_evidence_governance.py`,
`web/public/staging-data/mlb/research-desk.json`.

Lesson: A family champion or promising challenger is not enough. The system needs
review packets that explain evidence, gating status, and why a model is promoted,
blocked, or left experimental.

Action: Keep promotion decisions evidence-backed and visible in research-desk
payloads. Separate champion selection from betting overlay appeal.

### 2026-05-04 - Latest research and promotion evidence are different pointers

Source: current working tree, `src/services/model_compare.py`,
`web/app/api/nested-tournament/route.ts`,
`web/scripts/staging-contract.ts`.

Lesson: A latest artifact pointer can mean either "latest research
recommendation" or "latest full-ledger promotion-review evidence." Reusing one
`current_best_models.json` interpretation across dashboards lets smoke,
fixture, or tournament outputs look eligible for promotion.

Action: Keep `latest_artifact_role`, `promotion_eligible`, and
`evidence_status` on dashboard/staging payloads. Fixture/demo/smoke and nested
tournament evidence must stay research-only; promotion-eligible latest pointers
must be separate from research recommendation pointers.

### 2026-05-04 - Feature diagnostics belong to fit artifacts

Source: `2761801`, `web/lib/server/repositories/model-feature-map.ts`,
`web/lib/server/repositories/model-feature-map.test.ts`.

Lesson: Dashboard feature diagnostics should come from training and fit artifacts,
not from duplicated frontend assumptions. Otherwise the UI can describe the wrong
features for the model that actually produced a prediction.

Action: Wire feature maps from persisted fit artifacts and keep repository tests
around fallback behavior, diagnostics, and staging payload shape.

### 2026-05-04 - Frozen predictions require first-pitch proof

Source: current working tree, `src/services/train.py`, `src/storage/schema.py`,
`tests/test_prediction_history_contract.py`,
`tests/test_training_artifact_contract.py`.

Lesson: Pregame-only prediction ledgers need the scheduled first pitch alongside
`as_of_utc`. A prediction row without `start_time_utc`, or with `as_of_utc` at or
after first pitch, is not an immutable pregame forecast.

Action: Preserve `start_time_utc` in `predictions`, `prediction_diagnostics`, and
`upcoming_game_forecasts`. Fail fast before persisting any frozen prediction that
cannot prove strict pregame timing.

### 2026-05-04 - Feature-set metadata is mandatory for training outputs

Source: current working tree, `src/services/train.py`,
`tests/test_training_artifact_contract.py`.

Lesson: Falling back to `unknown_feature_set` destroys traceability. Frozen
predictions and model artifacts must point to the feature set that produced them.

Action: Training should fail when `feature_sets` metadata is missing. Run the
feature build before training so every prediction can be tied to an explicit
`feature_set_version`.

### 2026-05-05 - Outcome-informed candidate ranking must be fold-local

Source: DGLM margin evidence packet review, `src/research/model_comparison.py`,
`tests/test_dglm_margin.py`.

Lesson: Ranking DGLM dispersion features from realized score margins is a valid
fit-window modeling choice, but doing it once on the full CV window leaks inner
validation outcomes into parameter selection.

Action: Any outcome-informed candidate feature ranking used inside tuning must
be recomputed on each inner training fold, or the resulting evidence must be
explicitly labeled outside CV validation.

### 2026-04-29 - Split big validation and tournament modules before they harden

Source: `5433ba3`, `f64f2ac`, validation and nested-tournament module tests.

Lesson: Validation and tournament logic became easier to govern once split into
task, support, artifact, feature, evaluation, parallel, sequential, and reporting
modules. Monoliths hide leakage boundaries and make acceptance tests vague.

Action: Keep future validation/tournament additions in the focused modules. Add a
contract or boundary test when a new module owns a new responsibility.

## Open Risks To Revisit

- Full production champion promotion is still blocked by bounded-data evidence,
  stronger market complements, and long-horizon validation depth.
- The GLM factory is only partially hardened; later sprints still need deeper
  diagnostics, ensemble validation, and release hardening.
- Static staging snapshots remain high-churn artifacts. Keep contract tests tight
  so regenerated JSON does not become noise.
