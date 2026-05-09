# SportsModeling Code Review Policy

Review for behavioral risk first. Style-only comments are out of scope unless they hide a correctness, maintainability, or evidence problem.

## Review Priorities

- MLB-first scope: no revived NBA/NHL/other-league behavior unless explicitly requested.
- Pregame integrity: features, predictions, diagnostics, and ledgers must prove `as_of_utc` is strictly before `start_time_utc`.
- Leakage: reject post-start outcomes, same-game final stats, market data without explicit availability timing, and accidental target proxies.
- Theory governance: modeling and validation changes must be labeled as direct theory implementation, theory-compatible engineering support, betting overlay, or dashboard/reporting layer.
- Market handling: vig-free transforms, offsets, and complements must be explicit; do not collapse the model into market copying.
- Promotion evidence: champion or ensemble promotion requires written evidence, calibration/lift/drift/holdout support, and visible gating status.
- Staging sync: dashboard/API payload shape changes must update `web/public/staging-data/mlb/` and `web/public/staging-data/manifest.json`.
- Generated artifacts: manifests, generated docs, and staging payloads should be regenerated through project commands, not hand-edited.
- Safety: flag commands or automations that commit, push, reset, delete, or overwrite user work unexpectedly.

## Expected Evidence

- Cite exact files and line numbers for findings.
- Name the command or test that catches the issue when one exists.
- If no test exists, state the missing test or contract assertion.
- Separate blocker findings from residual risks and open questions.

