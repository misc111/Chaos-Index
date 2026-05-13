# MLB Daily Automation Policy

Routine daily automation is a fresh-scoring lane, not a full retraining lane.

## Default Daily Path

Use `make daily_score` for scheduled or manual daily evidence runs. It executes
the MLB data refresh steps, then runs current-season predictiveness scoring
against frozen pregame predictions. The path is safe on a dirty worktree because
it does not commit, push, publish, rebuild features, retrain models, or require a
clean `main` checkout.

Use `make daily_score DRY_RUN=1` to verify the exact step plan before execution.
If the live fetch path is blocked by an environment or network failure, use
`make daily_score SCORE_ONLY=1` to refresh current-season predictiveness from
existing frozen pregame ledgers without retrying the data refresh first.

## Adjacent Paths

- `make data_refresh`: data and odds refresh only; no scoring and no retraining.
- `make daily_score SCORE_ONLY=1`: score current-season settled outcomes from
  existing frozen ledgers when live fetch is unavailable.
- `make predictiveness`: score current-season settled outcomes from the frozen
  pregame ledger only.
- `make run_daily`: routine refresh plus scoring; add `RETRAIN=1` only when an
  operator explicitly wants retraining inside that command.
- `make hard_refresh`: integrated rebuild, retraining, staging snapshot, commit,
  push, and publish closeout. This requires a clean `main` worktree.

## Retraining Policy

Retraining should happen when there is a deliberate model checkpoint: feature
contract changes were approved, new evidence supports a champion or challenger
update, data coverage materially changed, validation diagnostics need a saved
model run, or an operator is preparing an integrated staging/publish closeout.

Retraining should not happen during routine daily scoring, while the worktree is
dirty with unrelated edits, to make settled-game predictiveness measurable, as a
side effect of odds/data refresh, or as a workaround for stale staging payloads.
