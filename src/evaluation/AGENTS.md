# Evaluation Rules

- Promotion requires written evidence, not just a leaderboard winner.
- Validation claims must cover calibration, lift, drift, holdout behavior, and governance-lane labels where applicable.
- Do not loosen gates to promote a cosmetically improved model; document blocked status and the reason.
- Keep validation artifacts machine-readable for dashboard/staging consumers and human-readable enough for promotion review.
- Add contract or boundary tests when a validation module takes ownership of a new artifact, gate, metric, or report section.
- Never use Playwright for evaluation work, ever.
