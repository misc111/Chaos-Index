# Web Dashboard Rules

- Treat the local Next.js dashboard and committed static staging snapshots as separate delivery targets.
- If API payload shape, server repositories, dashboard routes, or UI semantics change, update `web/public/staging-data/mlb/` and `web/public/staging-data/manifest.json` in the same workstream.
- Prefer `npm run test:unit` for server/component contract changes, `npm run lint` for lint-sensitive edits, `npm run typecheck` for TypeScript contract changes, and `npm run test:smoke` for browser-facing flow changes.
- Do not hand-edit large staging JSON to fit a UI assumption; regenerate through the project scripts when possible and explain any unavoidable manual patch.
- Keep shipped staging MLB-only. Do not add root-level retired league directories or manifest entries.

