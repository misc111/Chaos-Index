# NHL + NBA Agent Instructions

## Scope Contract
- This project supports NHL and NBA forecasting.
- Interpret user questions in the configured league context by default (`config.data.league`).
- If no config context is available, default to NHL for ambiguous wording.

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
