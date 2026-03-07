# Server Lib

This subsystem holds server-only web business logic behind thin route handlers.

Shared across leagues:
- `repositories/` owns SQLite reads and generated-manifest access.
- `services/` owns route-level orchestration, sorting, shaping, and task execution.
- `payload-contracts.ts` defines the public dashboard/staging payload map consumed by both live API routes and `web/public/staging-data/`.

League-specific:
- League differences belong in manifest lookups, repositories, or explicitly named service policies.
- Route files should not embed league switches, SQL branches, or payload shaping inline.

Input/output contract:
- Inputs are `LeagueCode`, request parameters, and repository results.
- Outputs are JSON payloads with stable shapes for dashboard clients and committed staging snapshots under `web/public/staging-data/{league}/`.
