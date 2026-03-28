# Coordinator Agent Instructions

These instructions apply only when the user explicitly asks for coordinator-style orchestration.

## Instruction Scope

- Treat this file as a thin coordinator overlay for repository-specific constraints only.
- For general planning, delegation, prompt design, polling, verification, review, and completion behavior, follow the agent's normal workflow unless this file states a repo-specific exception.
- Do not infer extra coordinator process rules from this file beyond the explicit local constraints written here.

## Local Constraints

- This workflow must stay in the top-level Codex thread that actually has access to the agent-spawning tools.
- Do not assume a spawned child agent can itself spawn more child agents in this environment.
- Read [AGENTS.md](AGENTS.md) first, then read [SUBAGENTS.md](SUBAGENTS.md) before choosing child-agent models, reasoning levels, or prompt styles.
- These instructions extend, and do not replace, [AGENTS.md](AGENTS.md).
- The coordinator owns task decomposition, routing, synthesis, conflict handling, and final user-facing completion judgment.
- Keep overlapping write scopes to one owner whenever possible. If two lanes are likely to edit the same files, either keep that work with one child agent or keep it coordinator-owned.
- When child agents return work, review and integrate it before treating the task as complete.
- Leave unrelated worktree changes alone.

## Activation Gate

- Only use this mode when the user explicitly asks for coordinator-style delegation, orchestration across multiple workstreams, or equivalent language.
- If the user explicitly asks for coordinator mode, you may proceed without an extra confirmation prompt.
- If the task is small, single-threaded, or blocked on one immediate answer, do not recommend coordinator mode.
- If you are already inside a spawned child agent and do not have child-agent spawning tools, do not attempt to act as the coordinator. Report the limitation and hand orchestration back to the top-level thread instead.
