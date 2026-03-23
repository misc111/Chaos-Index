# Subagent Model Routing Guide

Last updated: 2026-03-17

This file tells Codex which model and reasoning level to use when spawning subagents.

This file covers child-agent model selection and prompt style only.

This file is the source of truth for child-agent model selection and prompt style.

Do not collapse this guide into a fixed `explorer -> model`, `worker -> model`, or `default -> model` lookup table. The coordinator should inspect each delegated task and choose based on task shape, risk, tool needs, and deliverable.

It does not transfer routing ownership, final synthesis, or final acceptance decisions away from the top-level coordinator when coordinator mode is active.

The recommendations below are evidence-first:

- Prefer official OpenAI benchmark tables and model docs over intuition.
- Do not treat all benchmarks as interchangeable.
- Do not assume a model is good at repo editing because it scored well on academic reasoning.
- When OpenAI did not publish accessible benchmark numbers for a model, mark the recommendation as lower-confidence instead of guessing.

## Benchmark Map

Use the benchmark that matches the delegated task:

- `SWE-Bench Pro`: realistic software-engineering patch tasks across multiple languages. Best signal for non-trivial code fixes.
- `Terminal-Bench 2.0`: shell-heavy agent work in real terminal environments. Best signal for CLI, build, setup, and debug loops.
- `OSWorld-Verified`: screenshot-driven computer use. Best signal for GUI/browser automation and visual debugging.
- `GDPval`: professional deliverables across occupations. Best signal for research, documents, spreadsheets, presentations, and business-style outputs.
- `BrowseComp`, `Toolathlon`, `MCP Atlas`, `tau2-bench`: tool use, browsing, and workflow orchestration. Best signal for agent/tool reliability.
- `GPQA Diamond`, `HLE`, `MMMU`, `OmniDocBench`, long-context evals`: useful secondary signals for science reasoning, hard general reasoning, vision, document understanding, and context handling. These are not primary repo-editing benchmarks.

## Evidence Grades

- `A`: official benchmark table plus official model/task description.
- `B`: official model/task description, but no accessible benchmark table in current docs.

## Reasoning Effort Rules

Use reasoning effort conservatively:

- `low`: bounded, well-specified, low-risk execution. Good for grep-and-summarize, tiny edits, narrow transforms, or mechanical follow-through.
- `medium`: default for most subagents. Use for standard code search, medium-complexity file review, bounded bug fixes, and focused document work.
- `high`: use when the task is ambiguous, multi-file, or likely to need iteration and self-correction.
- `xhigh`: reserve for the hardest or highest-stakes delegated work. Published top benchmark numbers often use `xhigh`; do not assume you get those results at lower efforts.

Codex-specific availability in this environment:

- `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.3-codex`, `gpt-5.3-codex-spark`, `gpt-5.2-codex`, `gpt-5.2`, and `gpt-5.1-codex-max` expose `low`, `medium`, `high`, and `xhigh`.
- `gpt-5.1-codex-mini` exposes `medium` and `high`.

## Default Routing

If, after checking the actual delegated task, there is still no strong reason to do otherwise:

1. Use `gpt-5.4-mini` at `medium` for bounded parallel sidecars.
2. Upgrade to `gpt-5.4` at `medium` or `high` when the task mixes coding with tool use, computer use, long context, or business/research judgment.
3. Use `gpt-5.3-codex` at `high` for pure coding tasks where coding quality matters more than broad generalist ability.
4. Use `gpt-5.3-codex-spark` at `low` or `medium` only when latency matters more than depth.
5. Use `gpt-5.2*` and `gpt-5.1*` models mainly for reproducibility, compatibility, or explicit user preference.

## Prompting Families

Prompting guidance is also evidence-based, but it comes from family-level docs rather than benchmark tables:

- General-purpose GPT family: `gpt-5.4`, `gpt-5.4-mini`, and `gpt-5.2`
  - Start lean.
  - Treat reasoning effort as a last-mile knob, not the first fix.
  - Add explicit prompt blocks only when the workflow needs them: `tool_persistence_rules`, `dependency_checks`, `completeness_contract`, `verification_loop`, `citation_rules`, `research_mode`, `parallel_tool_calling`, or a verbosity clamp.
  - Prefer explicit local criteria and examples over long repetitive scaffolding.
- Codex family: `gpt-5.3-codex`, `gpt-5.3-codex-spark`, `gpt-5.2-codex`, `gpt-5.1-codex-max`, and `gpt-5.1-codex-mini`
  - Give durable task context: goal, repo area, constraints, acceptance criteria, and allowed tools.
  - Prompt for persistence and end-to-end completion, not just planning.
  - Add terminal hygiene and verification explicitly.
  - For long-running flows, keep goals and acceptance criteria stable across turns and preserve any host `phase` handling.

## Model-by-Model Guidance

### `gpt-5.4`

Evidence grade: `A`

Use for:

- deep reviewer or planner for the most important delegated task when the coordinator wants high-quality input before making the final call
- hard tasks that mix code, tools, browser/desktop automation, and judgment
- large-repo or long-context investigations
- document, spreadsheet, or presentation generation that also touches code
- high-stakes comparative analysis between competing subagent outputs, while the coordinator still owns final adjudication

Default effort:

- `medium` by default
- `high` for ambiguous multi-step tasks
- `xhigh` only when the task is hard enough that extra latency is acceptable

Why:

- Official model docs position `gpt-5.4` as the frontier model for complex professional work. In Codex app routing, do not assume the API-only 1.05M-token context window is available; treat `gpt-5.4` as the best choice for context-heavy app tasks without relying on that API limit.
- OpenAI reports `gpt-5.4` at 83.0% on `GDPval`, 57.7% on `SWE-Bench Pro`, 75.1% on `Terminal-Bench 2.0`, 75.0% on `OSWorld-Verified`, 54.6% on `Toolathlon`, and 82.7% on `BrowseComp`.
- OpenAI’s latest-model guide says `gpt-5.4` is the default for the most important general-purpose and coding work, and the better default over `gpt-5.3-codex` when the workflow spans software engineering plus planning, writing, or other business tasks.

Prompting techniques:

- Start with a lean task prompt. If the workflow is long-horizon or tool-heavy, add `tool_persistence_rules`, `dependency_checks`, and `verification_loop` before increasing reasoning effort.
- For research, browsing, or citation-sensitive work, add `research_mode`, `citation_rules`, and `empty_result_handling`.
- For batch deliverables, explicitly enumerate the expected outputs and add a `completeness_contract`.
- If output length matters, add a hard verbosity spec instead of repeating style instructions throughout the prompt.

Avoid when:

- the subtask is narrow enough for `gpt-5.4-mini`
- the task is pure coding and you want a specialized coding model with lower cost
- ultra-low latency matters more than depth

### `gpt-5.4-mini`

Evidence grade: `A`

Use for:

- default explorer-style subagents
- parallel codebase search, large-file review, and supporting-document digestion
- bounded debugging, focused patch proposals, and medium-scope implementation
- screenshot interpretation and light computer-use sidecars
- high-volume tool-using workers where strong reasoning still matters

Default effort:

- `medium` by default
- `low` for mechanical search/extract/transform tasks
- `high` when a bounded task is still tricky
- avoid `xhigh` unless you intentionally want to trade away the speed/cost advantage

Why:

- OpenAI explicitly says `gpt-5.4-mini` is its strongest mini model yet for coding, computer use, and subagents.
- OpenAI says it is more than 2x faster than `gpt-5 mini` and approaches `gpt-5.4` on several evals.
- Official scores: 54.4% on `SWE-Bench Pro`, 60.0% on `Terminal-Bench 2.0`, 42.9% on `Toolathlon`, 57.7% on `MCP Atlas`, 72.1% on `OSWorld-Verified`, 88.0% on `GPQA Diamond`, 41.5% on `HLE` with tools, and 76.6% on `MMMU Pro`.
- OpenAI’s own subagent examples position `gpt-5.4-mini` well for narrower subtasks like codebase search, large-file review, and supporting documents.

Prompting techniques:

- Use the same GPT-family prompt blocks as `gpt-5.4`, but keep them shorter and tighter.
- Give one bounded objective, explicit success criteria, and the exact output shape. Mini models benefit from narrower scope and fewer implied obligations.
- Prefer examples, counts, and local rules over abstract policy prose.
- If tools are needed, state whether steps are independent enough for parallel retrieval or must stay sequential.

Avoid when:

- the task is the single most important delegated work item
- you need the best possible long-horizon judgment across mixed modalities and tools
- the task is tiny enough that `gpt-5.3-codex-spark` is the better latency play

### `gpt-5.3-codex`

Evidence grade: `A`

Use for:

- pure coding specialists
- repo-wide refactors, migrations, and feature builds inside code-first environments
- terminal-heavy engineering work
- frontend generation when code quality matters more than raw speed
- long-running coding tasks that do not need `gpt-5.4`'s broader professional-work advantages

Default effort:

- `high` by default for substantial coding tasks
- `medium` for moderate implementation work
- `xhigh` for the hardest refactors, debugging investigations, or long-horizon builds

Why:

- OpenAI’s model page calls it the most capable agentic coding model to date and says it is optimized for agentic coding in Codex-like environments.
- OpenAI reports 56.8% on `SWE-Bench Pro` and 77.3% on `Terminal-Bench 2.0`, with all blog evals run at `xhigh`.
- OpenAI says it is 25% faster than `gpt-5.2-codex`, uses fewer tokens than prior models, and is built to take on long-running tasks that involve research, tool use, and complex execution.
- OpenAI also says it supports the full software lifecycle: debugging, deploying, monitoring, PRDs, tests, metrics, and more.

Prompting techniques:

- Prompt it like an autonomous coding agent: state the goal, the code area, constraints, and the acceptance checks up front.
- Ask it to inspect the codebase first, then act, then verify. Do not ask for a plan-only response unless you truly want one.
- Include explicit terminal hygiene and verification expectations: what commands are allowed, what tests or builds to run, and what evidence counts as done.
- For long-running or replayed flows, keep durable task context stable and preserve any `phase` handling instead of rephrasing the task every turn.

Avoid when:

- the task mixes coding with heavier business/research deliverables
- you need the broader cross-domain strength of `gpt-5.4`, especially for context-heavy work in the app, without assuming the API-max context window
- low cost per sidecar matters more than maximum coding strength

### `gpt-5.3-codex-spark`

Evidence grade: `B`

Use for:

- ultra-fast interactive workers
- tiny patches, local logic reshaping, quick UI tweaks, and fast code Q&A
- rapid iteration loops where the human or parent agent expects an answer almost immediately
- low-stakes scratch work where retrying is cheap

Default effort:

- `low` by default
- `medium` if the task is still small but slightly subtle
- avoid `high` and `xhigh` unless measured latency is still acceptable and the task remains well-bounded

Why:

- OpenAI describes it as a smaller version of `gpt-5.3-codex`, designed for real-time coding and delivering more than 1000 tokens per second.
- OpenAI says it makes minimal, targeted edits and does not automatically run tests unless asked.
- OpenAI says it shows strong performance on `SWE-Bench Pro` and `Terminal-Bench 2.0` while finishing in a fraction of the time of `gpt-5.3-codex`, but the accessible release text does not publish the numeric scores.

Prompting techniques:

- Keep the prompt single-scope: one bug, one file cluster, one question, or one tiny patch.
- Ask for minimal targeted edits and make the change budget explicit.
- Say explicitly whether to run tests or not. Do not assume Spark will perform broader verification unless told to.
- Avoid vague multi-stage autonomy prompts. If the task can branch widely, route it upward instead.

Avoid when:

- the task needs deep autonomous iteration
- running tests by default is important
- the task is large, ambiguous, or high stakes
- you need image input or long context beyond its current text-only 128K profile

### `gpt-5.2-codex`

Evidence grade: `A`

Use for:

- legacy/reproducibility runs against older Codex behavior
- strong coding work when `gpt-5.3-codex` or `gpt-5.4` are unavailable
- Windows-heavy coding tasks
- defensive cybersecurity-related coding work when you explicitly want this model generation

Default effort:

- `medium` for standard legacy coding work
- `high` for multi-file fixes
- `xhigh` for the hardest legacy-comparison tasks

Why:

- OpenAI describes it as an upgraded `gpt-5.2` optimized for agentic coding and long-horizon work.
- OpenAI says it achieved state-of-the-art performance on `SWE-Bench Pro` and `Terminal-Bench 2.0` at release, with stronger long-context understanding, reliable tool calling, factuality, compaction, Windows performance, and cybersecurity capability.
- In the later `gpt-5.3-codex` appendix, OpenAI reports 56.4% on `SWE-Bench Pro`, 64.0% on `Terminal-Bench 2.0`, 38.2% on `OSWorld-Verified`, 67.4% on professional CTF challenges, and 76.0% on `SWE-Lancer IC Diamond`.

Prompting techniques:

- Use the same Codex-family prompt skeleton as `gpt-5.3-codex`: durable context, explicit acceptance checks, and mandatory verification.
- Be more explicit about environment details when they matter, especially OS, shell, security boundaries, and allowed tools.
- If the reason for choosing this model is reproducibility, freeze the task contract tightly so the run is comparable to older results.

Avoid when:

- the task is new work and a later model is available
- you do not specifically need legacy behavior or compatibility

### `gpt-5.2`

Evidence grade: `A`

Use for:

- legacy general-purpose professional work
- reproducing or comparing against prior frontier-model behavior
- non-code knowledge work where you specifically want the `gpt-5.2` generation

Default effort:

- `medium` by default
- `high` for hard document/research tasks
- `xhigh` only for difficult judgment-heavy legacy runs

Why:

- OpenAI’s model page calls it the previous frontier model for complex professional work and recommends `gpt-5.4` instead for new work.
- OpenAI reports 70.9% on `GDPval`, 55.6% on `SWE-Bench Pro`, 92.4% on `GPQA Diamond`, and 45.5% on `HLE` with search and Python.
- This is a strong legacy generalist, but not the best current default if newer models are available.

Prompting techniques:

- Use the GPT-family prompting style: start lean, then add `research_mode`, `citation_rules`, `verification_loop`, or structured-output constraints only if the workflow needs them.
- If the run is for comparison against older behavior, match the current reasoning setting first before tuning prompt blocks.
- For business or research outputs, specify the deliverable sections and required grounding explicitly instead of relying on implied format.

Avoid when:

- the task is a new delegation and a newer model is available
- the task is code-first and a Codex-specialized model is available

### `gpt-5.1-codex-max`

Evidence grade: `A`

Use for:

- legacy long-horizon coding
- compaction-heavy tasks and extended coding sessions
- historical reproduction against `gpt-5.1`-era Codex behavior
- complex debugging or refactoring where you specifically want this older Codex generation

Default effort:

- `medium` for most tasks
- `xhigh` for non-latency-sensitive hard problems

Why:

- OpenAI says it is built for long-running, detailed work and can operate coherently across multiple context windows via compaction.
- OpenAI explicitly recommends `medium` as the daily driver and says `xhigh` is for non-latency-sensitive tasks.
- Official numbers: 77.9% on `SWE-bench Verified`, 79.9% on `SWE-Lancer IC SWE`, and 58.1% on `Terminal-Bench 2.0` at `xhigh`.
- OpenAI also says `medium` beats base `gpt-5.1-codex` at the same effort while using 30% fewer thinking tokens.

Prompting techniques:

- Prompt it for long-horizon persistence: durable objective, constraints, acceptance tests, and explicit follow-through to completion.
- Use `medium` first and move to `xhigh` only when the task is hard enough that latency is not the main concern.
- Because this model is used for longer runs and compaction-heavy work, keep the invariant task contract stable across turns instead of rewording it repeatedly.

Avoid when:

- you do not need legacy behavior
- a newer coding model is available

### `gpt-5.1-codex-mini`

Evidence grade: `B`

Use for:

- cheapest legacy coding sidecars
- short, simple, clearly bounded coding support tasks
- mechanical transforms, boilerplate, file-local edits, and simple code Q&A
- low-stakes background helpers when you want to preserve stronger models for harder work

Default effort:

- `medium` by default
- `high` only when the task is still short but needs a bit more care

Why:

- OpenAI’s current model page describes it as a smaller, more cost-effective, less-capable version of `gpt-5.1-codex`.
- I did not find a current official benchmark table for this model in the accessible docs, so treat this as a budget helper, not a frontier performer.

Prompting techniques:

- Keep prompts extremely concrete: one small target, one or two constraints, and one verification step.
- Prefer explicit examples and exact output shapes over broad autonomy.
- Avoid multi-phase tasks, wide codebase exploration, or ambiguous success criteria.

Avoid when:

- correctness matters a lot
- the task spans multiple files or needs deeper planning
- a faster and stronger mini model like `gpt-5.4-mini` is available

## Practical Task Mapping

Use this as the shortest path to a choice:

- Codebase search, large-file reading, supporting-doc processing: `gpt-5.4-mini` at `medium`
- Small mechanical patch or instant code Q&A: `gpt-5.3-codex-spark` at `low`
- Medium bug fix in a known area: `gpt-5.4-mini` at `medium`
- Large refactor or migration in a code-first task: `gpt-5.3-codex` at `high`
- Mixed coding plus research, docs, or tool orchestration: `gpt-5.4` at `medium`
- GUI/browser debugging from screenshots: `gpt-5.4` at `high`, or `gpt-5.4-mini` at `medium` for cheaper sidecars
- Legacy comparison or reproducibility: `gpt-5.2-codex`, `gpt-5.2`, or `gpt-5.1-codex-max` as needed
- Cheapest safe legacy helper: `gpt-5.1-codex-mini` at `medium`

## Anti-Patterns

Do not:

- use `gpt-5.3-codex-spark` as the default worker for high-stakes coding just because it is fast
- use `gpt-5.1-codex-mini` when `gpt-5.4-mini` is available and quality matters
- use `gpt-5.2` for code-first work if a Codex model is available
- route by one benchmark alone; match the benchmark family to the delegated task
- assume published `xhigh` benchmark wins carry over to `low` or `medium`

## Sources

Official OpenAI sources used for this file:

- [Introducing GPT-5.4](https://openai.com/index/introducing-gpt-5-4/)
- [GPT-5.4 model page](https://developers.openai.com/api/docs/models/gpt-5.4)
- [Using GPT-5.4 / latest model guide](https://developers.openai.com/api/docs/guides/latest-model)
- [Codex Prompting Guide](https://developers.openai.com/codex/prompting-guide)
- [Introducing GPT-5.4 mini and nano](https://openai.com/index/introducing-gpt-5-4-mini-and-nano/)
- [GPT-5.4 mini model page](https://developers.openai.com/api/docs/models/gpt-5.4-mini)
- [Introducing GPT-5.3-Codex](https://openai.com/index/introducing-gpt-5-3-codex/)
- [GPT-5.3-Codex model page](https://developers.openai.com/api/docs/models/gpt-5.3-codex)
- [Introducing GPT-5.3-Codex-Spark](https://openai.com/index/introducing-gpt-5-3-codex-spark/)
- [Introducing GPT-5.2-Codex](https://openai.com/index/introducing-gpt-5-2-codex/)
- [GPT-5.2-Codex model page](https://developers.openai.com/api/docs/models/gpt-5.2-codex)
- [Introducing GPT-5.2](https://openai.com/index/introducing-gpt-5-2/)
- [GPT-5.2 model page](https://developers.openai.com/api/docs/models/gpt-5.2)
- [Building more with GPT-5.1-Codex-Max](https://openai.com/index/gpt-5-1-codex-max/)
- [GPT-5.1-Codex-Max model page](https://developers.openai.com/api/docs/models/gpt-5.1-codex-max)
- [GPT-5.1-Codex-mini model page](https://developers.openai.com/api/docs/models/gpt-5.1-codex-mini)
