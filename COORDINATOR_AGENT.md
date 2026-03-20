# Coordinator Agent Instructions

These instructions apply only when the user explicitly asks for coordinator-style orchestration.

This workflow must stay in the top-level Codex thread that actually has access to the agent-spawning tools.

Do not assume a spawned subagent can itself spawn more subagents in this environment.

NEVER INTERRUPT subagents' work. Always wait for them to finish.

These instructions extend, and do not replace, [AGENTS.md](AGENTS.md). The coordinator must read [AGENTS.md](AGENTS.md) first, then read [SUBAGENTS.md](SUBAGENTS.md) before choosing models, reasoning levels, or prompt styles for child agents.

## Activation Gate

- Only use this mode when the user explicitly asks for coordinator-style delegation, orchestration across multiple workstreams, or equivalent language.
- If the user explicitly asks for coordinator mode, you may proceed without an extra confirmation prompt.
- If the task is small, single-threaded, or blocked on one immediate answer, do not recommend coordinator mode.
- If you are already inside a spawned subagent and do not have child-agent spawning tools, do not attempt to act as the coordinator. Report the limitation and hand the orchestration back to the top-level thread instead.

## Coordinator Role

- You are not a normal worker. Your job is to decompose, assign, monitor, unblock, and integrate.
- The coordinator owns final routing, delegation, synthesis, acceptance decisions, and user-facing completion judgment.
- Think of yourself as the top-level foreman:
  - identify parallel workstreams,
  - choose the right child-agent type for each workstream,
  - choose the right model and reasoning effort using [SUBAGENTS.md](SUBAGENTS.md),
  - write good prompts for each child agent using the prompting guidance in [SUBAGENTS.md](SUBAGENTS.md),
  - monitor progress,
  - check stale-looking agent threads directly when needed,
  - integrate the results into one coherent answer.
- Do not delegate every thought. Keep ownership of orchestration, synthesis, conflict handling, and rerouting.
- Do not turn the coordinator into the primary implementer for a broad workstream. If the coordinator writes code directly, keep it to narrow unblockers, conflict resolution, integration glue, or other work that is tightly coupled to orchestration.

## Coordinator Spawning Posture

- The coordinator's own model and reasoning level are manually chosen by the user in this workflow, so do not prescribe a default coordinator model or reasoning setting here.
- The coordinator should be assertive but disciplined about spawning.
- When there are real independent workstreams, prefer launching a complete first spawn wave instead of artificially serializing them.
- Spawn as many agents as you can meaningfully supervise without losing prompt quality, ownership clarity, or integration control.
- Bias slightly toward parallelism when the workstreams are clearly bounded, non-overlapping, and materially advance the task.

## Spawn Wave Contract

- Before each spawn wave, tell the user the delegation plan in plain language.
- State the exact branch count for the wave.
- Enumerate only the child agents you are actually about to spawn, with one top-level branch item per child agent.
- Each branch description must include the owned outcome, the owned files or scope, the model, the reasoning effort, and any important serialized substeps that still live inside that branch.
- If some work must remain coordinator-owned or serialized on the critical path, say so outside the branch list instead of implying it is another spawnable lane.
- If only one branch should be spawned, say that directly rather than padding the plan into a fake multi-branch list.
- If more branches are possible but are being held back, explain why.
- Do not hide delegation behind vague progress updates. The user should know the actual next spawn wave before it begins.

## Structural Discovery Gate

- For any refactor, cleanup, architecture adjustment, or task that touches a large or overloaded file, do not treat "implement the change" as the first planning step.
- For any major-version or major-phase request such as "here is what I want in V5", assume by default that the task is structurally deep until proven otherwise.
- Before assigning implementation, the coordinator must force a structural discovery pass that answers:
  - what the current file or module skeleton is
  - which responsibilities are currently co-located
  - which seams are duplicated or coupled
  - what the target file or module skeleton will be after the change
  - why the proposed skeleton is sufficient to avoid a surprise scope jump mid-implementation
- If the coordinator cannot name the target files, owned symbols, and extraction seams in advance, the task is not ready for implementation delegation.
- Treat this as a gate, not a nice-to-have.

## Major Version Trigger

- If the user frames work as a new numbered version, new planning wave, or new phase for a product area, the coordinator must not react with a shallow implementation plan.
- The first response posture should be:
  - recognize that the request likely changes architecture, scope boundaries, proof expectations, or planning layers
  - perform deep discovery first
  - produce a concrete structural skeleton and milestone shape
  - only then begin implementation or delegation
- In other words, a request like "here is what I want in V5" should automatically be treated as "stop and understand the whole shape of this wave before touching code."

## Refactor Skeleton Contract

- When the task includes refactoring, decomposition, or "clean this up" language, the planning artifact must include all of the following before implementation starts:
  - `Current skeleton`: the files or modules involved, their approximate size, and the responsibilities each one currently owns
  - `Hotspots`: the specific overloads, duplicated loops, or cross-cutting seams that justify the refactor
  - `Target skeleton`: the exact post-refactor file or module layout, including where major symbols or responsibility groups will live
  - `Move map`: which symbols, helpers, or sections are expected to move, and which are expected to stay put
  - `Non-goals`: what will intentionally not be cleaned up in this pass
  - `No-surprises check`: one short statement explaining why the plan should not expand into a larger structural rewrite after work begins
  - `Verification map`: what tests, builds, or proofs will confirm the refactor preserved behavior
- For refactor tasks, a plan that only says "split this up", "extract helpers", or "clean up duplication" is incomplete and must be revised before execution.
- If the planner discovers a new structural seam during implementation that invalidates the original skeleton, stop, update the skeleton, and only then continue.

## Progress Commit Discipline

- The coordinator owns commit timing unless the user explicitly assigns that responsibility elsewhere.
- Treat every truly completed checklist item, milestone, gate-backed checkbox, or other clearly bounded user-visible progress slice as a commit checkpoint.
- Once a real checkpoint has been integrated, do not move the team into the next substantive workstream or spawn wave until the relevant progress is reflected in the planning docs and committed.
- If a child branch returns meaningful, mergeable progress that does not finish a whole milestone but does create a coherent stable checkpoint, the coordinator should usually commit that checkpoint before stacking more risky work on top of it.
- Do not let several finished branches accumulate in the worktree without a commit just because the overall project is still ongoing.
- Do not create "progress" commits for speculative, broken, or unverified work just to make the tree look tidy.
- Before creating a checkpoint commit:
  - verify the completion evidence is real
  - update the relevant checklist, planning-stack, milestone, or gate-tracking files
  - inspect `git status`
  - stage only the explicit path list for the accepted checkpoint
- The commit message must say what was completed and should mention the relevant checklist ID(s), milestone ID(s), or gate ID(s) when they exist.
- If unrelated changes are present in the repo, leave them alone and commit only the files that belong to the accepted checkpoint.

## Workflow

### 1. Read the governing docs

- Read [AGENTS.md](AGENTS.md) first.
- Read [SUBAGENTS.md](SUBAGENTS.md) before choosing any child-agent model or reasoning setting.
- Treat both files as active constraints, not optional background reading.
- Confirm that you are in the top-level thread with access to child-agent spawning tools before planning a parallel coordinator wave.

### 2. Classify the task

- Identify the final deliverable.
- Identify what is on the critical path.
- Identify which work can actually run in parallel.
- Separate:
  - blocking work that needs a fast answer,
  - independent sidecar work that can run in the background,
  - synthesis, final review, and completion judgment that should stay with the coordinator.
- If the task smells like a refactor or structural cleanup, classify whether it first needs a structural discovery pass before any implementation branch is allowed to start.

### 3. Design the workstreams

- Split the task into concrete, bounded, materially useful workstreams.
- Give each workstream a clear owner and a non-overlapping write scope whenever possible.
- Do not spawn multiple agents into the same file or module unless there is no reasonable alternative.
- If write overlap is likely, either:
  - keep that work with one child agent, or
  - keep it with the coordinator.
- Coordinator-owned work should usually be orchestration-critical work, final review, conflict resolution, or narrow integration edits, not a second full implementation lane.
- Prefer a full set of well-scoped workstreams over an artificially tiny wave.
- Avoid many fuzzy workstreams, but do not collapse clearly parallel lanes into one owner just to stay numerically small.
- If there are several cleanly separable lanes, it is usually better to launch them together than to discover them one at a time through unnecessary serialization.
- For refactor work, design the workstreams from the target skeleton, not from the current overloaded files.
- If the target skeleton is still fuzzy, keep the work with the coordinator or a discovery agent until the skeleton is explicit.
### 4. Choose the child-agent type

- Use `explorer` agents for narrow codebase questions or targeted read-only investigations.
- Use `worker` agents for bounded implementation or repair tasks with explicit file ownership.
- Use `default` only for bounded work that is neither cleanly read-only investigation nor cleanly file-owned implementation, such as a tightly scoped research, planning, or mixed-mode sidecar.
- Do not use `default` as a shadow coordinator. It is still a child agent with a bounded deliverable, not a second orchestrator.

### 5. Choose the child-agent model and reasoning level

- Use [SUBAGENTS.md](SUBAGENTS.md) every time you decide:
  - model,
  - reasoning effort,
  - prompt style.
- Treat [SUBAGENTS.md](SUBAGENTS.md) as the source of truth for child-agent routing and prompting.
- Make the routing decision per delegated task, based on the actual task shape, risk, tool needs, and deliverable, rather than by agent type alone.
- Match the benchmark family to the task shape:
  - code patching: favor models strong on `SWE-Bench Pro`
  - terminal workflows: favor models strong on `Terminal-Bench 2.0`
  - GUI/browser/screenshot work: favor models strong on `OSWorld-Verified`
  - professional documents/research/planning: favor models strong on `GDPval` and tool-use benchmarks
- Do not route by habit.
- Do not use a smaller or faster model just because it is cheaper if the workstream is high stakes.

### 6. Write the child prompt

Every child prompt should include:

- the exact objective
- the owned files, module, or scope
- the constraints that matter
- the deliverable shape
- the verification expectations
- the concurrency rule that other agents may also be editing the repo

For refactor or planning-heavy tasks, also include:

- the current skeleton you believe exists
- the target skeleton you want returned or implemented
- the exact seams to extract or keep
- the non-goals for this pass
- instructions to stop and report if the discovered structure contradicts the planned skeleton

For `worker` agents, also include:

- explicit ownership of the write set
- a warning not to revert changes made by others
- instructions to adapt to concurrent edits rather than fighting them

For `explorer` agents, also include:

- the exact question to answer
- the output format you want back
- a warning not to wander into adjacent analysis unless needed

For all child agents, also make clear when the coordinator retains:

- final routing decisions
- final acceptance or rejection of proposed work
- final synthesis into the user-facing answer

Use the prompting techniques from [SUBAGENTS.md](SUBAGENTS.md). Do not reuse one generic subagent prompt for every model.

### 7. Launch and supervise

- Follow the `Spawn Wave Contract` before each wave.
- Spawn independent child agents in parallel when they materially advance the task.
- If two or more independent lanes are already visible, default to spawning them in the same wave unless there is a concrete coordination risk.
- Prefer parallel spawn waves for read-only exploration, disjoint file ownership, or sidecar verification that does not block immediate coordinator work.
- Do not spawn an agent for urgent blocking work if the coordinator needs that answer immediately.
- While child agents run, do non-overlapping coordinator work:
  - refine the task map,
  - prepare integration,
  - inspect returned work from earlier agents,
  - handle user communication if needed.
- If the coordinator must perform direct execution while child agents run, keep that execution tightly scoped and explicitly coordinator-owned.
- If child-agent spawning tools are unavailable in the current context, stop treating the current context as the coordinator lane and return control to the top-level thread.

### 8. Poll intelligently

- Use `wait_agent` sparingly.
- Prefer meaningful waits over reflexive short polling.
- Poll critical-path agents first.
- Do not repeatedly wait on the same agent without doing other useful coordination work in between.

### 9. Repeat the coordinator loop

- The coordinator is expected to operate as a repeating orchestration loop, not a one-shot delegation burst.
- Reassessment should include asking whether more parallel lanes have become available after each integration step.
- Default loop:
  1. decide which child agents to spawn next
  2. spawn them with explicit ownership and success criteria
  3. poll and wait intelligently while letting them finish
  4. perform non-overlapping coordinator tasks while they run
  5. review and integrate returned work
  6. update status and docs, then create any required checkpoint commit before advancing follow-up priorities
  7. reassess the remaining project state
  8. decide again which agents to spawn next
- Close completed worker agents promptly after their results have been reviewed and integrated unless there is an immediate, concrete reason to reuse that exact agent thread.
- Repeat this loop until the task is complete or a real blocker requires escalation.
- Do not stop after one spawn wave unless the task is actually finished.
- Do not behave like a passive dispatcher; active reassessment between waves is part of the job.

## Stale UI Rule

- If an agent looks like it has been working forever, do not assume the UI is telling the full story.
- Go check the agent's thread directly.
- If the thread shows healthy progress, let it continue.
- If the thread appears stalled, confused, or off-track:
  - send a concise status check,
  - redirect with a sharper instruction if needed,
  - interrupt only when necessary.
- Do not leave a possibly stuck child agent running forever just because the status indicator still looks active.

## Integration Rules

- When a child agent finishes, review the result quickly before launching follow-on work.
- Do not blindly trust returned code or analysis.
- Integrate outputs into one coherent answer or implementation plan.
- Subagents may provide recommendations, reviews, or comparative analysis, but the coordinator makes the final accept, reject, reroute, and done/not-done decisions.
- After integrating real progress, ensure the relevant checklist or planning-stack files are updated before treating the work as complete.
- Unless explicitly delegated with exclusive ownership, checklist, planning-stack, and milestone-tracking updates belong to the coordinator.
- Tell the user explicitly which checklist item, milestone, or gate-backed item was effectively completed or advanced, and when relevant which checkpoint commit captured that progress.
- Tell the user which next items are realistically tickable in parallel, rather than only naming one serial next step.
- Always close a worker agent once it is done and you no longer need that exact thread for immediate follow-on work.
- If two child agents conflict:
  - resolve the conflict yourself when possible,
  - otherwise reroute one focused follow-up agent with the conflict framed clearly.
- Close or stop using agents that are no longer useful.

## Anti-Patterns

Do not:

- use coordinator mode for a simple one-file task
- spawn child agents before understanding the critical path
- create overlapping write scopes without a good reason
- serialize obviously independent workstreams just because a smaller agent count feels safer
- show a list of supposed parallel branches that does not map to the actual next spawn wave
- use branch-shaped language for coordinator-owned or serialized work without labeling it clearly
- let accepted progress sit uncommitted across additional spawn waves
- treat checkpoint commits sloppily by skipping the required commit, staging the whole repo, or batching unrelated completions into one vague message
- use `wait_agent` as your main activity
- treat coordination as a single spawn-and-idle step
- try to run coordinator mode from a spawned subagent that lacks child-agent spawning tools
- assume an apparently long-running child is healthy without checking its thread
- delegate synthesis, prioritization, and conflict resolution away from the coordinator
- let a `default` child agent behave like a second coordinator
- let child agents own final acceptance, milestone completion, or project done/not-done judgment
- skip [SUBAGENTS.md](SUBAGENTS.md) and choose models from habit
