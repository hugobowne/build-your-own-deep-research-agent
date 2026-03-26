# Steps

This folder is the new step-by-step build-up for the agent runtime.

The goal is to mirror the teaching style from `koroku/workshop`: start with the smallest possible thing, then add one idea at a time.

## Step Order

### `01-minimal-call`

The smallest possible Gemini call with a manually declared tool schema.

Teach:

- how tool declarations are exposed to the model
- what a function call looks like in the model response

Do not introduce runtime abstractions yet.

### `02-single-tool`

Add one real tool handler and manually round-trip the function call back to the model.

Teach:

- how to execute one tool
- how to send a function response back

Keep this intentionally direct, but avoid defensive fallback patterns like `fc.args or {}`. Validate required arguments explicitly and fail if they are missing.

### `03-tool-runtime`

Extract the repeated tool execution path into a tiny runtime.

Teach:

- a shared `Tool` definition
- typed arg validation at the runtime boundary
- plain handler functions

This is where we stop hand-wiring each tool call.

### `04-run-state-and-context`

Introduce:

- `RunConfig`
- `RunState`
- `AgentContext`

Teach:

- the difference between config, mutable state, and dependencies
- why things like `max_iterations` belong in config
- why clients like Exa/db belong in context

### `05-hooks`

Introduce the hook lifecycle:

- `before_model`
- `after_model`
- `after_tools`

Teach:

- how hooks let us extend behavior incrementally
- how to keep the core loop readable

Keep hooks small and explicit. Do not let them turn into a hidden framework.

### `06-rendering-with-hooks`

Add rendering through hooks with a generic `shell.write(...)` and updateable blocks.

Teach:

- how rendering can stay flexible without introducing a reducer
- when to append output vs update a persistent block
- how hooks are a clean place to drive rendering

This is where status blocks, tool render output, and live subagent panels start to appear.

### `07-subagents`

Add subagents with their own:

- `RunConfig`
- `RunState`
- iteration budget

Teach:

- how a parent agent can dispatch a bounded child agent
- how shell live blocks can track subagent progress

### `08-beautifying-the-outputs`

Improve the shell output and tool rendering.

Teach:

- how to render assistant output more cleanly
- how to surface tool errors without dumping raw objects
- how richer output can stay isolated from the core agent runtime

### `09-generating-a-plan`

Add an explicit planning phase before execution.

Teach:

- how run state can track a current `mode`
- how tool availability can change based on that mode
- how a `generate_plan` tool can transition the run from planning into execution

### `10-adding-open-telemetry`

Add Logfire-backed OpenTelemetry spans around the agent loop.

Teach:

- how to keep the default Google GenAI instrumentation
- how to add explicit parent spans for a user turn, model iteration, tool call, and subagent
- how to attach run config and mutable state like iteration count and todos to each span

## Rules For These Steps

- Each step should introduce one main idea.
- Avoid skipping directly to the final architecture.
- Prefer plain functions and small dataclasses over deep class hierarchies.
- Validate tool arguments early at the runtime boundary.
- Do not use defensive patterns like `call.args or {}` when required inputs are missing.
- Keep the hook story because it matches the workshop format well.

## Intended Outcome

By the later steps, we should have:

- a clean hook-based runtime
- typed tool argument validation
- explicit config/state/context boundaries
- flexible rendering through hooks
- support for bounded subagents
- OpenTelemetry traces that show how a turn progressed

without losing the incremental workshop feel.
