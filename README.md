# Build Your Own Deep Research Agent

This repo is being rebuilt as a step-by-step workshop.

The goal is to mirror the teaching style from `koroku/workshop`: start with the smallest possible thing, then add one idea at a time.

## Steps

The progression lives in [steps](/Users/ivanleo/Documents/coding/build-your-own-deep-research-agent/steps).

### `01-minimal-call`

Accomplish:

- make the smallest possible Gemini call
- declare a tool schema manually
- inspect what a model-emitted function call actually looks like

This step is intentionally bare. The point is to see the raw API shape before adding any runtime abstractions.

### `02-single-tool`

Accomplish:

- implement one real tool handler
- execute the tool after the model calls it
- send the function response back to the model

By the end of this step, you have a full manual tool round-trip with no shared runtime yet.

### `03-tool-runtime`

Accomplish:

- extract a reusable `Tool` definition
- validate typed tool args at the runtime boundary
- stop hand-wiring every tool execution path

This is the first real runtime step: the code starts to become reusable without getting abstract for abstraction's sake.

### `04-run-state-and-context`

Accomplish:

- separate static configuration from mutable run state
- separate runtime dependencies from both
- introduce `RunConfig`, `RunState`, and `AgentContext`
- make concepts like iteration limits, todos, and clients live in the right place

### `05-hooks`

Accomplish:

- add explicit extension points for rendering and observation
- introduce `prepare_request()`, `message`, `llm_tool_call`, and `tool_result`
- keep request shaping visible through `prepare_request()`
- avoid turning the runtime loop into a giant conditional renderer

### `06-creating-an-agent`

Accomplish:

- wrap the runtime in an `Agent` class
- add a `run_until_idle()` loop that keeps going until tool use is finished
- move from one-off scripts to a reusable agent abstraction

### `07-subagents`

Accomplish:

- delegate focused work to child agents
- give each subagent its own bounded run config and state
- let the parent aggregate subagent results
- show live progress for concurrent delegated queries

### `08-prompt-builder`

Accomplish:

- move prompt construction into one place
- compose prompts from reusable parts instead of scattering strings around
- make later upgrades like templating possible without changing the runtime shape

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

without losing the incremental workshop feel.
