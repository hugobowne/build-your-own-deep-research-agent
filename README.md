# Build Your Own Deep Research Agent

Small workshop examples for building a deep research agent with the Gemini Interactions API.

## Setup

Install dependencies:

```bash
uv sync
```

Environment variables:

- `GOOGLE_API_KEY` — required for all workshops

## Workshop Progression

Scripts are progressive within each section — start from the top and work down.

### 1 - Writing an Agent Framework

This section builds up an agent framework from scratch, one concept at a time.

| # | File | What's introduced |
|---|------|-------------------|
| 1 | `genai_sdk.py` | Smallest possible Gemini SDK call with a manually-defined tool schema. No agent loop. |
| 2 | `genai_sdk_telemetry.py` | Adds **Logfire telemetry** to the same SDK call. |
| 3 | `single_turn_agent.py` | Introduces `AgentTool` / `ToolResult` Pydantic abstractions and a REPL loop. Tool execution is still inline (`if fc_name == "readFile"`). |
| 4 | `agent.py` | Extracts an `Agent` class with a `run_until_idle` agentic loop, a tool registry, and **Rich** console rendering. |
| 5 | `agent_with_hooks.py` | Adds an **event/hook system** (`on("turn_start", ...)`, `emit()`), decoupling rendering from the agent loop. |
| 6 | `agent_with_state.py` | Introduces `AgentRunState` (iteration counting, tool disabling at limit), a `BaseAgent` ABC with `get_tools` / `get_contents` / `update_run_state`, and the **Bash** tool. |
| 7 | `agent_with_final_hook.py` | Introduces **todos** — `todos: list[str]` on `AgentRunState`, the `ModifyTodo` tool, the `verify_turn_complete` hook, and the `ensure_all_todos_completed` guard. |
| 8 | `final_agent.py` | Wraps `run_until_idle` in a **Logfire span** for tracing and adds a `quit` command. |

```bash
uv run python3 "workshop/1 - Writing an Agent Framework/genai_sdk.py"
uv run python3 "workshop/1 - Writing an Agent Framework/single_turn_agent.py"    # interactive
uv run python3 "workshop/1 - Writing an Agent Framework/agent.py"                # interactive
uv run python3 "workshop/1 - Writing an Agent Framework/agent_with_hooks.py"     # interactive
uv run python3 "workshop/1 - Writing an Agent Framework/agent_with_state.py"     # interactive
uv run python3 "workshop/1 - Writing an Agent Framework/agent_with_final_hook.py" # interactive
uv run python3 "workshop/1 - Writing an Agent Framework/final_agent.py"          # interactive
```

### 2 - Creating a Plan

This section takes the final agent from section 1, makes it **async**, and migrates the UI to a **Textual Shell**.

| File | What's introduced |
|------|-------------------|
| `shell.py` | A reusable Textual `Shell` wrapper — persistent input box, scrolling transcript, loading spinner, and an `await shell.input()` API for interactive follow-ups. |
| `agent.py` | The section 1 agent adapted for Textual: `run_until_idle` becomes `async` (uses `client.aio.models.generate_content`), rendering is routed through a `UIHooks` class, and everything is wired together in `WorkshopApp`. |

```bash
uv run python3 "workshop/2 - Creating a Plan/shell.py"   # shell demo
uv run python3 "workshop/2 - Creating a Plan/agent.py"    # interactive agent with Textual UI
```

## Notes

- These scripts use the Gemini Interactions API via `google-genai`.
- Rich is used for terminal formatting, including markdown rendering.
- The examples are intentionally simple and optimized for workshop/demo use rather than production robustness.
