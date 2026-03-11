# Build Your Own Deep Research Agent

Small workshop examples for building a deep research agent with the Gemini Interactions API.

## Setup

Install dependencies:

```bash
uv sync
```

Run examples from the repo root:

```bash
python3 "workshops/1 - Generating our Research Plan/response.py"
python3 "workshops/1 - Generating our Research Plan/tools.py"
python3 "workshops/1 - Generating our Research Plan/questions.py"
python3 "workshops/1 - Generating our Research Plan/questions_with_plan.py"
python3 "workshops/2 - migrating to Textual/shell.py"
```

## 1 - Generating our Research Plan

This section is about turning a vague research idea into a scoped research plan.

Rough shape:

- `response.py`: the smallest possible Interactions API example
- `tools.py`: a single tool-calling example with `clarifyScope`
- `questions.py`: a clarification loop that gathers missing information and outputs a research brief
- `questions_with_plan.py`: takes that research brief and turns it into a user-facing response plus a todo list

## 2 - migrating to Textual

This section is about moving the agent UI to Textual.

Rough shape:

- a persistent input box at the bottom of the screen
- a transcript area above it
- a reusable shell foundation for wiring in the queue and agent runtime

Archived:

- `workshops/archive/2 - Creating our minimal agent`: the plain terminal prototype with the in-memory steer / queue behavior

## Notes

- These scripts use the experimental Gemini Interactions API via `google-genai`.
- Rich is used for terminal formatting, including markdown rendering.
- The examples are intentionally simple and optimized for workshop/demo use rather than production robustness.
