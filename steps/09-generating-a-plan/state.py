from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, TypeAlias

from exa_py import Exa


SearchAgentRunner = Callable[[list[str]], Awaitable[list[dict[str, str]]]]
Mode: TypeAlias = Literal["plan", "execute"]


@dataclass(slots=True)
class RunConfig:
    model: str = "gemini-3.1-pro-preview"
    thinking_level: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    max_iterations: int = 30


@dataclass(slots=True)
class RunState:
    iteration_count: int = 0
    mode: Mode = "plan"
    todos: list[str] = field(default_factory=list)

    def add_todos(self, todos: list[str]) -> list[str]:
        added: list[str] = []
        for todo in todos:
            todo = todo.strip()
            if todo and todo not in self.todos:
                self.todos.append(todo)
                added.append(todo)
        return added

    def remove_todos(self, todos: list[str]) -> tuple[list[str], list[str]]:
        removed: list[str] = []
        not_found: list[str] = []
        for todo in todos:
            todo = todo.strip()
            todo_lower = todo.lower()
            existing = next(
                (item for item in self.todos if item.lower() == todo_lower),
                None,
            )
            if existing is None:
                not_found.append(todo)
                continue
            self.todos.remove(existing)
            removed.append(existing)
        return removed, not_found

    def is_incomplete(self) -> str | None:
        if self.mode == "plan":
            return None

        if self.todos:
            return f"""
You still have pending todos.

<todos>
{chr(10).join(self.todos)}
</todos>

Check off all todos before you end.
""".strip()

        return None


@dataclass(slots=True)
class AgentContext:
    exa: Exa | None = None
    search_agent_runner: SearchAgentRunner | None = None
    live: Any | None = None
    subagent_statuses: dict[str, str] = field(default_factory=dict)
