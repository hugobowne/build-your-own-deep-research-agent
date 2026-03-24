from dataclasses import dataclass, field


@dataclass(slots=True)
class RunConfig:
    model: str = "gemini-3-flash-preview"
    max_iterations: int = 30


@dataclass(slots=True)
class RunState:
    iteration_count: int = 0
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


@dataclass(slots=True)
class AgentContext:
    pass

