from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, TypeVar

from google.genai import types
from pydantic import BaseModel

from state import AgentContext, RunState


ArgsT = TypeVar("ArgsT", bound=BaseModel)
ToolHandler = Callable[[ArgsT, RunState, AgentContext], Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler

    def to_genai_tool(self) -> types.Tool:
        schema = self.args_model.model_json_schema()
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=self.name,
                    description=self.description,
                    parameters=types.Schema(
                        type="OBJECT",
                        properties=schema["properties"],
                        required=schema.get("required", []),
                    ),
                )
            ]
        )


class ReadFileArgs(BaseModel):
    path: str


async def read_file(
    args: ReadFileArgs,
    state: RunState,
    context: AgentContext,
) -> dict[str, Any]:
    path = Path(args.path)
    if not path.exists() or not path.is_file():
        return {"error": f"File does not exist: {args.path}"}

    return {
        "result": f"""
Read file at path {args.path}

<content>
{path.read_text(encoding="utf-8")}
</content>""".strip()
    }


class ModifyTodoArgs(BaseModel):
    action: Literal["add", "remove"]
    todos: list[str]


async def modify_todo(
    args: ModifyTodoArgs,
    state: RunState,
    context: AgentContext,
) -> dict[str, Any]:
    if args.action == "add":
        state.add_todos(args.todos)
        return {
            "result": f"""
Todos updated to

<todos>
{chr(10).join(state.todos)}
</todos>""".strip()
        }

    requested = [todo.strip() for todo in args.todos]
    missing = []
    existing_lower = {todo.lower() for todo in state.todos}
    for todo in requested:
        if todo.lower() not in existing_lower:
            missing.append(todo)

    if missing:
        return {"error": f"Todos not found: {', '.join(missing)}"}

    state.remove_todos(args.todos)
    return {
        "result": f"""
Todos updated to

<todos>
{chr(10).join(state.todos)}
</todos>""".strip()
    }


READ_FILE_TOOL = Tool(
    name="read_file",
    description="Read a UTF-8 text file and return its contents.",
    args_model=ReadFileArgs,
    handler=read_file,
)

MODIFY_TODO_TOOL = Tool(
    name="modify_todo",
    description="Add or remove todos from the current run state.",
    args_model=ModifyTodoArgs,
    handler=modify_todo,
)

