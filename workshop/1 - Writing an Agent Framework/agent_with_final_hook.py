"""
Extends the stateful agent with a final verification hook.
"""

from dataclasses import dataclass, field
from pydantic import BaseModel
from abc import ABC, abstractmethod
import re
from google.genai import types, Client
from typing import Any, Callable, Literal, TypeAlias, overload
import os
import logfire
import subprocess
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"

logfire.configure(console=False)
logfire.instrument_google_genai()


@dataclass(slots=True)
class AgentRunState:
    iteration_count: int = 0
    max_iterations: int = 10
    todos: list[str] = field(default_factory=list)

    def increment_turn(self) -> None:
        self.iteration_count += 1

    def hit_iteration_limit(self) -> bool:
        return self.iteration_count >= self.max_iterations

    @property
    def should_disable_tools(self) -> bool:
        return self.hit_iteration_limit()

    def add_todos(self, todos: list[str]) -> list[str]:
        added = []
        for todo in todos:
            todo = todo.strip()
            if todo and todo not in self.todos:
                self.todos.append(todo)
                added.append(todo)
        return added

    def remove_todos(self, todos: list[str]) -> tuple[list[str], list[str]]:
        removed = []
        not_found = []
        for todo in todos:
            todo_lower = todo.strip().lower()
            found = False
            for existing in list(self.todos):
                if existing.lower() == todo_lower:
                    self.todos.remove(existing)
                    removed.append(existing)
                    found = True
                    break
            if not found:
                not_found.append(todo.strip())
        return removed, not_found


class ToolResult(BaseModel):
    error: bool
    name: str
    function_id: str
    response: dict[str, Any]

    model_config = {"arbitrary_types_allowed": True}

    def to_genai_part(self) -> types.Part:
        return types.Part(
            function_response=types.FunctionResponse(
                name=self.name, response=self.response, id=self.function_id
            )
        )


ConversationItem: TypeAlias = types.Content
TurnStartHook = Callable[[list[ConversationItem], AgentRunState], None]
LLMResponseHook = Callable[[types.Content, AgentRunState], None]
LLMToolCallHook = Callable[[types.FunctionCall, AgentRunState], None]
LLMToolResultHook = Callable[[types.FunctionCall, ToolResult, AgentRunState], None]
VerifyTurnCompleteHook = Callable[[types.Content, AgentRunState], bool]
HookType: TypeAlias = Literal[
    "turn_start",
    "llm_response",
    "llm_tool_call",
    "llm_tool_result",
    "verify_turn_complete",
]
AnyHook: TypeAlias = (
    TurnStartHook
    | LLMResponseHook
    | LLMToolCallHook
    | LLMToolResultHook
    | VerifyTurnCompleteHook
)


class AgentTool(BaseModel, ABC):
    @classmethod
    def tool_name(cls) -> str:
        name = cls.__name__
        parts = re.split(r"[_\s]+", name)
        if len(parts) > 1:
            return parts[0].lower() + "".join(part.capitalize() for part in parts[1:])
        return name[:1].lower() + name[1:]

    def tool_result(
        self, *, error: bool, function_id: str, response: dict[str, Any]
    ) -> ToolResult:
        return ToolResult(
            error=error,
            name=self.__class__.tool_name(),
            function_id=function_id,
            response=response,
        )

    @classmethod
    def to_genai_schema(cls) -> types.Tool:
        json_schema = cls.model_json_schema()
        tool_name = cls.tool_name()
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=tool_name,
                    description=json_schema.get("description", f"{tool_name} tool"),
                    parameters=types.Schema(
                        type="OBJECT",
                        properties=json_schema["properties"],
                        required=json_schema.get("required", []),
                    ),
                )
            ]
        )

    @abstractmethod
    def execute(self, function_id: str, run_state: AgentRunState) -> ToolResult:
        raise NotImplementedError


class ReadFile(AgentTool):
    """
    Read the contents of a file from disk.
    """

    path: str

    def execute(self, function_id: str, run_state: AgentRunState) -> ToolResult:
        del run_state
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return self.tool_result(
                    error=False,
                    function_id=function_id,
                    response={"path": self.path, "content": f.read()},
                )
        except Exception as e:
            return self.tool_result(
                error=True,
                function_id=function_id,
                response={"path": self.path, "error": str(e)},
            )


class Bash(AgentTool):
    """
    Run a bash command and return its output.
    """

    command: str

    def execute(self, function_id: str, run_state: AgentRunState) -> ToolResult:
        del run_state
        try:
            result = subprocess.run(
                ["bash", "-lc", self.command],
                capture_output=True,
                text=True,
            )
            return self.tool_result(
                error=result.returncode != 0,
                function_id=function_id,
                response={
                    "command": self.command,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                },
            )
        except Exception as e:
            return self.tool_result(
                error=True,
                function_id=function_id,
                response={"command": self.command, "error": str(e)},
            )


class ModifyTodo(AgentTool):
    """
    Add or remove todos from the agent run state. Supports multiple todos at once.
    """

    type: Literal["add", "remove"]
    todos: list[str]

    def execute(self, function_id: str, run_state: AgentRunState) -> ToolResult:
        if self.type == "add":
            added = run_state.add_todos(self.todos)
            return self.tool_result(
                error=False,
                function_id=function_id,
                response={"action": "add", "added": added, "todos": run_state.todos},
            )

        removed, not_found = run_state.remove_todos(self.todos)
        return self.tool_result(
            error=len(not_found) > 0,
            function_id=function_id,
            response={
                "action": "remove",
                "removed": removed,
                "not_found": not_found,
                "todos": run_state.todos,
            },
        )


SYSTEM_INSTRUCTION = """
You are Koroku, a coding agent built by Ivan.

Be polite, positive and helpful where you can.
Use modifyTodo to keep an explicit todo list of the work you still need to finish.
Remove todos when they are completed.

You must create a todo before embarking on any task
"""

FINAL_MESSAGE_REMINDER = """
You have no tools left.
Before finishing, make sure all todos are completed and removed.
If there are remaining todos, continue the turn until they are all cleared.
"""

console = Console()


def format_tool_call(call: types.FunctionCall) -> str:
    if call.name == "readFile" and call.args and call.args.get("path"):
        return f"Reading {call.args['path']}..."
    if call.name == "bash" and call.args and call.args.get("command"):
        return f"$ {call.args['command']}"
    if call.name == "modifyTodo" and call.args:
        todos = call.args.get("todos", [])
        return f"Todo {call.args.get('type')}: {', '.join(todos)}"
    return f"{call.name}: {call.args or {}}"


def render_text_response(text: str) -> None:
    text = text.strip()
    if not text:
        return

    if "```" in text or text.startswith("#") or "\n# " in text:
        console.print(Markdown(text))
        return

    console.print(Text(f"* {text}", style="white"))


def render_todos(existing: list[str], changed: list[str], action: str) -> None:
    console.print(Text("  todos", style="bold cyan"))
    if action == "remove":
        changed_lower = {t.strip().lower() for t in changed}
        for todo in existing:
            if todo.lower() in changed_lower:
                continue
            console.print(Text(f"    [ ] {todo}", style="cyan"))
        for todo in changed:
            line = Text("    ", style="dim cyan")
            line.append(f"[ ] {todo}", style="strike dim")
            console.print(line)
    else:
        for todo in existing:
            console.print(Text(f"    [ ] {todo}", style="cyan"))
        for todo in changed:
            console.print(Text(f"  + [ ] {todo}", style="cyan"))


def format_todos(todos: list[str]) -> str:
    if not todos:
        return "Current todos:\n- none"
    return "Current todos:\n" + "\n".join(f"- {todo}" for todo in todos)


class BaseAgent(ABC):
    def __init__(
        self,
        *,
        tools: list[type[AgentTool]],
        run_state: AgentRunState,
        model: str = "gemini-3.1-pro-preview",
    ) -> None:
        self.client = Client()
        self.model = model
        self.run_state = run_state
        self.tool_registry = {tool.tool_name(): tool for tool in tools}
        self._hooks: dict[HookType, list[AnyHook]] = {
            "turn_start": [],
            "llm_response": [],
            "llm_tool_call": [],
            "llm_tool_result": [],
            "verify_turn_complete": [],
        }

    @overload
    def on(
        self, event: Literal["turn_start"], handler: TurnStartHook
    ) -> "BaseAgent": ...

    @overload
    def on(
        self, event: Literal["llm_response"], handler: LLMResponseHook
    ) -> "BaseAgent": ...

    @overload
    def on(
        self, event: Literal["llm_tool_call"], handler: LLMToolCallHook
    ) -> "BaseAgent": ...

    @overload
    def on(
        self, event: Literal["llm_tool_result"], handler: LLMToolResultHook
    ) -> "BaseAgent": ...

    @overload
    def on(
        self,
        event: Literal["verify_turn_complete"],
        handler: VerifyTurnCompleteHook,
    ) -> "BaseAgent": ...

    def on(self, event: HookType, handler: AnyHook) -> "BaseAgent":
        self._hooks[event].append(handler)
        return self

    def emit(self, event: HookType, *args: Any) -> None:
        if event == "verify_turn_complete":
            raise ValueError(
                "Use verify_turn_complete() for verify_turn_complete hooks"
            )
        for handler in self._hooks[event]:
            handler(*args)

    def verify_turn_complete(self, message: types.Content) -> bool:
        handlers = self._hooks["verify_turn_complete"]
        if not handlers:
            return True
        return all(handler(message, self.run_state) for handler in handlers)

    @abstractmethod
    def get_tools(self, run_state: AgentRunState) -> list[types.Tool]:
        raise NotImplementedError

    @abstractmethod
    def get_contents(
        self, contents: list[types.Content], run_state: AgentRunState
    ) -> list[types.Content]:
        raise NotImplementedError

    @abstractmethod
    def update_run_state(
        self, run_state: AgentRunState, message: types.Content
    ) -> None:
        raise NotImplementedError

    def get_config(self, run_state: AgentRunState) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            tools=self.get_tools(run_state),
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
        )

    def execute_tool(
        self, tool_name: str, args: dict[str, Any], function_id: str
    ) -> ToolResult:
        tool_cls = self.tool_registry.get(tool_name)
        if tool_cls is None:
            return ToolResult(
                error=True,
                name=tool_name,
                function_id=function_id,
                response={"error": f"Unknown tool: {tool_name}"},
            )

        try:
            tool_input = tool_cls.model_validate(args or {})
            return tool_input.execute(function_id=function_id, run_state=self.run_state)
        except Exception as e:
            return ToolResult(
                error=True,
                name=tool_name,
                function_id=function_id,
                response={"error": str(e)},
            )

    def run_until_idle(self, contents: list[types.Content]) -> list[types.Part]:
        output_parts: list[types.Part] = []
        self.emit("turn_start", contents, self.run_state)

        while True:
            response = self.client.models.generate_content(
                model=self.model,
                contents=self.get_contents(contents, self.run_state),
                config=self.get_config(self.run_state),
            )

            message = response.candidates[0].content
            parts = message.parts or []
            contents.append(message)
            output_parts.extend(parts)
            self.emit("llm_response", message, self.run_state)
            self.update_run_state(self.run_state, message)

            function_calls = [
                part.function_call for part in parts if part.function_call
            ]
            if not function_calls:
                if self.verify_turn_complete(message):
                    return output_parts

                reminder = (
                    "You are not done yet. Clear all remaining todos before ending.\n"
                    f"{format_todos(self.run_state.todos)}"
                )
                contents.append(
                    types.UserContent(parts=[types.Part.from_text(text=reminder)])
                )
                continue

            tool_parts: list[types.Part] = []
            for call in function_calls:
                self.emit("llm_tool_call", call, self.run_state)
                result = self.execute_tool(
                    tool_name=call.name,
                    args=call.args or {},
                    function_id=call.id,
                )
                self.emit("llm_tool_result", call, result, self.run_state)
                tool_parts.append(result.to_genai_part())

            contents.append(types.UserContent(parts=tool_parts))


class Agent(BaseAgent):
    def __init__(
        self,
        *,
        tools: list[type[AgentTool]],
        run_state: AgentRunState,
        model: str = "gemini-3.1-pro-preview",
        system_instruction: str = SYSTEM_INSTRUCTION,
        final_message_reminder: str = FINAL_MESSAGE_REMINDER,
    ) -> None:
        super().__init__(tools=tools, run_state=run_state, model=model)
        self.system_instruction = system_instruction
        self.final_message_reminder = final_message_reminder

    def get_tools(self, run_state: AgentRunState) -> list[types.Tool]:
        if run_state.should_disable_tools:
            return []

        return [tool.to_genai_schema() for tool in self.tool_registry.values()]

    def get_contents(
        self, contents: list[types.Content], run_state: AgentRunState
    ) -> list[types.Content]:
        next_contents = [
            types.UserContent(
                parts=[types.Part.from_text(text=self.system_instruction)]
            ),
            types.UserContent(
                parts=[types.Part.from_text(text=format_todos(run_state.todos))]
            ),
            *contents,
        ]

        if run_state.should_disable_tools:
            next_contents.append(
                types.UserContent(
                    parts=[types.Part.from_text(text=self.final_message_reminder)]
                )
            )

        return next_contents

    def update_run_state(
        self, run_state: AgentRunState, message: types.Content
    ) -> None:
        function_calls = [
            part.function_call for part in message.parts or [] if part.function_call
        ]
        if function_calls:
            run_state.increment_turn()


def print_turn_start(_: list[ConversationItem], __: AgentRunState) -> None:
    console.print()
    console.print(Text("Assistant:", style="bold green"))


def print_llm_response(message: types.Content, _: AgentRunState) -> None:
    for part in message.parts or []:
        if part.text:
            render_text_response(part.text)


def print_llm_tool_call(call: types.FunctionCall, run_state: AgentRunState) -> None:
    if call.name == "modifyTodo" and call.args:
        action = call.args.get("type", "add")
        changed = call.args.get("todos", [])
        render_todos(existing=run_state.todos, changed=changed, action=action)
        return
    console.print(Text(f"  {format_tool_call(call)}", style="yellow"))


def print_llm_tool_result(
    call: types.FunctionCall, result: ToolResult, run_state: AgentRunState
) -> None:
    if call.name == "modifyTodo":
        return
    if result.error:
        console.print(
            Text(f"  {call.name} failed: {result.response}", style="bold red")
        )


def ensure_all_todos_completed(_: types.Content, run_state: AgentRunState) -> bool:
    return not run_state.todos


agent = Agent(tools=[ReadFile, Bash, ModifyTodo], run_state=AgentRunState())
agent.on("turn_start", print_turn_start)
agent.on("llm_response", print_llm_response)
agent.on("llm_tool_call", print_llm_tool_call)
agent.on("llm_tool_result", print_llm_tool_result)
agent.on("verify_turn_complete", ensure_all_todos_completed)

contents = []
while True:
    user_input = console.input("[bold cyan]You:[/bold cyan] ")
    contents.append(types.UserContent(parts=[types.Part.from_text(text=user_input)]))
    agent.run_until_idle(contents)
