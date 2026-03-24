"""
Final workshop agent with todo verification hooks, tracing spans, and Textual Shell UI.
"""

from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import re
from google.genai import types, Client
from typing import Any, Callable, Literal, TypeAlias, overload, Union
import os
import logfire
import subprocess
from rich.text import Text

# Import your shell
from shell import Shell

os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"

logfire.configure(console=False)
logfire.instrument_google_genai()


@dataclass(slots=True)
class AgentRunState:
    iteration_count: int = 0
    max_iterations: int = 30
    todos: list[str] = field(default_factory=list)
    current_state: Union[Literal["plan"], Literal["execute"]] = "plan"

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


@dataclass(slots=True)
class AgentContext:
    shell: Shell


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
    async def execute(
        self,
        function_id: str,
        run_state: AgentRunState,
        context: AgentContext,
    ) -> ToolResult:
        raise NotImplementedError


class ReadFile(AgentTool):
    """
    Read the contents of a file from disk.
    """

    path: str

    async def execute(
        self,
        function_id: str,
        run_state: AgentRunState,
        context: AgentContext,
    ) -> ToolResult:
        del run_state
        del context
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

    async def execute(
        self,
        function_id: str,
        run_state: AgentRunState,
        context: AgentContext,
    ) -> ToolResult:
        del run_state
        del context
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

    async def execute(
        self,
        function_id: str,
        run_state: AgentRunState,
        context: AgentContext,
    ) -> ToolResult:
        del context
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


class GenerateClarifyingQuestions(AgentTool):
    """
    This is a tool you should call to be able to generate a clarifying response/question to the user.

    You should call this at most 2 times.
    """

    clarifying_response: str = Field(
        description="This is a response that should contain 1 or more questions to the user to clarify the scope of the deep research query."
    )

    async def execute(
        self,
        function_id: str,
        run_state: AgentRunState,
        context: AgentContext,
    ) -> ToolResult:
        user_response = await context.shell.input()
        context.shell.print(Text(f"* You: {user_response}", style="bold cyan"))
        return self.tool_result(
            error=False,
            function_id=function_id,
            response={
                "clarifying_response": self.clarifying_response,
                "user_response": user_response,
            },
        )


class GeneratePlan(AgentTool):
    """
    Use this tool here when you've obtained enough information from the user about what he wants to find out about so that you can generate a summary of what the user wants and a list of at most 3 todos
    """

    request_summary: str = Field(
        description="This is a summary of the scope of the research that the user wants you to accomplish in around 3-4 sentences at most"
    )
    todos: list[str] = Field(
        min_length=2,
        max_length=4,
        description="These are the initial set of todos that you want make sure you check off when dealing with the user's deep research query. These should be action oriented (Eg. Breakdown the history of Singapore's colonial transition as it moved from a colony to a fully independent nation into distinct phases)",
    )

    async def execute(
        self,
        function_id: str,
        run_state: AgentRunState,
        context: AgentContext,
    ) -> ToolResult:

        return self.tool_result(
            error=False,
            function_id=function_id,
            response={
                "result": "Plan accepted by user and todos updated. Please start executing deep research query",
                "todos": f"Added the following todos {self.todos}.",
            },
        )


PLAN_INSTRUCTION = """
You are Koroku a deep research planner built by Ivan.

When the user provides you with a query, make sure you clarify the scope of the query. You should ask at most 2 times before generating your initial plan to tackle the user's query. This is incredibly important to do so.

Be polite, positive and helpful where you can. Use the GenerateClarifyingQuestions tool to clarify the scope of the user's request and then GeneratePlan when you're ready to start.

You must use GeneratePlan when you're ready to start working on the query once you've gathered more information.
"""

SYSTEM_INSTRUCTION = """
You are Koroku, a deep research agent built by Ivan.

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


def format_tool_call(call: types.FunctionCall) -> str:

    if call.name == "readFile" and call.args and call.args.get("path"):
        return f"Reading {call.args['path']}..."
    if call.name == "bash" and call.args and call.args.get("command"):
        return f"$ {call.args['command']}"
    if call.name == "modifyTodo" and call.args:
        todos = call.args.get("todos", [])
        return f"Todo {call.args.get('type')}: {', '.join(todos)}"
    return f"{call.name}: {call.args or {}}"


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
        context: AgentContext,
        model: str = "gemini-3.1-pro-preview",
    ) -> None:
        self.client = Client()
        self.model = model
        self.run_state = run_state
        self.context = context
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

    async def execute_tool(
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
            return await tool_input.execute(
                function_id=function_id,
                run_state=self.run_state,
                context=self.context,
            )
        except Exception as e:
            return ToolResult(
                error=True,
                name=tool_name,
                function_id=function_id,
                response={"error": str(e)},
            )

    async def run_until_idle(self, contents: list[types.Content]) -> list[types.Part]:
        with logfire.span(
            "run_until_idle",
            iteration_count=self.run_state.iteration_count,
            todo_count=len(self.run_state.todos),
            tools_disabled=self.run_state.should_disable_tools,
        ):
            output_parts: list[types.Part] = []
            self.emit("turn_start", contents, self.run_state)

            while True:
                # Use async client to avoid blocking the Shell app
                response = await self.client.aio.models.generate_content(
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
                    result = await self.execute_tool(
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
        context: AgentContext,
        model: str = "gemini-3.1-pro-preview",
        system_instruction: str = SYSTEM_INSTRUCTION,
        final_message_reminder: str = FINAL_MESSAGE_REMINDER,
    ) -> None:
        super().__init__(tools=tools, run_state=run_state, context=context, model=model)
        self.system_instruction = system_instruction
        self.final_message_reminder = final_message_reminder

    def get_tools(self, run_state: AgentRunState) -> list[types.Tool]:
        tools = []
        if run_state.should_disable_tools:
            return tools

        if run_state.current_state == "plan":
            tools = [GeneratePlan, GenerateClarifyingQuestions]
        else:
            tools = [ModifyTodo, Bash, ReadFile]

        return [tool.to_genai_schema() for tool in tools]

    def get_contents(
        self, contents: list[types.Content], run_state: AgentRunState
    ) -> list[types.Content]:
        system_prompt = (
            SYSTEM_INSTRUCTION
            if run_state.current_state == "execute"
            else PLAN_INSTRUCTION
        )
        next_contents = [
            types.UserContent(parts=[types.Part.from_text(text=system_prompt)]),
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

        # Now we check if it's just called the generatePLan call
        if run_state.current_state == "plan":
            has_generated_plan = any(
                [item for item in function_calls if item.name == "generatePlan"]
            )
            if has_generated_plan:
                run_state.current_state = "execute"


# --- Shell Integration Classes ---


class UIHooks:
    """Handles routing agent hook events to the Textual shell UI."""

    def __init__(self, shell: Shell):
        self.shell = shell

    def print_turn_start(self, _: list[ConversationItem], __: AgentRunState) -> None:
        self.shell.print(Text("Assistant:", style="bold green"))

    def print_llm_response(self, message: types.Content, _: AgentRunState) -> None:
        for part in message.parts or []:
            if part.text:
                self.shell.print_markdown(part.text)

    def print_llm_tool_call(
        self, call: types.FunctionCall, run_state: AgentRunState
    ) -> None:
        if call.name == "generateClarifyingQuestions":
            args = GenerateClarifyingQuestions(**call.args)
            self.shell.print(
                f"""
{args.clarifying_response}
""".strip()
            )
            return

        if call.name == "generatePlan":
            args = GeneratePlan(**call.args)

            # Print the summary first
            self.shell.print(f"\n{args.request_summary}\n")

            # Print a styled header for the todos
            self.shell.print(
                Text("  Generated the following todos:", style="bold cyan")
            )

            # Iterate through the list and style each todo
            if args.todos:
                for todo in args.todos:
                    self.shell.print(Text(f"  + [ ] {todo}", style="cyan"))
            else:
                self.shell.print(Text("    (No todos generated)", style="dim cyan"))

            return  # Added return to match the pattern from the previous block

        if call.name == "modifyTodo" and call.args:
            action = call.args.get("type", "add")
            changed = call.args.get("todos", [])
            self._render_todos(existing=run_state.todos, changed=changed, action=action)
            return
        self.shell.print(Text(f"  {format_tool_call(call)}", style="yellow"))

    def _render_todos(
        self, existing: list[str], changed: list[str], action: str
    ) -> None:
        self.shell.print(Text("  todos", style="bold cyan"))
        if action == "remove":
            changed_lower = {t.strip().lower() for t in changed}
            for todo in existing:
                if todo.lower() in changed_lower:
                    continue
                self.shell.print(Text(f"    [ ] {todo}", style="cyan"))
            for todo in changed:
                line = Text("    ", style="dim cyan")
                line.append(f"[ ] {todo}", style="strike dim")
                self.shell.print(line)
        else:
            for todo in existing:
                self.shell.print(Text(f"    [ ] {todo}", style="cyan"))
            for todo in changed:
                self.shell.print(Text(f"  + [ ] {todo}", style="cyan"))

    def print_llm_tool_result(
        self, call: types.FunctionCall, result: ToolResult, _: AgentRunState
    ) -> None:
        if call.name == "modifyTodo":
            return
        if result.error:
            self.shell.print(
                Text(f"  {call.name} failed: {result.response}", style="bold red")
            )


class WorkshopApp:
    def __init__(self):
        self.shell = Shell()
        self.agent = Agent(
            tools=[
                ReadFile,
                Bash,
                ModifyTodo,
                GenerateClarifyingQuestions,
                GeneratePlan,
            ],
            run_state=AgentRunState(),
            context=AgentContext(shell=self.shell),
        )
        self.hooks = UIHooks(self.shell)
        self.contents: list[types.Content] = []

        self.agent.on("turn_start", self.hooks.print_turn_start)
        self.agent.on("llm_response", self.hooks.print_llm_response)
        self.agent.on("llm_tool_call", self.hooks.print_llm_tool_call)
        self.agent.on("llm_tool_result", self.hooks.print_llm_tool_result)

        def ensure_all_todos_completed(
            _: types.Content, run_state: AgentRunState
        ) -> bool:
            return not run_state.todos

        self.agent.on("verify_turn_complete", ensure_all_todos_completed)

    async def on_submit(self, text: str) -> None:
        self.shell.print(Text(f"* You: {text}", style="bold cyan"))
        self.contents.append(types.UserContent(parts=[types.Part.from_text(text=text)]))

        self.shell.set_loading(True)
        try:
            with logfire.span("agent_session_turn"):
                await self.agent.run_until_idle(self.contents)
        finally:
            self.shell.set_loading(False)

    def run(self):
        self.shell.initialize(on_submit=self.on_submit)
        self.shell.run()


if __name__ == "__main__":
    WorkshopApp().run()
