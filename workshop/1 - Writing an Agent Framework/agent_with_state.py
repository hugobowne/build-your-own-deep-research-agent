"""
See an example here of an agent which has a response fed to it and is able to respond to queries.

First turn: https://logfire-us.pydantic.dev/public-trace/2f236cc4-0a38-48e4-8c1e-d79bdccb45cf?spanId=eb60eb75ac37b798
Second turn: https://logfire-us.pydantic.dev/public-trace/51c7abaf-23a5-4734-b7e6-c9dc4ceb86aa?spanId=0e51b2189ec7ee5f
"""

from dataclasses import dataclass
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
    max_iterations: int = 1

    def increment_turn(self) -> None:
        self.iteration_count += 1

    def hit_iteration_limit(self) -> bool:
        return self.iteration_count >= self.max_iterations

    @property
    def should_disable_tools(self) -> bool:
        return self.hit_iteration_limit()


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
HookType: TypeAlias = Literal[
    "turn_start",
    "llm_response",
    "llm_tool_call",
    "llm_tool_result",
]
AnyHook: TypeAlias = (
    TurnStartHook | LLMResponseHook | LLMToolCallHook | LLMToolResultHook
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
    def execute(self, function_id: str) -> ToolResult:
        raise NotImplementedError


class ReadFile(AgentTool):
    """
    Read the contents of a file from disk.
    """

    path: str

    def execute(self, function_id):
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

    def execute(self, function_id: str) -> ToolResult:
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


SYSTEM_INSTRUCTION = """
You are Koroku, a coding agent built by Ivan.

Be polite, positive and helpful where you can. 
"""

FINAL_MESSAGE_REMINDER = """
You've run out of tools to call, do not call any more tools.

Simply provide the best final answer using the context you already have. 
"""

console = Console()


def format_tool_call(call: types.FunctionCall) -> str:
    if call.name == "readFile" and call.args and call.args.get("path"):
        return f"Reading {call.args['path']}..."
    if call.name == "bash" and call.args and call.args.get("command"):
        return f"$ {call.args['command']}"
    return f"{call.name}: {call.args or {}}"


def render_text_response(text: str) -> None:
    text = text.strip()
    if not text:
        return

    if "```" in text or text.startswith("#") or "\n# " in text:
        console.print(Markdown(text))
        return

    console.print(Text(f"* {text}", style="white"))


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

    def on(self, event: HookType, handler: AnyHook) -> "BaseAgent":
        self._hooks[event].append(handler)
        return self

    def emit(self, event: HookType, *args: Any) -> None:
        for handler in self._hooks[event]:
            handler(*args)

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
            return tool_input.execute(function_id=function_id)
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
                return output_parts

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
        if not function_calls:
            return

        run_state.increment_turn()


agent = Agent(tools=[ReadFile, Bash], run_state=AgentRunState())


def print_turn_start(_: list[ConversationItem], __: AgentRunState) -> None:
    console.print()
    console.print(Text("Assistant:", style="bold green"))


def print_llm_response(message: types.Content, _: AgentRunState) -> None:
    for part in message.parts or []:
        if part.text:
            render_text_response(part.text)


def print_llm_tool_call(call: types.FunctionCall, _: AgentRunState) -> None:
    console.print(Text(f"  {format_tool_call(call)}", style="yellow"))


def print_llm_tool_result(
    call: types.FunctionCall, result: ToolResult, _: AgentRunState
) -> None:
    if result.error:
        console.print(
            Text(f"  {call.name} failed: {result.response}", style="bold red")
        )


agent.on("turn_start", print_turn_start)
agent.on("llm_response", print_llm_response)
agent.on("llm_tool_call", print_llm_tool_call)
agent.on("llm_tool_result", print_llm_tool_result)

contents = []
while True:
    user_input = console.input("[bold cyan]You:[/bold cyan] ")
    contents.append(types.UserContent(parts=[types.Part.from_text(text=user_input)]))

    agent.run_until_idle(contents)
