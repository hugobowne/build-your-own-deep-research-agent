from copy import deepcopy
from typing import Any, Awaitable, Callable, Literal, TypeAlias

from google.genai import Client, types
from pydantic import ValidationError
from rich import print
from rich.markdown import Markdown

from state import AgentContext, RunConfig, RunState
from tools import (
    BashMetadata,
    EditFileMetadata,
    GeneratePlanMetadata,
    ReadFileMetadata,
    Tool,
    ToolExecutionResult,
    WriteFileMetadata,
)


MessageHook: TypeAlias = Callable[
    [types.Content, RunConfig, RunState, AgentContext],
    Awaitable[None] | None,
]

LLMToolCallHook: TypeAlias = Callable[
    [types.FunctionCall, RunConfig, RunState, AgentContext],
    Awaitable[None] | None,
]

ToolResultHook: TypeAlias = Callable[
    [types.FunctionCall, ToolExecutionResult, RunConfig, RunState, AgentContext],
    Awaitable[None] | None,
]

HookType: TypeAlias = Literal["message", "llm_tool_call", "tool_result"]
AnyHook: TypeAlias = MessageHook | LLMToolCallHook | ToolResultHook

MAX_ITERATIONS_REACHED_MESSAGE = (
    "You've reached the maximum number of iterations, generate a quick summary "
    "of everything you've accomplished."
)


class Agent:
    def __init__(
        self,
        *,
        client: Client,
        config: RunConfig,
        state: RunState,
        context: AgentContext,
        plan_tools: list[Tool],
        execute_tools: list[Tool],
        plan_system_instruction: str,
        execute_system_instruction: str,
    ) -> None:
        self.client = client
        self.config = config
        self.state = state
        self.context = context
        self.plan_system_instruction = plan_system_instruction
        self.execute_system_instruction = execute_system_instruction
        self.plan_tools = {tool.name: tool for tool in plan_tools}
        self.execute_tools = {tool.name: tool for tool in execute_tools}
        self.tools = {**self.plan_tools, **self.execute_tools}
        self.hooks: dict[HookType, list[AnyHook]] = {
            "message": [],
            "llm_tool_call": [],
            "tool_result": [],
        }

    def on(self, hook_type: HookType, hook: AnyHook) -> None:
        self.hooks[hook_type].append(hook)

    async def emit(self, hook_type: HookType, **kwargs: Any) -> None:
        for hook in self.hooks[hook_type]:
            result = hook(**kwargs)
            if result is not None:
                await result

    def prepare_request(
        self, contents: list[types.Content]
    ) -> tuple[types.GenerateContentConfig, list[types.Content]]:
        active_tools = (
            self.plan_tools if self.state.mode == "plan" else self.execute_tools
        )
        tools = []
        if self.state.iteration_count < self.config.max_iterations:
            tools = [tool.to_genai_tool() for tool in active_tools.values()]
        else:
            contents.append(
                types.UserContent(
                    parts=[types.Part.from_text(text=MAX_ITERATIONS_REACHED_MESSAGE)]
                )
            )
        config = types.GenerateContentConfig(
            tools=tools,
            system_instruction=(
                self.plan_system_instruction
                if self.state.mode == "plan"
                else self.execute_system_instruction
            ),
            thinking_config=types.ThinkingConfig(
                thinking_level=self.config.thinking_level
            ),
        )
        return config, list(contents)

    async def execute_tool_call(self, call: types.FunctionCall) -> dict[str, Any]:
        tool = self.tools.get(call.name)
        if tool is None:
            execution_result = ToolExecutionResult(
                model_response={"error": f"Unknown tool: {call.name}"}
            )
            return {
                "name": call.name,
                "execution_result": execution_result,
                "response": execution_result.model_response,
            }
        if call.args is None:
            execution_result = ToolExecutionResult(
                model_response={
                    "error": f"Tool call '{call.name}' did not include arguments."
                }
            )
            return {
                "name": call.name,
                "execution_result": execution_result,
                "response": execution_result.model_response,
            }
        try:
            args = tool.args_model.model_validate(call.args)
        except ValidationError as error:
            execution_result = ToolExecutionResult(
                model_response={
                    "error": (f"Invalid arguments for tool '{call.name}':\n{error}")
                }
            )
            return {
                "name": call.name,
                "execution_result": execution_result,
                "response": execution_result.model_response,
            }
        execution_result = await tool.handler(args, self.state, self.context)
        return {
            "name": call.name,
            "execution_result": execution_result,
            "response": execution_result.model_response,
        }

    def render_todos(self, previous_state: RunState, current_state: RunState) -> None:
        if previous_state.todos == current_state.todos:
            return

        removed = [
            todo for todo in previous_state.todos if todo not in current_state.todos
        ]

        print("\nTodos:")
        for todo in previous_state.todos:
            if todo in removed:
                print(f"[ ] [strike]{todo}[/strike]")
            else:
                print(f"[ ] {todo}")

        for todo in current_state.todos:
            if todo not in previous_state.todos:
                print(f"+[ ] {todo}")

    async def run_until_idle(self, contents: list[types.Content]) -> types.Content:
        self.state.iteration_count = 0
        while True:
            self.state.iteration_count += 1
            request_config, request_contents = self.prepare_request(contents)

            completion = await self.client.aio.models.generate_content(
                model=self.config.model,
                contents=request_contents,
                config=request_config,
            )

            message = completion.candidates[0].content
            contents.append(message)
            await self.emit(
                "message",
                message=message,
                config=self.config,
                state=self.state,
                context=self.context,
            )

            function_calls = [
                part.function_call for part in message.parts if part.function_call
            ]

            if not function_calls:
                reason = self.state.is_incomplete()
                if reason is None:
                    return message

                contents.append(
                    types.UserContent(parts=[types.Part.from_text(text=reason)])
                )
                continue

            previous_state = deepcopy(self.state)
            tool_parts: list[types.Part] = []
            for call in function_calls:
                await self.emit(
                    "llm_tool_call",
                    call=call,
                    config=self.config,
                    state=self.state,
                    context=self.context,
                )
                result = await self.execute_tool_call(call)
                await self.emit(
                    "tool_result",
                    call=call,
                    result=result["execution_result"],
                    config=self.config,
                    state=self.state,
                    context=self.context,
                )
                tool_parts.append(
                    types.Part.from_function_response(
                        name=result["name"],
                        response=result["response"],
                    )
                )

            contents.append(types.UserContent(parts=tool_parts))
            self.render_todos(previous_state, self.state)


async def render_message(
    message: types.Content,
    config: RunConfig,
    state: RunState,
    context: AgentContext,
) -> None:
    for part in message.parts:
        if part.text:
            print()
            print(Markdown(part.text))


async def render_tool_call(
    call: types.FunctionCall,
    config: RunConfig,
    state: RunState,
    context: AgentContext,
) -> None:
    return


async def render_tool_result(
    call: types.FunctionCall,
    result: ToolExecutionResult,
    config: RunConfig,
    state: RunState,
    context: AgentContext,
) -> None:
    error = result.model_response.get("error")
    if error:
        print()
        print(f"[red]Tool error ({call.name}):[/red]")
        print(Markdown(f"```text\n{error}\n```"))
        return

    metadata = result.metadata
    if isinstance(metadata, ReadFileMetadata):
        print()
        print(f"Read file: {metadata.path}")
        print(Markdown(f"```text\n{metadata.contents}\n```"))
    if isinstance(metadata, WriteFileMetadata):
        print()
        print(f"Wrote file: {metadata.path}")
    if isinstance(metadata, EditFileMetadata):
        print()
        print(f"Edited file: {metadata.path}")
    if isinstance(metadata, BashMetadata):
        print()
        print(f"Ran command: {metadata.command}")
        print(f"Exit code: {metadata.returncode}")
        print(Markdown(f"```text\n{metadata.stdout}\n```"))
        if metadata.stderr:
            print(Markdown(f"```text\n{metadata.stderr}\n```"))
    if isinstance(metadata, GeneratePlanMetadata):
        print()
        print("[green]Switched to execute mode.[/green]")
    return
