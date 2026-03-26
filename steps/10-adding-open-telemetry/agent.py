import json
from copy import deepcopy
from typing import Any, Awaitable, Callable, Literal, TypeAlias

from google.genai import Client, types
import logfire
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


def summarize_text(text: str, max_length: int = 240) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 3].rstrip()}..."


def clip_text(text: str, max_length: int = 400) -> str:
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 3].rstrip()}..."


def summarize_message(message: types.Content) -> dict[str, str | int | list[str]]:
    text_parts = [part.text for part in message.parts if part.text]
    function_calls = [
        part.function_call.name for part in message.parts if part.function_call
    ]
    summary: dict[str, str | int | list[str]] = {
        "agent.response.text_part_count": len(text_parts),
        "agent.response.function_call_count": len(function_calls),
        "agent.response.function_call_names": function_calls,
    }
    if text_parts:
        summary["agent.response.text_preview"] = summarize_text("\n".join(text_parts))
    return summary


def serialize_request_payload(
    request_contents: list[types.Content],
    request_config: types.GenerateContentConfig,
) -> dict[str, str]:
    return {
        "agent.request.contents_json": json.dumps(
            [
                content.model_dump(mode="json", exclude_none=True)
                for content in request_contents
            ],
            sort_keys=True,
        ),
        "agent.request.config_json": json.dumps(
            request_config.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
        ),
        "agent.request.tools_json": json.dumps(
            [
                tool.model_dump(mode="json", exclude_none=True)
                for tool in request_config.tools or []
            ],
            sort_keys=True,
        ),
    }


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
        span_title: str = "agent.run",
        span_attributes: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.state = state
        self.context = context
        self.plan_system_instruction = plan_system_instruction
        self.execute_system_instruction = execute_system_instruction
        self.span_title = span_title
        self.span_attributes = dict(span_attributes or {})
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
    ) -> tuple[types.GenerateContentConfig, list[types.Content], list[str]]:
        active_tools = (
            self.plan_tools if self.state.mode == "plan" else self.execute_tools
        )
        tools = []
        active_tool_names: list[str] = []
        if self.state.iteration_count < self.config.max_iterations:
            tools = [tool.to_genai_tool() for tool in active_tools.values()]
            active_tool_names = list(active_tools.keys())
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
        return config, list(contents), active_tool_names

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
        with logfire.span(
            self.span_title,
            **self.config.telemetry_attributes(),
            **self.state.telemetry_attributes(),
            **self.span_attributes,
        ) as turn_span:
            while True:
                self.state.iteration_count += 1
                request_config, request_contents, active_tool_names = (
                    self.prepare_request(contents)
                )

                request_payload = serialize_request_payload(
                    request_contents, request_config
                )
                request_summary_attributes = {
                    "agent.request.model": self.config.model,
                    "agent.request.content_count": len(request_contents),
                    "agent.request.active_tool_count": len(active_tool_names),
                    "agent.request.active_tool_names": active_tool_names,
                }
                turn_span.set_attributes(
                    {
                        **self.config.telemetry_attributes(),
                        **self.state.telemetry_attributes(),
                        **request_summary_attributes,
                    }
                )

                with logfire.span(
                    "agent.run",
                    **self.config.telemetry_attributes(),
                    **self.state.telemetry_attributes(),
                    **self.span_attributes,
                    **request_summary_attributes,
                    **request_payload,
                ) as run_span:
                    run_span.set_attributes(
                        {
                            **self.config.telemetry_attributes(),
                            **self.state.telemetry_attributes(),
                            **request_payload,
                        }
                    )
                    completion = await self.client.aio.models.generate_content(
                        model=self.config.model,
                        contents=request_contents,
                        config=request_config,
                    )

                    message = completion.candidates[0].content
                    contents.append(message)
                    run_span.set_attributes(
                        {
                            **self.state.telemetry_attributes(),
                            **summarize_message(message),
                        }
                    )
                    turn_span.set_attributes(
                        {
                            **self.state.telemetry_attributes(),
                            **summarize_message(message),
                        }
                    )

                    await self.emit(
                        "message",
                        message=message,
                        config=self.config,
                        state=self.state,
                        context=self.context,
                    )

                    function_calls = [
                        part.function_call
                        for part in message.parts
                        if part.function_call
                    ]

                    if not function_calls:
                        with logfire.span(
                            "agent.response",
                            **self.config.telemetry_attributes(),
                            **self.state.telemetry_attributes(),
                            **request_payload,
                            **summarize_message(message),
                        ):
                            pass
                        reason = self.state.is_incomplete()
                        if reason is None:
                            run_span.set_attributes(self.state.telemetry_attributes())
                            turn_span.set_attributes(self.state.telemetry_attributes())
                            return message

                        contents.append(
                            types.UserContent(parts=[types.Part.from_text(text=reason)])
                        )
                        continue

                    previous_state = deepcopy(self.state)
                    tool_parts: list[types.Part] = []
                    for call in function_calls:
                        with logfire.span(
                            "agent.tool_call {agent_tool_name}",
                            **self.config.telemetry_attributes(),
                            **self.state.telemetry_attributes(),
                            **request_payload,
                            agent_tool_name=call.name,
                            agent_tool_args_json=json.dumps(
                                call.args or {}, sort_keys=True
                            ),
                        ):
                            await self.emit(
                                "llm_tool_call",
                                call=call,
                                config=self.config,
                                state=self.state,
                                context=self.context,
                            )
                            with logfire.span(
                                "agent.tool_executed {agent_tool_name}",
                                **self.config.telemetry_attributes(),
                                **self.state.telemetry_attributes(),
                                **request_payload,
                                agent_tool_name=call.name,
                                agent_tool_args_json=json.dumps(
                                    call.args or {}, sort_keys=True
                                ),
                            ) as tool_span:
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
                                execution_result = result["execution_result"]
                                tool_span.set_attributes(
                                    {
                                        "agent.tool.status": (
                                            "error"
                                            if execution_result.model_response.get(
                                                "error"
                                            )
                                            else "ok"
                                        ),
                                        "agent.tool.metadata_type": (
                                            type(execution_result.metadata).__name__
                                            if execution_result.metadata is not None
                                            else "None"
                                        ),
                                        **self.state.telemetry_attributes(),
                                    }
                                )
                                if execution_result.model_response.get("error"):
                                    tool_span.set_attribute(
                                        "agent.tool.error",
                                        str(execution_result.model_response["error"]),
                                    )

                    contents.append(types.UserContent(parts=tool_parts))
                    self.render_todos(previous_state, self.state)
                    run_span.set_attributes(self.state.telemetry_attributes())
                    turn_span.set_attributes(self.state.telemetry_attributes())


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
        print(Markdown(f"```text\n{clip_text(metadata.contents)}\n```"))
    if isinstance(metadata, WriteFileMetadata):
        print()
        print(f"Wrote file: {metadata.path}")
    if isinstance(metadata, EditFileMetadata):
        print()
        print(f"Edited file: {metadata.path}")
    if isinstance(metadata, BashMetadata):
        print()
        print(f"Ran command: {metadata.command}")
        print(Markdown(f"```text\n{clip_text(metadata.stdout)}\n```"))
        if metadata.stderr:
            print(Markdown(f"```text\n{clip_text(metadata.stderr)}\n```"))
    if isinstance(metadata, GeneratePlanMetadata):
        print()
        print("[green]Switched to execute mode.[/green]")
    return
