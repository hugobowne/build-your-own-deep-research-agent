import asyncio
from typing import Any, Awaitable, Callable, Literal, TypeAlias

from google.genai import Client, types
from rich import print

from state import AgentContext, RunConfig, RunState
from tools import MODIFY_TODO_TOOL, READ_FILE_TOOL, Tool

MessageHook: TypeAlias = Callable[
    [types.Content, RunConfig, RunState, AgentContext],
    Awaitable[None] | None,
]

LLMToolCallHook: TypeAlias = Callable[
    [types.FunctionCall, RunConfig, RunState, AgentContext],
    Awaitable[None] | None,
]

ToolResultHook: TypeAlias = Callable[
    [types.FunctionCall, dict[str, Any], RunConfig, RunState, AgentContext],
    Awaitable[None] | None,
]

HookType: TypeAlias = Literal[
    "message",
    "llm_tool_call",
    "tool_result",
]
AnyHook: TypeAlias = MessageHook | LLMToolCallHook | ToolResultHook


class AgentRuntime:
    def __init__(
        self,
        *,
        config: RunConfig,
        state: RunState,
        context: AgentContext,
        tools: list[Tool],
    ) -> None:
        self.config = config
        self.state = state
        self.context = context
        self.tools = {tool.name: tool for tool in tools}
        self.hooks: dict[HookType, list[AnyHook]] = {
            "message": [],
            "llm_tool_call": [],
            "tool_result": [],
        }

    def on(self, hook_type: HookType, hook: AnyHook) -> None:
        self.hooks[hook_type].append(hook)

    def prepare_request(
        self, contents: list[types.Content]
    ) -> tuple[types.GenerateContentConfig, list[types.Content]]:
        tools = []
        if self.state.iteration_count < self.config.max_iterations:
            tools = [tool.to_genai_tool() for tool in self.tools.values()]
        config = types.GenerateContentConfig(tools=tools)
        return config, list(contents)

    async def emit(self, hook_type: HookType, **kwargs: Any) -> None:
        for hook in self.hooks[hook_type]:
            result = hook(**kwargs)
            if result is not None:
                await result

    async def execute_tool_call(self, call: types.FunctionCall) -> dict[str, Any]:
        tool = self.tools.get(call.name)
        if tool is None:
            raise RuntimeError(f"Unknown tool: {call.name}")

        if call.args is None:
            raise RuntimeError(f"Tool call '{call.name}' did not include arguments.")

        args = tool.args_model.model_validate(call.args)
        response = await tool.handler(args, self.state, self.context)
        return {"name": call.name, "response": response}


async def render_model_response(
    message: types.Content,
    config: RunConfig,
    state: RunState,
    context: AgentContext,
) -> None:
    for part in message.parts:
        if part.text:
            print(f"\nAssistant:\n{part.text}")


async def render_tool_call(
    call: types.FunctionCall,
    config: RunConfig,
    state: RunState,
    context: AgentContext,
) -> None:
    print(f"\nTool Call: {call.name}")
    print(call.args)


async def render_tool_result(
    call: types.FunctionCall,
    result: dict[str, Any],
    config: RunConfig,
    state: RunState,
    context: AgentContext,
) -> None:
    pass


async def main() -> None:
    client = Client()
    config = RunConfig(max_iterations=5)
    state = RunState()
    context = AgentContext()
    runtime = AgentRuntime(
        config=config,
        state=state,
        context=context,
        tools=[READ_FILE_TOOL, MODIFY_TODO_TOOL],
    )

    runtime.on("message", render_model_response)
    runtime.on("llm_tool_call", render_tool_call)
    runtime.on("tool_result", render_tool_result)

    contents: list[types.Content] = []

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        contents.append(
            types.UserContent(parts=[types.Part.from_text(text=user_input)])
        )
        while True:
            state.iteration_count += 1
            request_config, request_contents = runtime.prepare_request(contents)

            completion = await client.aio.models.generate_content(
                model=config.model,
                contents=request_contents,
                config=request_config,
            )

            message = completion.candidates[0].content
            contents.append(message)
            await runtime.emit(
                "message",
                message=message,
                config=config,
                state=state,
                context=context,
            )

            function_calls = [
                part.function_call for part in message.parts if part.function_call
            ]
            if not function_calls:
                break

            tool_parts: list[types.Part] = []
            for call in function_calls:
                await runtime.emit(
                    "llm_tool_call",
                    call=call,
                    config=config,
                    state=state,
                    context=context,
                )
                result = await runtime.execute_tool_call(call)
                await runtime.emit(
                    "tool_result",
                    call=call,
                    result=result,
                    config=config,
                    state=state,
                    context=context,
                )
                tool_parts.append(
                    types.Part.from_function_response(
                        name=result["name"],
                        response=result["response"],
                    )
                )

            contents.append(types.UserContent(parts=tool_parts))


if __name__ == "__main__":
    asyncio.run(main())
