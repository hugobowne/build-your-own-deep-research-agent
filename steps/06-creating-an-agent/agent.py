import asyncio
from copy import deepcopy
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

HookType: TypeAlias = Literal["message", "llm_tool_call", "tool_result"]
AnyHook: TypeAlias = MessageHook | LLMToolCallHook | ToolResultHook

SYSTEM_INSTRUCTION = """
Use todos to track progress.
You must add todos before you do anything.
Make sure you check off all todos before you end.
""".strip()


class Agent:
    def __init__(
        self,
        *,
        client: Client,
        config: RunConfig,
        state: RunState,
        context: AgentContext,
        tools: list[Tool],
    ) -> None:
        self.client = client
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

    async def emit(self, hook_type: HookType, **kwargs: Any) -> None:
        for hook in self.hooks[hook_type]:
            result = hook(**kwargs)
            if result is not None:
                await result

    def prepare_request(
        self, contents: list[types.Content]
    ) -> tuple[types.GenerateContentConfig, list[types.Content]]:
        tools = []
        if self.state.iteration_count < self.config.max_iterations:
            tools = [tool.to_genai_tool() for tool in self.tools.values()]
        config = types.GenerateContentConfig(
            tools=tools,
            system_instruction=SYSTEM_INSTRUCTION,
        )
        return config, list(contents)

    async def execute_tool_call(self, call: types.FunctionCall) -> dict[str, Any]:
        tool = self.tools.get(call.name)
        if tool is None:
            raise RuntimeError(f"Unknown tool: {call.name}")

        if call.args is None:
            raise RuntimeError(f"Tool call '{call.name}' did not include arguments.")

        args = tool.args_model.model_validate(call.args)
        response = await tool.handler(args, self.state, self.context)
        return {"name": call.name, "response": response}

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

        added_todos = [
            todo for todo in current_state.todos if todo not in previous_state.todos
        ]

        for todo in added_todos:
            print(f"+[ ] {todo}")

    async def run_until_idle(self, contents: list[types.Content]) -> None:
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
                return

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
                    result=result,
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
            print(f"\nAssistant:\n{part.text}")


async def render_tool_call(
    call: types.FunctionCall,
    config: RunConfig,
    state: RunState,
    context: AgentContext,
) -> None:
    print(f"\nTool Call: {call.name}")
    print(call.args)


async def main() -> None:
    client = Client()
    config = RunConfig(max_iterations=5)
    state = RunState()
    context = AgentContext()
    agent = Agent(
        client=client,
        config=config,
        state=state,
        context=context,
        tools=[READ_FILE_TOOL, MODIFY_TODO_TOOL],
    )

    agent.on("message", render_message)
    agent.on("llm_tool_call", render_tool_call)

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
        await agent.run_until_idle(contents)


if __name__ == "__main__":
    asyncio.run(main())
