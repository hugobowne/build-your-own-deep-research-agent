import asyncio
import warnings
from collections import deque
from typing import Any

from google.genai import Client, types
from rich import print
from rich.markdown import Markdown

import agent_tools
from agent_tools import AgentContext, ToolResult

warnings.filterwarnings(
    "ignore",
    message="Interactions usage is experimental and may change in future versions.",
    category=UserWarning,
)


MODEL = "gemini-3-flash-preview"
SYSTEM_PROMPT = """
You are a minimal deep research agent.
Work through the user's request step by step.
Use tools when needed.
If the user sends a steering update, treat it as a high-priority instruction that should shape the next part of your work.
If the user sends a queued message, treat it as a follow-up request that should be handled after the current task is complete.
Be concise when talking to the user.
""".strip()


class Agent:
    def __init__(self, model: str = MODEL):
        self.model = model
        self.client = Client()
        self.context = AgentContext()
        self.tools = {tool.__name__: tool for tool in agent_tools.TOOLS}

    def get_tools(self) -> list[types.Tool]:
        return [tool_cls.to_genai_tool() for tool_cls in self.tools.values()]

    async def execute_tool(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        tool_cls = self.tools.get(tool_name)
        if tool_cls is None:
            return ToolResult(
                error=True,
                name=tool_name,
                response={"error": f"Unknown tool: {tool_name}"},
            )

        try:
            tool_input = tool_cls.model_validate(args or {})
            return await tool_input.execute(self.context)
        except Exception as exc:
            return ToolResult(
                error=True,
                name=tool_name,
                response={"error": str(exc)},
            )

    async def step(self, conversation: list[types.Content]) -> tuple[bool, bool]:
        completion = await self.client.aio.models.generate_content(
            model=self.model,
            contents=conversation,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=self.get_tools(),
            ),
        )

        message = completion.candidates[0].content
        conversation.append(message)

        for part in message.parts:
            if part.text:
                print()
                print(Markdown(part.text))
                print()

        function_calls = [part.function_call for part in message.parts if part.function_call]
        if not function_calls:
            return False, False

        tool_parts: list[types.Part] = []
        for call in function_calls:
            call_args = call.args or {}
            print(f"[cyan]Tool:[/cyan] {call.name} {call_args}")
            result = await self.execute_tool(call.name, call_args)
            status = "[green]✓[/green]" if not result.error else "[red]✗[/red]"
            print(f"{status} [bold]{call.name}[/bold]")
            tool_parts.append(result.to_genai_part())

        conversation.append(types.UserContent(parts=tool_parts))
        return True, True


class AgentApp:
    def __init__(self):
        self.agent = Agent()
        self.conversation: list[types.Content] = []
        self.steer_queue: deque[str] = deque()
        self.message_queue: deque[str] = deque()
        self.wake_event = asyncio.Event()
        self.state = "idle"

    def enqueue_user_message(self, text: str) -> None:
        self.message_queue.append(text)
        self.wake_event.set()

    def enqueue_steer(self, text: str) -> None:
        self.steer_queue.append(text)

    def inject_queued_message(self) -> bool:
        if not self.message_queue:
            return False

        message = self.message_queue.popleft()
        self.conversation.append(
            types.UserContent(parts=[types.Part.from_text(text=message)])
        )
        print(f"[bold blue]Queued[/bold blue] {message}")
        return True

    def inject_steer_message(self) -> bool:
        if not self.steer_queue:
            return False

        steer = self.steer_queue.popleft()
        self.conversation.append(
            types.UserContent(
                parts=[
                    types.Part.from_text(
                        text=f"Steering update from the user: {steer}"
                    )
                ]
            )
        )
        print(f"[bold yellow]Steer[/bold yellow] {steer}")
        return True

    def inject_next_message(self) -> bool:
        if self.inject_steer_message():
            return True
        return self.inject_queued_message()

    async def runner(self) -> None:
        while True:
            if not self.message_queue and not self.steer_queue:
                self.state = "idle"
                self.wake_event.clear()
                await self.wake_event.wait()

            if self.state == "idle" and self.inject_next_message():
                self.state = "running"

            while self.state == "running":
                has_tool_calls, reached_checkpoint = await self.agent.step(self.conversation)

                if reached_checkpoint and self.inject_steer_message():
                    continue

                if not has_tool_calls:
                    if self.inject_next_message():
                        continue
                    self.state = "idle"
                    break

    async def input_loop(self) -> None:
        help_text = (
            "\nCommands:\n"
            "  /steer <message>  send a steering update after the next tool call\n"
            "  /exit             quit\n"
            "  anything else     queue a follow-up request for when the model finishes\n"
        )
        print(help_text)

        while True:
            prompt = f"\n[{self.state}] > "
            raw = await asyncio.to_thread(input, prompt)
            user_input = raw.strip()

            if not user_input:
                continue

            if user_input.lower() in {"/exit", "exit", "quit"}:
                raise SystemExit

            if user_input.startswith("/steer "):
                self.enqueue_steer(user_input.removeprefix("/steer ").strip())
                print("[yellow]Queued steering update[/yellow]")
                continue

            self.enqueue_user_message(user_input)


async def main() -> None:
    app = AgentApp()
    runner_task = asyncio.create_task(app.runner())

    try:
        await app.input_loop()
    finally:
        runner_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runner_task


if __name__ == "__main__":
    import contextlib

    asyncio.run(main())
