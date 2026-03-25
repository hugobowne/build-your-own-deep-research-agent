import asyncio
import os

from exa_py import Exa
from google.genai import Client, types
from rich.live import Live
from rich import box
from rich.table import Table

from agent import Agent, render_message, render_tool_call, render_tool_result
from state import AgentContext, RunConfig, RunState
from tools import (
    DELEGATE_SEARCH_TOOL,
    MODIFY_TODO_TOOL,
    PARENT_SYSTEM_INSTRUCTION,
    READ_FILE_TOOL,
    SEARCH_SUBAGENT_SYSTEM_INSTRUCTION,
    SEARCH_WEB_TOOL,
)


def truncate_cell(text: str, max_length: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 4].rstrip()}...."


def render_subagent_table(statuses: dict[str, str]) -> Table:
    table = Table(title="Subagents", box=box.SQUARE, show_lines=True)
    table.add_column("Search Query", no_wrap=True)
    table.add_column("Latest Action", no_wrap=True)
    for query, status in statuses.items():
        table.add_row(truncate_cell(query, 60), truncate_cell(status, 72))
    return table


async def run_search_subagent(
    exa: Exa,
    query: str,
    context: AgentContext,
) -> dict[str, str]:
    child_agent = Agent(
        client=Client(),
        config=RunConfig(max_iterations=4),
        state=RunState(),
        context=AgentContext(
            exa=exa,
            live=context.live,
            subagent_statuses=context.subagent_statuses,
        ),
        tools=[SEARCH_WEB_TOOL],
        system_instruction=SEARCH_SUBAGENT_SYSTEM_INSTRUCTION,
    )

    async def update_tool_call(
        call: types.FunctionCall,
        config: RunConfig,
        state: RunState,
        context: AgentContext,
    ) -> None:
        if call.name == "search_web" and call.args is not None and "query" in call.args:
            context.subagent_statuses[query] = f"search_web: {call.args['query']}"
        else:
            context.subagent_statuses[query] = f"Calling {call.name}"
        if context.live is not None:
            context.live.update(render_subagent_table(context.subagent_statuses))

    child_agent.on("llm_tool_call", update_tool_call)

    child_contents: list[types.Content] = [
        types.UserContent(parts=[types.Part.from_text(text=query)])
    ]
    context.subagent_statuses[query] = "Starting"
    if context.live is not None:
        context.live.update(render_subagent_table(context.subagent_statuses))

    final_message = await child_agent.run_until_idle(child_contents)
    final_text = "\n".join(
        part.text for part in final_message.parts if part.text
    ).strip()
    context.subagent_statuses[query] = final_text[:100] or "Done"
    if context.live is not None:
        context.live.update(render_subagent_table(context.subagent_statuses))
    return {"query": query, "answer": final_text}


async def run_search_subagents(
    exa: Exa,
    queries: list[str],
    context: AgentContext,
) -> list[dict[str, str]]:
    context.subagent_statuses = {query: "Queued" for query in queries}
    with Live(
        render_subagent_table(context.subagent_statuses), refresh_per_second=8
    ) as live:
        context.live = live
        tasks = [
            asyncio.create_task(run_search_subagent(exa, query, context))
            for query in queries
        ]
        results = await asyncio.gather(*tasks)
        live.update(render_subagent_table(context.subagent_statuses))
        context.live = None
        return results


async def main() -> None:
    exa_api_key = os.getenv("EXA_API_KEY")
    if not exa_api_key:
        raise RuntimeError("EXA_API_KEY is required for step 07.")

    exa = Exa(api_key=exa_api_key)
    context = AgentContext(
        exa=exa,
        search_agent_runner=lambda queries: run_search_subagents(exa, queries, context),
    )
    agent = Agent(
        client=Client(),
        config=RunConfig(max_iterations=5),
        state=RunState(),
        context=context,
        tools=[READ_FILE_TOOL, MODIFY_TODO_TOOL, DELEGATE_SEARCH_TOOL],
        system_instruction=PARENT_SYSTEM_INSTRUCTION,
    )

    agent.on("message", render_message)
    agent.on("llm_tool_call", render_tool_call)
    agent.on("tool_result", render_tool_result)

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
