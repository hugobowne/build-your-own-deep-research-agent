import asyncio

from rich.markdown import Markdown
from rich.panel import Panel

from shell import Shell


def make_subagent_panel(step: str, notes: list[str]) -> Panel:
    body = "\n".join(f"- {note}" for note in notes)
    return Panel(
        f"[bold]Status:[/bold] {step}\n\n{body}",
        title="Subagent: Web Research",
        border_style="cyan",
    )


async def handle_submit(shell: Shell, text: str) -> None:
    shell.print(f"[bold cyan]User[/bold cyan]\n{text}")
    shell.set_loading(True)

    try:
        shell.print(
            f"[bold white]Assistant[/bold white]\nOk, let me call a subagent for this.\n\nWorking on: {text}"
        )

        async def run_subagent() -> None:
            shell.update_entry(
                "subagent-call",
                make_subagent_panel(
                    "Starting",
                    [
                        "Received a task from the main agent",
                        "Preparing to search for relevant sources",
                    ],
                ),
            )
            await asyncio.sleep(0.6)
            shell.update_entry(
                "subagent-call",
                make_subagent_panel(
                    "Searching",
                    [
                        "Finding the most relevant sources",
                        "Scoping what to read first",
                    ],
                ),
            )
            await asyncio.sleep(0.8)
            shell.update_entry(
                "subagent-call",
                make_subagent_panel(
                    "Reading",
                    [
                        "Extracting key points from source 1",
                        "Comparing it with source 2",
                    ],
                ),
            )
            await asyncio.sleep(0.8)
            shell.update_entry(
                "subagent-call",
                make_subagent_panel(
                    "Summarizing",
                    [
                        "Combining findings into a short brief",
                        "Handing results back to the main agent",
                    ],
                ),
            )
            await asyncio.sleep(0.8)
            shell.update_entry(
                "subagent-call",
                make_subagent_panel(
                    "Complete",
                    [
                        "Returned a short summary to the main agent",
                    ],
                ),
            )

        await run_subagent()
        await asyncio.sleep(1.0)
        shell.update_region(
            "todos",
            "[bold]Todos[/bold]\n1. Clarify the request\n[ ] 2. Draft the plan\n[ ] 3. Continue research",
        )

        await asyncio.sleep(1.0)
        shell.print(
            Markdown(
                "I have a first-pass research plan ready. "
                "Next I would branch into the main research loop."
            )
        )
        shell.update_region(
            "todos",
            "[bold]Todos[/bold]\n1. Clarify the request\n2. Draft the plan\n[ ] 3. Continue research",
        )

        await asyncio.sleep(1.0)
        shell.print("Research loop would continue from here.")
        shell.update_region(
            "todos",
            "[bold]Todos[/bold]\n1. Clarify the request\n2. Draft the plan\n3. Continue research",
        )
    finally:
        shell.set_loading(False)


async def on_ready(shell: Shell) -> None:
    shell.print(
        Markdown(
            "This is a reusable shell.\n"
            "It gives you a pinned input, `print(...)`, inline `update_entry(...)`, and named regions you can update later.\n\n"
            "Submit a message to see a single inline subagent/tool block update in place."
        )
    )
    shell.update_region(
        "todos",
        "[bold]Todos[/bold]\n[ ] 1. Clarify the request\n[ ] 2. Draft the plan\n[ ] 3. Continue research",
    )


def main() -> None:
    shell = Shell()
    shell.initialize(
        on_submit=lambda text: handle_submit(shell, text),
        on_ready=lambda: on_ready(shell),
    )
    shell.run()


if __name__ == "__main__":
    main()
