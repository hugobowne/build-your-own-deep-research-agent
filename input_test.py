from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import OptionList, Input, Label
from textual.widgets.option_list import Option
from rich.console import Console

console = Console()


class DeepResearchUI(App[dict]):
    # We use CSS to make it look like a floating panel
    CSS = """
    DeepResearchUI {
        height: auto;
        margin-bottom: 1;
    }
    
    #card {
        height: auto;
        border: round #8f8ca6; /* Soft muted border */
        background: transparent;
        padding: 1 2;
    }
    
    Label.title {
        text-style: bold;
        color: #d97757; /* Claude Copper */
        margin-bottom: 1;
    }
    
    OptionList {
        height: auto;
        max-height: 5;
        border: none;
        background: transparent;
        margin-bottom: 1;
    }
    
    OptionList:focus {
        border: none;
    }
    
    Input {
        border: tall #56949f; /* Muted cyan */
        background: transparent;
        width: 100%;
    }
    
    Input:focus {
        border: tall #9ccfd8; /* Bright cyan when typing */
    }
    """

    def __init__(self, ai_questions: list[str]):
        super().__init__()
        self.ai_questions = ai_questions
        # Default to the first question
        self.selected_context = (
            ai_questions[0] if ai_questions else "Custom Instruction"
        )

    def compose(self) -> ComposeResult:
        # Build the selectable options
        options = [Option(q, id=f"q_{i}") for i, q in enumerate(self.ai_questions)]
        # Always give them an escape hatch to reject/redirect
        options.append(
            Option("⛔ None of these (Provide new instructions)", id="redirect")
        )

        # Wrap everything in a Vertical container with the ID "card" for styling
        with Vertical(id="card"):
            yield Label(
                "✦ The AI has a few questions. Select one to answer, or redirect:",
                classes="title",
            )
            yield OptionList(*options, id="question-list")
            yield Input(
                placeholder="Type your response or new instructions here...",
                id="user-input",
            )

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        # Update our internal state as the user arrows up and down the list
        if event.option_id == "redirect":
            self.selected_context = "User Redirect"
        else:
            idx = int(event.option_id.split("_")[1])
            self.selected_context = self.ai_questions[idx]

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # When they press Enter in the input box, exit and return both pieces of data!
        self.exit({"context": self.selected_context, "response": event.value})


def main():
    console.print("\n[dim]Initializing Deep Research...[/dim]\n")

    # Simulated AI output
    mock_ai_questions = [
        "Should I focus on Python 3.10+ features exclusively?",
        "Do you want me to include error handling in the draft?",
        "Are we optimizing for speed or readability?",
    ]

    while True:
        try:
            app = DeepResearchUI(mock_ai_questions)

            # Run the UI inline!
            result = app.run(inline=True)

            if result and result["response"].strip():
                # Display what was captured
                console.print(f"[bold cyan]Action:[/bold cyan] {result['context']}")
                console.print(
                    f"[bold green]Your Input:[/bold green] {result['response']}\n"
                )

                # In a real app, you would pass `result` back to your LLM here
                break

        except KeyboardInterrupt:
            console.print("\n[dim]Research aborted.[/dim]")
            break


if __name__ == "__main__":
    main()
