import warnings

from google.genai import Client
from rich import print
from rich.markdown import Markdown

warnings.filterwarnings(
    "ignore",
    message="Interactions usage is experimental and may change in future versions.",
    category=UserWarning,
)


SCOPING_MODEL = "gemini-3-flash-preview"

clarify_scope_tool = {
    "type": "function",
    "name": "clarifyScope",
    "description": "Ask the user a clarifying question to better understand the deep research request.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The next clarifying question to ask the user.",
            }
        },
        "required": ["question"],
    },
}


def run_clarification():
    client = Client()
    previous_interaction_id = None

    print()
    print("[bold cyan]What do you want to run deep research on?[/bold cyan]")
    initial_request = input("> ")
    print()

    response = client.interactions.create(
        model=SCOPING_MODEL,
        input=f"""
You're helping scope a deep research request.
Use the clarifyScope tool only when you need more information from the user.
When you have enough information, stop calling the tool and return only a concise markdown research brief.
The brief should be 4-6 sentences and clearly explain the request in plain language.
Start with "The user requested a deep research report on ..."
Then explain the main areas they want covered, what they are especially keen to explore, and any comparison angle or framing they asked for.
Do not answer the research request itself.

User request: {initial_request}
""",
        tools=[clarify_scope_tool],
        previous_interaction_id=previous_interaction_id,
    )
    previous_interaction_id = response.id

    clarification_history = []

    while True:
        function_call = next(
            (output for output in response.outputs if output.type == "function_call"),
            None,
        )

        if not function_call or function_call.name != "clarifyScope":
            break

        question = function_call.arguments["question"]

        print()
        print(f"[bold cyan]{question}[/bold cyan]")
        answer = input("> ")
        print()
        clarification_history.append((question, answer))

        response = client.interactions.create(
            model=SCOPING_MODEL,
            input=[
                {
                    "type": "function_result",
                    "call_id": function_call.id,
                    "name": function_call.name,
                    "result": answer,
                }
            ],
            tools=[clarify_scope_tool],
            previous_interaction_id=previous_interaction_id,
        )
        previous_interaction_id = response.id

    return initial_request, clarification_history, response.outputs[-1].text


if __name__ == "__main__":
    _, _, response_text = run_clarification()

    print("[bold green]Research brief[/bold green]")
    print()
    print(Markdown(response_text))
