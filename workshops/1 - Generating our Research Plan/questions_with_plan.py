import warnings

from google.genai import Client
from rich import print
from rich.markdown import Markdown

from questions import run_clarification

warnings.filterwarnings(
    "ignore",
    message="Interactions usage is experimental and may change in future versions.",
    category=UserWarning,
)


PLANNING_MODEL = "gemini-3.1-pro-preview"

generate_plan_tool = {
    "type": "function",
    "name": "generate_plan",
    "description": "Create a structured research plan from the clarified request.",
    "parameters": {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "A 3-4 sentence natural-language summary of what the research plan will do.",
            },
            "todos": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concrete next-step tasks for the research agent.",
            },
        },
        "required": ["response", "todos"],
    },
}


if __name__ == "__main__":
    initial_request, clarification_history, response_text = run_clarification()

    print("[bold green]Research brief[/bold green]")
    print()
    print(Markdown(response_text))
    print()

    client = Client()
    plan_response = client.interactions.create(
        model=PLANNING_MODEL,
        input=f"""
Create a structured research plan for this clarified deep research request.
Call the generate_plan tool exactly once.
Put a 3-4 sentence natural-language first response to the user in `response`.
That response should acknowledge what they want, restate the research focus clearly, and give a short TL;DR of the plan.
Put the actionable next steps in `todos` as a list of strings.
Do not reply with normal text.

Initial request: {initial_request}
Clarifications: {clarification_history}
Scoped summary: {response_text}
""",
        tools=[generate_plan_tool],
    )

    function_call = next(
        (output for output in plan_response.outputs if output.type == "function_call"),
        None,
    )

    print("[bold magenta]Response[/bold magenta]")
    print()
    print(Markdown(function_call.arguments["response"]))
    print()

    print("[bold magenta]Todos[/bold magenta]")
    print()
    for todo in function_call.arguments["todos"]:
        print(f"[ ] {todo}")
