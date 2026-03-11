import warnings

from google.genai import Client
from rich import print

warnings.filterwarnings(
    "ignore",
    message="Interactions usage is experimental and may change in future versions.",
    category=UserWarning,
)


MODEL = "gemini-3-flash-preview"

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


client = Client()

print()
print("[bold cyan]What do you want to run deep research on?[/bold cyan]")
initial_request = input("> ")
print()

response = client.interactions.create(
    model=MODEL,
    input=(
        f"You're a deep research agent. Use the clarify Scope tool if you need more information from the user before generating your response. if not just reply normally to {initial_request}"
    ),
    tools=[clarify_scope_tool],
)

function_call = next(
    (output for output in response.outputs if output.type == "function_call"),
    None,
)

print("[bold green]Response[/bold green]")
print()

if function_call and function_call.name == "clarifyScope":
    print("<Function Call>")
    print(function_call)
    print("</Function Call>")

else:
    print(response.outputs)
