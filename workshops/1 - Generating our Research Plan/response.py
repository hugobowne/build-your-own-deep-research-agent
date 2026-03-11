import warnings
from google.genai import Client
from rich import print

warnings.filterwarnings(
    "ignore",
    message="Interactions usage is experimental and may change in future versions.",
    category=UserWarning,
)


client = Client()

response = client.interactions.create(
    model="gemini-3-flash-preview", input="This is a test"
)

print(response)
