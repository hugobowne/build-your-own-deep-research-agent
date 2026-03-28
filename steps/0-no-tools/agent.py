from google.genai import Client
from rich import print

client = Client()

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Please read the README.md file.",
)

print(response.text)
