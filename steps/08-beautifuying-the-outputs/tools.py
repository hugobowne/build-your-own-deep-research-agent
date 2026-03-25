from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, TypeAlias, TypeVar

from google.genai import types
from pydantic import BaseModel, Field, field_validator

from state import AgentContext, RunState


ArgsT = TypeVar("ArgsT", bound=BaseModel)
MAX_DELEGATED_QUERIES = 3
TODAY = datetime.now().strftime("%d %B %Y")


@dataclass(slots=True)
class ReadFileMetadata:
    path: str
    contents: str


@dataclass(slots=True)
class ModifyTodoMetadata:
    action: Literal["add", "remove"]
    todos: list[str]


@dataclass(slots=True)
class SearchWebMetadata:
    query: str
    raw_results: Any


@dataclass(slots=True)
class DelegateSearchMetadata:
    queries: list[str]
    results: list[dict[str, str]]


ToolMetadata: TypeAlias = (
    ReadFileMetadata | ModifyTodoMetadata | SearchWebMetadata | DelegateSearchMetadata
)


@dataclass(slots=True)
class ToolExecutionResult:
    model_response: dict[str, Any]
    metadata: ToolMetadata | None = None


ToolHandler = Callable[[ArgsT, RunState, AgentContext], Awaitable[ToolExecutionResult]]


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler

    def to_genai_tool(self) -> types.Tool:
        schema = self.args_model.model_json_schema()
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=self.name,
                    description=self.description,
                    parameters=types.Schema(
                        type="OBJECT",
                        properties=schema["properties"],
                        required=schema.get("required", []),
                    ),
                )
            ]
        )


class ReadFileArgs(BaseModel):
    path: str


async def read_file(
    args: ReadFileArgs,
    state: RunState,
    context: AgentContext,
) -> ToolExecutionResult:
    path = Path(args.path)
    if not path.exists() or not path.is_file():
        return ToolExecutionResult(
            model_response={"error": f"File does not exist: {args.path}"}
        )

    contents = path.read_text(encoding="utf-8")

    return ToolExecutionResult(
        model_response={
            "result": f"""
Read file at path {args.path}

<content>
{contents}
</content>""".strip()
        },
        metadata=ReadFileMetadata(
            path=args.path,
            contents=contents,
        ),
    )


class ModifyTodoArgs(BaseModel):
    action: Literal["add", "remove"]
    todos: list[str]


async def modify_todo(
    args: ModifyTodoArgs,
    state: RunState,
    context: AgentContext,
) -> ToolExecutionResult:
    if args.action == "add":
        state.add_todos(args.todos)
        return ToolExecutionResult(
            model_response={
                "result": f"""
Todos updated to

<todos>
{chr(10).join(state.todos)}
</todos>""".strip()
            },
            metadata=ModifyTodoMetadata(action=args.action, todos=list(args.todos)),
        )

    requested = [todo.strip() for todo in args.todos]
    missing = []
    existing_lower = {todo.lower() for todo in state.todos}
    for todo in requested:
        if todo.lower() not in existing_lower:
            missing.append(todo)

    if missing:
        return ToolExecutionResult(
            model_response={"error": f"Todos not found: {', '.join(missing)}"}
        )

    state.remove_todos(args.todos)
    return ToolExecutionResult(
        model_response={
            "result": f"""
Todos updated to

<todos>
{chr(10).join(state.todos)}
</todos>""".strip()
        },
        metadata=ModifyTodoMetadata(action=args.action, todos=list(args.todos)),
    )


class SearchWebArgs(BaseModel):
    query: str


async def search_web(
    args: SearchWebArgs,
    state: RunState,
    context: AgentContext,
) -> ToolExecutionResult:
    exa = context.exa
    if exa is None:
        return ToolExecutionResult(
            model_response={"error": "Exa client is not configured."}
        )

    results = exa.search(
        args.query,
        num_results=10,
        type="auto",
        contents={"highlights": {"max_characters": 4000}},
    )

    formatted_results: list[str] = []
    for item in results.results:
        highlights = item.highlights or []
        formatted_results.append(
            f"""
<result>
<title>{item.title or ""}</title>
<url>{item.url}</url>
<highlights>
{chr(10).join(f"- {highlight}" for highlight in highlights)}
</highlights>
</result>""".strip()
        )

    return ToolExecutionResult(
        model_response={
            "result": f"""
Search results for: {args.query}

<results>
{chr(10).join(formatted_results)}
</results>""".strip()
        },
        metadata=SearchWebMetadata(
            query=args.query,
            raw_results=results,
        ),
    )


class DelegateSearchArgs(BaseModel):
    queries: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_DELEGATED_QUERIES,
        description=(
            "A set of distinct search queries/questions that you need an answer to. "
            "Each query should cover a different aspect of the user's request."
        ),
    )

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, queries: list[str]) -> list[str]:
        normalized_queries: list[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized = " ".join(query.split())
            if not normalized:
                raise ValueError("Queries must not be empty.")
            normalized_key = normalized.lower()
            if normalized_key in seen:
                raise ValueError("Queries must be distinct.")
            seen.add(normalized_key)
            normalized_queries.append(normalized)
        return normalized_queries


async def delegate_search(
    args: DelegateSearchArgs,
    state: RunState,
    context: AgentContext,
) -> ToolExecutionResult:
    if context.search_agent_runner is None:
        return ToolExecutionResult(
            model_response={"error": "Search subagent runner is not configured."}
        )

    results = await context.search_agent_runner(args.queries)
    if not results:
        return ToolExecutionResult(
            model_response={"error": "Search subagent did not return any results."}
        )

    query_answers_xml = []
    for item in results:
        query_answers_xml.append(
            f"""
<query_answer>
<query>{item["query"]}</query>
<answer>
{item["answer"]}
</answer>
</query_answer>""".strip()
        )

    return ToolExecutionResult(
        model_response={
            "queries": list(args.queries),
            "results": results,
        },
        metadata=DelegateSearchMetadata(
            queries=list(args.queries),
            results=results,
        ),
    )


READ_FILE_TOOL = Tool(
    name="read_file",
    description="Read a UTF-8 text file and return its contents.",
    args_model=ReadFileArgs,
    handler=read_file,
)

MODIFY_TODO_TOOL = Tool(
    name="modify_todo",
    description="Add or remove todos from the current run state.",
    args_model=ModifyTodoArgs,
    handler=modify_todo,
)

SEARCH_WEB_TOOL = Tool(
    name="search_web",
    description="Search the web with Exa and return cited results.",
    args_model=SearchWebArgs,
    handler=search_web,
)

DELEGATE_SEARCH_TOOL = Tool(
    name="delegate_search",
    description="Delegate 1 to 3 distinct web research queries to search subagents.",
    args_model=DelegateSearchArgs,
    handler=delegate_search,
)


PARENT_SYSTEM_INSTRUCTION = """
You are a coding agent.
Today's date is {today}.
Use todos to track progress.
You must add todos before you do anything.
Make sure you check off all todos before you end.

When the user asks for web research or current information, use delegate_search.
Pass 1 to 3 distinct search queries.
Only include multiple queries when they cover meaningfully different questions.
If one search query is enough, pass exactly one.
""".strip().format(today=TODAY)

SEARCH_SUBAGENT_SYSTEM_INSTRUCTION = """
You are a focused web research subagent.
Today's date is {today}.
Answer the user's query in natural language.
Cite sources inline using markdown links.
Use the search_web tool when needed.
Do not ask follow-up questions.

Cast a wide net, look for information that's relevant and then filter out what's relevant. you should make sure that you do at least 2 searches
""".strip().format(today=TODAY)
