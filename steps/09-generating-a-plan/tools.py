import asyncio
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
class WriteFileMetadata:
    path: str
    contents: str


@dataclass(slots=True)
class EditFileMetadata:
    path: str
    old_text: str
    new_text: str


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


@dataclass(slots=True)
class GeneratePlanMetadata:
    todos: list[str]


@dataclass(slots=True)
class BashMetadata:
    command: str
    returncode: int
    stdout: str
    stderr: str


ToolMetadata: TypeAlias = (
    ReadFileMetadata
    | WriteFileMetadata
    | EditFileMetadata
    | ModifyTodoMetadata
    | SearchWebMetadata
    | DelegateSearchMetadata
    | GeneratePlanMetadata
    | BashMetadata
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


class WriteFileArgs(BaseModel):
    path: str
    contents: str


class EditFileArgs(BaseModel):
    path: str
    old_text: str = Field(
        ...,
        min_length=1,
        description="The exact text to replace.",
    )
    new_text: str = Field(
        ...,
        description="The replacement text.",
    )


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


async def write_file(
    args: WriteFileArgs,
    state: RunState,
    context: AgentContext,
) -> ToolExecutionResult:
    del state, context

    path = Path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args.contents, encoding="utf-8")

    return ToolExecutionResult(
        model_response={
            "result": f"Wrote file at path {args.path}",
            "path": args.path,
        },
        metadata=WriteFileMetadata(
            path=args.path,
            contents=args.contents,
        ),
    )


async def edit_file(
    args: EditFileArgs,
    state: RunState,
    context: AgentContext,
) -> ToolExecutionResult:
    del state, context

    path = Path(args.path)
    if not path.exists() or not path.is_file():
        return ToolExecutionResult(
            model_response={"error": f"File does not exist: {args.path}"}
        )

    contents = path.read_text(encoding="utf-8")
    if args.old_text not in contents:
        return ToolExecutionResult(
            model_response={
                "error": (
                    f"Could not find the requested text to replace in {args.path}"
                )
            }
        )

    updated_contents = contents.replace(args.old_text, args.new_text, 1)
    path.write_text(updated_contents, encoding="utf-8")

    return ToolExecutionResult(
        model_response={
            "result": f"Edited file at path {args.path}",
            "path": args.path,
        },
        metadata=EditFileMetadata(
            path=args.path,
            old_text=args.old_text,
            new_text=args.new_text,
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
            "A set of distinct search questions that you need an answer to. "
            "Each item should be written as a question for a sub-agent to answer, "
            "and each question should cover a meaningfully different aspect of the "
            "user's request. For example, for the history of Google, ask separate "
            "questions like who the first founding members were, how the founders "
            "met, and why they chose the name Google."
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


class BashArgs(BaseModel):
    command: str = Field(
        ...,
        min_length=1,
        description="A bash command to run in the current working directory.",
    )


async def bash(
    args: BashArgs,
    state: RunState,
    context: AgentContext,
) -> ToolExecutionResult:
    del state, context

    try:
        process = await asyncio.create_subprocess_shell(
            args.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=30,
        )
    except TimeoutError:
        return ToolExecutionResult(
            model_response={
                "error": f"Command timed out after 30 seconds: {args.command}"
            }
        )

    stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
    returncode = process.returncode or 0

    return ToolExecutionResult(
        model_response={
            "result": f"""
Executed the following command:
{args.command}

Exit code: {returncode}

<result>
{stdout}
</result>

<stderr>
{stderr}
</stderr>""".strip(),
        },
        metadata=BashMetadata(
            command=args.command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        ),
    )


class GeneratePlanArgs(BaseModel):
    todos: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Call this when you have enough information from the user. "
            "Provide the initial list of todos needed to execute the task."
        ),
    )

    @field_validator("todos")
    @classmethod
    def validate_todos(cls, todos: list[str]) -> list[str]:
        normalized_todos: list[str] = []
        seen: set[str] = set()
        for todo in todos:
            normalized = " ".join(todo.split())
            if not normalized:
                raise ValueError("Todos must not be empty.")
            normalized_key = normalized.lower()
            if normalized_key in seen:
                raise ValueError("Todos must be distinct.")
            seen.add(normalized_key)
            normalized_todos.append(normalized)
        return normalized_todos


async def generate_plan(
    args: GeneratePlanArgs,
    state: RunState,
    context: AgentContext,
) -> ToolExecutionResult:
    del context
    added = state.add_todos(args.todos)
    state.mode = "execute"
    return ToolExecutionResult(
        model_response={
            "result": "Plan accepted. Start executing the task.",
            "todos": list(state.todos),
            "mode": state.mode,
        },
        metadata=GeneratePlanMetadata(todos=added),
    )


READ_FILE_TOOL = Tool(
    name="read_file",
    description="Read a UTF-8 text file and return its contents.",
    args_model=ReadFileArgs,
    handler=read_file,
)

WRITE_FILE_TOOL = Tool(
    name="write_file",
    description="Write a UTF-8 text file to disk.",
    args_model=WriteFileArgs,
    handler=write_file,
)

EDIT_FILE_TOOL = Tool(
    name="edit_file",
    description="Replace an exact text snippet in a UTF-8 text file.",
    args_model=EditFileArgs,
    handler=edit_file,
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

BASH_TOOL = Tool(
    name="bash",
    description="Run a bash command in the current working directory and capture stdout and stderr.",
    args_model=BashArgs,
    handler=bash,
)

GENERATE_PLAN_TOOL = Tool(
    name="generate_plan",
    description="Call this when you have enough information from the user.",
    args_model=GeneratePlanArgs,
    handler=generate_plan,
)


PLAN_INSTRUCTION = """
You are Koroku a deep research planner built by Ivan.

Today's date is {today}.

When the user provides you with a query, make sure you clarify the scope of the query.
You should ask at most 2 times before generating your initial plan to tackle the user's query.
This is incredibly important to do so.

Be polite, positive and helpful where you can.
Use normal assistant messages to clarify the scope of the user's request.
Ask things like how detailed the final report should be, whether there are specific angles to cover, or whether there are sources they prefer.

Use the generate_plan tool when you're ready to start working on the query once you've gathered enough information from the user.
Make sure that this plan has todos which are distinct and help the model track different parts of the overall request.
This will be used to track progress, so the initial outline should be good.

IMPORTANT: Always include a final todo step: "Review the final drafted file to ensure all factual claims have corresponding inline citations."
This step is mandatory and must appear in every generated plan.
""".strip().format(today=TODAY)

SYSTEM_INSTRUCTION = """
You are Kuroko, a deep research agent built by Ivan. Today's date is {today}.

Your primary directive is to conduct deep, multi-layered research across any subject matter.
Never settle for surface-level summaries.

For every topic, you must:

1. Task Management: Use the modify_todo tool to track your progress. Use the "add" action for new sub-topics, and the "remove" action to mark them as completed.
2. Note Taking And Drafting: Use write_file and edit_file to maintain notes and to write the final report to a markdown file such as report.md.
3. Source Diversity: For each claim, query multiple independent sources. Trace the evolution of the topic over time and highlight where interpretations differ.
4. Writing Style: Write in cohesive prose rather than fragmented notes. Do not use numbered headers (e.g., "1. Botanical Origins"). Prefer depth and synthesis over terse summaries. Prefer longer, detailed paragraphs where possible.
5. Citations: When writing or editing files, you MUST preserve and include all citations and source links provided by the sub-agents. Never strip citations for the sake of narrative flow. A report without citations is considered a failure. Use standard markdown footnotes for citations in the text like [^1], [^2], etc., and include the corresponding footnote links at the bottom of the report (e.g., [^1]: https://...).
6. Rabbit Holes: If research reveals an important adjacent thread, add it to your todo list and investigate it.
7. Timeline Verification: Ground your work in the current date and explicitly look for the latest developments when the topic is evolving.
8. Search Strategy: Start broad to map the landscape, then go deeper into the most relevant sources.
9. Final Output: Save the report to a markdown file using write_file or edit_file. Do not dump the whole report into the chat response. Give the user a concise update on what you researched and what sources you used. As the final step before finishing the task, ensure the report includes a summary of everything researched at the top, and a short conclusion at the end summing up what was learned.
10. Local Inspection: Use the bash tool when you need to inspect the repository or run a local command.

Use delegate_search to find sources and citations that substantiate the facts you need.
This is important because it helps you cover more depth by letting sub-agents investigate distinct parts of the topic in parallel.
Pass 1 to 3 distinct search questions.
Write each delegated search as a question because it will be processed by a sub-agent.
Only include multiple questions when they cover meaningfully different parts of the request.
For example, if the user asks about the history of Google, good delegated questions would be:
1. Who were the first founding members of Google?
2. How did the founders of Google meet?
3. What were some of the reasons why they chose the name Google?
If one search question is enough, pass exactly one.
""".strip().format(today=TODAY)

SEARCH_SUBAGENT_SYSTEM_INSTRUCTION = """
You are a focused web research subagent.
Today's date is {today}.
Answer the user's query in natural language.
Cite sources inline using markdown links.
Use the search_web tool when needed.
Do not ask follow-up questions.

Cast a wide net, look for information that's relevant and then filter out what's relevant. you should make sure that you do at least 2 searches.

IMPORTANT: Your final response MUST include all source URLs. Every factual claim must have a corresponding citation. Never omit sources. Format citations as numbered references like [1], [2] and list the full URLs at the end of your response.
""".strip().format(today=TODAY)
