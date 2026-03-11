from abc import ABC, abstractmethod
from typing import Any, Awaitable
import os
import subprocess

from google.genai import types
from pydantic import BaseModel


class AgentContext:
    pass


class ToolResult(BaseModel):
    error: bool
    name: str
    response: dict[str, Any]

    model_config = {"arbitrary_types_allowed": True}

    def to_genai_part(self) -> types.Part:
        return types.Part.from_function_response(
            name=self.name,
            response=self.response,
        )


class AgentTool(BaseModel, ABC):
    @classmethod
    def tool_name(cls) -> str:
        return cls.__name__

    def tool_result(self, *, error: bool, response: dict[str, Any]) -> ToolResult:
        return ToolResult(
            error=error,
            name=self.__class__.tool_name(),
            response=response,
        )

    @classmethod
    def to_genai_tool(cls) -> types.Tool:
        json_schema = cls.model_json_schema()
        tool_name = cls.tool_name()
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=tool_name,
                    description=json_schema.get(
                        "description",
                        f"Call the {tool_name} tool",
                    ),
                    parameters=types.Schema(
                        type="OBJECT",
                        properties=json_schema["properties"],
                        required=json_schema.get("required", []),
                    ),
                )
            ]
        )

    @abstractmethod
    def execute(self, _context: AgentContext) -> Awaitable[ToolResult]:
        raise NotImplementedError


class ReadFile(AgentTool):
    path: str

    async def execute(self, _context: AgentContext) -> ToolResult:
        if not os.path.exists(self.path) or not os.path.isfile(self.path):
            return self.tool_result(error=True, response={"error": "File does not exist"})

        try:
            with open(self.path, "r", encoding="utf-8") as file:
                return self.tool_result(
                    error=False,
                    response={"file_content": file.read()},
                )
        except Exception as exc:
            return self.tool_result(
                error=True,
                response={"error": f"Failed to read '{self.path}': {exc}"},
            )


class Write(AgentTool):
    path: str
    content: str

    async def execute(self, _context: AgentContext) -> ToolResult:
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)

            with open(self.path, "w", encoding="utf-8") as file:
                file.write(self.content)

            return self.tool_result(
                error=False,
                response={"result": f"Successfully wrote content to {self.path}"},
            )
        except Exception as exc:
            return self.tool_result(
                error=True,
                response={"error": f"Failed to write '{self.path}': {exc}"},
            )


class Edit(AgentTool):
    path: str
    old_str: str
    new_str: str
    replace_all: bool = False

    async def execute(self, _context: AgentContext) -> ToolResult:
        if not os.path.exists(self.path) or not os.path.isfile(self.path):
            return self.tool_result(
                error=True,
                response={"error": f"Path does not exist: {self.path}"},
            )

        try:
            with open(self.path, "r", encoding="utf-8") as file:
                original = file.read()
        except Exception as exc:
            return self.tool_result(
                error=True,
                response={"error": f"Failed to read '{self.path}': {exc}"},
            )

        if self.old_str not in original:
            return self.tool_result(
                error=True,
                response={"error": f"old_str not found in {self.path}"},
            )

        occurrences = original.count(self.old_str)
        if not self.replace_all and occurrences > 1:
            return self.tool_result(
                error=True,
                response={
                    "error": (
                        "old_str appears multiple times; set replace_all=True or "
                        "provide a more specific old_str"
                    )
                },
            )

        if self.replace_all:
            updated = original.replace(self.old_str, self.new_str)
        else:
            updated = original.replace(self.old_str, self.new_str, 1)

        try:
            with open(self.path, "w", encoding="utf-8") as file:
                file.write(updated)
            return self.tool_result(
                error=False,
                response={"result": f"Applied edit to {self.path}"},
            )
        except Exception as exc:
            return self.tool_result(
                error=True,
                response={"error": f"Failed to write '{self.path}': {exc}"},
            )


class Bash(AgentTool):
    command: str
    working_dir: str = "."
    timeout_seconds: int = 30

    async def execute(self, _context: AgentContext) -> ToolResult:
        if self.timeout_seconds <= 0:
            return self.tool_result(
                error=True,
                response={"error": "timeout_seconds must be > 0"},
            )

        if not os.path.isdir(self.working_dir):
            return self.tool_result(
                error=True,
                response={"error": f"Invalid working directory: {self.working_dir}"},
            )

        try:
            completed = subprocess.run(
                self.command,
                shell=True,
                cwd=self.working_dir,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            return self.tool_result(
                error=False,
                response={"result": f"{completed.stdout}{completed.stderr}"},
            )
        except subprocess.TimeoutExpired:
            return self.tool_result(
                error=True,
                response={"error": f"Command timed out after {self.timeout_seconds}s"},
            )
        except Exception as exc:
            return self.tool_result(
                error=True,
                response={"error": f"Bash execution failed: {exc}"},
            )


TOOLS = [ReadFile, Write, Edit, Bash]
