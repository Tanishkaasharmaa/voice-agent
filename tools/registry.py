"""ToolRegistry — the only way tools execute.

* every tool declares name, description, JSON schema, permission level and
  a confirmation requirement;
* arguments are validated against the schema before any handler runs;
* confirmation is enforced via a context callback (the UI shows a dialog);
* the LLM can never execute arbitrary Python or unrestricted shell here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from tools.permissions import (
    BLOCKED_TOOLS,
    PermissionDenied,
    PermissionLevel,
    ToolArgumentError,
    ToolNotFound,
    validate_parameters,
)

ToolHandler = Callable[..., dict[str, Any]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    permission: PermissionLevel = PermissionLevel.SAFE
    requires_confirmation: bool = False

    def to_ollama_schema(self) -> dict[str, Any]:
        """JSON schema in the shape Ollama's tools API expects."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolContext:
    """Per-execution context injected into every tool handler.

    Attributes are populated by the Application; handlers use them to reach
    memory, settings, and (rarely) to request UI confirmation.
    """

    def __init__(self) -> None:
        self.memory: Any = None
        self.settings: Any = None
        self.confirm: Callable[[str], bool] = lambda prompt: False
        self.log: Callable[[str], None] = lambda _msg: None


@dataclass
class ToolResult:
    success: bool
    output: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "data": self.data,
            "error": self.error,
        }


class ToolRegistry:
    def __init__(self, context: ToolContext | None = None) -> None:
        self.context = context or ToolContext()
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_many(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFound(f"Tool '{name}' is not registered")
        return tool

    def list_tools(self) -> list[Tool]:
        return sorted(self._tools.values(), key=lambda t: t.name)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_ollama_schema() for t in self.list_tools()]

    def tool_names(self) -> list[str]:
        return [t.name for t in self.list_tools()]

    # ------------------------------------------------------------ execution

    def execute(
        self,
        name: str,
        args: dict[str, Any],
        require_confirmation: bool | None = None,
    ) -> ToolResult:
        try:
            tool = self.get(name)
        except ToolNotFound as exc:
            return ToolResult(False, error=str(exc))
        if tool.name in BLOCKED_TOOLS:
            return ToolResult(False, error="Tool is blocked")

        try:
            validate_parameters(tool.parameters, args or {})
        except ToolArgumentError as exc:
            return ToolResult(False, error=str(exc))

        needs_confirm = tool.requires_confirmation
        if require_confirmation is not None:
            needs_confirm = require_confirmation
        elif self.context.settings is not None:
            if self.context.settings.get("tools.confirm_destructive", True) and tool.permission == PermissionLevel.CONFIRM:
                needs_confirm = True

        if needs_confirm or tool.permission == PermissionLevel.ASK:
            prompt = f"Allow LocalVoiceAgent to run '{tool.name}'?"
            if args:
                prompt += f"\nArguments: {args}"
            try:
                if not self.context.confirm(prompt):
                    raise PermissionDenied(f"User denied '{tool.name}'")
            except PermissionDenied as exc:
                return ToolResult(False, error=str(exc))

        started = time.monotonic()
        try:
            result = tool.handler(**args, _ctx=self.context)
        except Exception as exc:  # noqa: BLE001 - tool boundary
            duration = int((time.monotonic() - started) * 1000)
            if self.context.memory is not None:
                self.context.memory.record_tool_execution(
                    name, args, False, error=str(exc), duration_ms=duration
                )
            return ToolResult(False, error=str(exc))

        duration = int((time.monotonic() - started) * 1000)
        if not isinstance(result, dict):
            result = {"success": True, "output": str(result)}
        result.setdefault("success", True)
        result.setdefault("output", result.get("output") or "")
        result.setdefault("data", {})
        result.setdefault("error", "")
        if self.context.memory is not None:
            self.context.memory.record_tool_execution(
                name, args, bool(result.get("success")),
                output=str(result.get("output")) or None,
                error=str(result.get("error")) or None,
                duration_ms=duration,
            )
        if self.context.log:
            self.context.log(f"tool '{name}' -> {'ok' if result['success'] else 'error'} ({duration}ms)")
        return ToolResult(
            success=bool(result["success"]),
            output=str(result.get("output") or ""),
            data=result.get("data") or {},
            error=str(result.get("error") or ""),
        )
