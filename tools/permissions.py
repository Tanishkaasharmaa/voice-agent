"""Tool permission model.

Every tool has a permission level:
* SAFE     — executes without confirmation (read-only, benign);
* CONFIRM  — requires user confirmation before executing;
* ASK      — requires confirmation on every call (destructive/sensitive).

Confirmation is a policy decision the LLM can never override — it happens in
the ToolRegistry before the handler runs.
"""

from __future__ import annotations

from enum import Enum


class PermissionLevel(str, Enum):
    SAFE = "safe"
    CONFIRM = "confirm"
    ASK = "ask"


# Tool names the agent may never touch, regardless of what the LLM asks for.
BLOCKED_TOOLS: set[str] = set()
# Arguments that are never accepted by any tool (path/shell safety).
BLOCKED_ARGS: set[str] = {"--force", "-f", "--yes", "-y", "--no-input"}


class PermissionDenied(Exception):
    pass


class ToolNotFound(Exception):
    pass


class ToolArgumentError(Exception):
    pass


def validate_parameters(parameters: dict, args: dict) -> None:
    """Validate tool arguments against the declared JSON schema."""
    props = parameters.get("properties", {})
    required = parameters.get("required", [])
    for name in required:
        if name not in args or args[name] in (None, ""):
            raise ToolArgumentError(f"Missing required argument: {name}")
    for name, value in args.items():
        if name in BLOCKED_ARGS:
            raise ToolArgumentError(f"Argument '{name}' is blocked")
        if name not in props:
            raise ToolArgumentError(f"Unknown argument: {name}")
        schema = props[name]
        type_ = schema.get("type")
        if type_ == "string" and not isinstance(value, str):
            raise ToolArgumentError(f"Argument '{name}' must be a string")
        if type_ == "number" and not isinstance(value, (int, float)):
            raise ToolArgumentError(f"Argument '{name}' must be a number")
        if type_ == "boolean" and not isinstance(value, bool):
            raise ToolArgumentError(f"Argument '{name}' must be a boolean")
        if type_ == "array" and not isinstance(value, list):
            raise ToolArgumentError(f"Argument '{name}' must be a list")
        if type_ == "integer" and not isinstance(value, int):
            raise ToolArgumentError(f"Argument '{name}' must be an integer")
