import tempfile
from pathlib import Path

import pytest

from memory.database import Database
from memory.manager import MemoryManager
from tools.notes import build_tools as notes_tools
from tools.permissions import PermissionLevel
from tools.registry import Tool, ToolContext, ToolRegistry
from tools.shell import build_tools as shell_tools


def make_context(tmp_path):
    context = ToolContext()
    context.memory = MemoryManager(Database(Path(tmp_path) / "mem.db"))
    return context


def test_registry_validation_rejects_bad_args():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="echo",
            description="echo",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=lambda text, _ctx: {"success": True, "output": text},
        )
    )
    result = registry.execute("echo", {"text": 123})
    assert result.success is False
    assert "string" in result.error
    result = registry.execute("echo", {})
    assert result.success is False
    assert "Missing required" in result.error


def test_unknown_tool_returns_error():
    registry = ToolRegistry()
    result = registry.execute("nope", {})
    assert result.success is False
    assert "not registered" in result.error


def test_confirmation_denied(tmp_path):
    context = make_context(tmp_path)
    denied = []
    context.confirm = lambda prompt: False
    registry = ToolRegistry(context)
    registry.register(
        Tool(
            name="sensitive",
            description="sensitive",
            parameters={"type": "object", "properties": {}},
            handler=lambda _ctx: {"success": True, "output": "ran"},
            permission=PermissionLevel.ASK,
        )
    )
    result = registry.execute("sensitive", {})
    assert result.success is False
    assert "denied" in result.error


def test_confirmation_approved(tmp_path):
    context = make_context(tmp_path)
    context.confirm = lambda prompt: True
    registry = ToolRegistry(context)
    registry.register(
        Tool(
            name="sensitive",
            description="sensitive",
            parameters={"type": "object", "properties": {}},
            handler=lambda _ctx: {"success": True, "output": "ran"},
            permission=PermissionLevel.ASK,
        )
    )
    result = registry.execute("sensitive", {})
    assert result.success is True


def test_tool_exception_reported(tmp_path):
    registry = ToolRegistry(make_context(tmp_path))
    registry.register(
        Tool(
            name="boom",
            description="boom",
            parameters={"type": "object", "properties": {}},
            handler=lambda _ctx: (_ for _ in ()).throw(RuntimeError("kaboom")),
        )
    )
    result = registry.execute("boom", {})
    assert result.success is False
    assert "kaboom" in result.error


def test_shell_tool_blocked_by_default(tmp_path):
    registry = ToolRegistry(make_context(tmp_path))
    # shell tools are only registered when enabled
    assert shell_tools(enabled=False) == []


def test_shell_blocklist(tmp_path):
    context = make_context(tmp_path)
    context.confirm = lambda prompt: True  # shell requires confirmation each time
    registry = ToolRegistry(context)
    for tool in shell_tools(enabled=True):
        registry.register(tool)
    result = registry.execute("run_shell_command", {"command": "format c:"})
    assert result.success is False
    assert "blocked" in result.output.lower()


def test_shell_requires_confirmation(tmp_path):
    context = make_context(tmp_path)
    registry = ToolRegistry(context)
    for tool in shell_tools(enabled=True):
        registry.register(tool)
    result = registry.execute("run_shell_command", {"command": "echo hi"})
    assert result.success is False
    assert "denied" in result.error


def test_notes_via_memory(tmp_path):
    registry = ToolRegistry(make_context(tmp_path))
    for tool in notes_tools():
        registry.register(tool)
    created = registry.execute("create_note", {"title": "Groceries", "content": "milk, eggs"})
    assert created.success is True
    note_id = created.data["note_id"]
    found = registry.execute("search_notes", {"query": "milk"})
    assert found.success is True
    assert found.data["notes"]
    updated = registry.execute("update_note", {"note_id": note_id, "content": "milk only"})
    assert updated.success is True
    deleted = registry.execute("delete_note", {"note_id": note_id})
    assert deleted.success is True


def test_ollama_schema_shape():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="get_current_time",
            description="time",
            parameters={"type": "object", "properties": {}},
            handler=lambda _ctx: {"success": True},
        )
    )
    schema = registry.schemas()[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "get_current_time"
