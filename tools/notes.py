"""Notes tools — create / search / update / delete notes via the memory DB."""

from __future__ import annotations

from tools.registry import ToolContext, Tool


def _create_note(title: str, content: str = "", _ctx: ToolContext = None) -> dict:
    ctx: ToolContext = _ctx
    if ctx.memory is None:
        return {"success": False, "output": "Memory system unavailable"}
    body = f"{title}\n{content}".strip() if content else title
    note_id = ctx.memory.save_memory(body, kind="note")
    return {"success": True, "output": "Note saved.", "data": {"note_id": note_id}}


def _search_notes(query: str, _ctx: ToolContext = None) -> dict:
    ctx: ToolContext = _ctx
    if ctx.memory is None:
        return {"success": False, "output": "Memory system unavailable"}
    rows = ctx.memory.list_memories(kind="note", limit=200)
    query = query.lower()
    matches = [r for r in rows if query in r["content"].lower() or query in (r["tags"] or "")]
    if not matches:
        return {"success": True, "output": "No notes found.", "data": {"notes": []}}
    previews = [{"id": r["id"], "text": r["content"][:200]} for r in matches[:5]]
    return {"success": True, "output": f"Found {len(matches)} note(s).", "data": {"notes": previews}}


def _update_note(note_id: int, content: str, _ctx: ToolContext = None) -> dict:
    ctx: ToolContext = _ctx
    if ctx.memory is None:
        return {"success": False, "output": "Memory system unavailable"}
    ok = ctx.memory.update_memory(note_id, content)
    return {"success": ok, "output": "Note updated." if ok else "Note not found."}


def _delete_note(note_id: int, _ctx: ToolContext = None) -> dict:
    ctx: ToolContext = _ctx
    if ctx.memory is None:
        return {"success": False, "output": "Memory system unavailable"}
    ok = ctx.memory.delete_memory(note_id)
    return {"success": ok, "output": "Note deleted." if ok else "Note not found."}


def build_tools() -> list:
    return [
        Tool(
            name="create_note",
            description="Save a note locally. Use for things the user wants to keep.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Note title"},
                    "content": {"type": "string", "description": "Note body (optional)"},
                },
                "required": ["title"],
            },
            handler=_create_note,
        ),
        Tool(
            name="search_notes",
            description="Search saved notes by keyword.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=_search_notes,
        ),
        Tool(
            name="update_note",
            description="Replace the content of a saved note by id.",
            parameters={
                "type": "object",
                "properties": {
                    "note_id": {"type": "integer"},
                    "content": {"type": "string"},
                },
                "required": ["note_id", "content"],
            },
            handler=_update_note,
        ),
        Tool(
            name="delete_note",
            description="Delete a saved note by id.",
            parameters={
                "type": "object",
                "properties": {"note_id": {"type": "integer"}},
                "required": ["note_id"],
            },
            handler=_delete_note,
        ),
    ]
