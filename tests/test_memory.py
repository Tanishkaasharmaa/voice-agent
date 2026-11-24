from memory.database import Database
from memory.manager import MemoryManager


def make_manager(tmp_path):
    db = Database(tmp_path / "test.db")
    return MemoryManager(db)


def test_conversation_and_messages(tmp_path):
    memory = make_manager(tmp_path)
    conv = memory.new_conversation("Test")
    memory.add_message("user", "open notepad")
    memory.add_message("assistant", "Done.", tool_name="open_application", tool_input={"name": "notepad"})
    messages = memory.get_conversation_messages(conv)
    assert len(messages) == 2
    # Ordered newest-first (id DESC).
    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_name"] == "open_application"
    assert messages[1]["role"] == "user"


def test_save_and_search_memories(tmp_path):
    memory = make_manager(tmp_path)
    memory.save_memory("User prefers dark mode", kind="memory", tags=["preference"])
    memory.save_memory("Coffee order: latte", kind="memory")
    results = memory.search_memory("latte")
    assert len(results) == 1
    assert "Coffee" in results[0]["content"]


def test_delete_memory(tmp_path):
    memory = make_manager(tmp_path)
    mid = memory.save_memory("temporary fact")
    assert memory.delete_memory(mid) is True
    assert memory.search_memory("temporary") == []


def test_relevant_memories(tmp_path):
    memory = make_manager(tmp_path)
    memory.save_memory("User works at Acme Corp", kind="memory")
    memory.save_memory("Favorite color is blue", kind="memory")
    relevant = memory.get_relevant_memories("tell me about Acme", limit=5)
    assert any("Acme" in r["content"] for r in relevant)


def test_notes_are_kind_notes(tmp_path):
    memory = make_manager(tmp_path)
    memory.save_memory("Buy milk", kind="note")
    notes = memory.list_memories(kind="note")
    assert len(notes) == 1
    assert memory.list_memories(kind="memory") == []


def test_settings_and_install_history(tmp_path):
    memory = make_manager(tmp_path)
    memory.set_setting("theme", "dark")
    assert memory.get_setting("theme") == "dark"
    memory.record_install("Ollama", "installed", version="0.5.0")
    memory.record_install("Ollama", "missing")
    # Upsert keeps one row, latest status.
    rows = memory.db.query("SELECT * FROM install_history WHERE dependency_name = 'Ollama'")
    assert len(rows) == 1
    assert rows[0]["installation_status"] == "missing"


def test_multithreaded_writes(tmp_path):
    import threading

    memory = make_manager(tmp_path)
    errors = []

    def writer(n):
        try:
            for i in range(20):
                memory.save_memory(f"fact {n}-{i}")
                memory.add_message("user", f"hello {n}-{i}")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(memory.list_memories(limit=500)) == 80


def test_close_all_releases_connections(tmp_path):
    db = Database(tmp_path / "test.db")
    conn = db.connection()
    assert len(db._conns) == 1
    db.close_all()
    assert len(db._conns) == 0
    # Reopening after close_all still works (lazy reconnect).
    assert db.scalar("SELECT 1") == 1
