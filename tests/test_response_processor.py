from response.processor import ResponseProcessor


def test_strips_markdown_and_code():
    processor = ResponseProcessor()
    text = "**Done.** I opened `notepad` for you.\n```json\n{\"ok\": true}\n```"
    cleaned = processor.clean(text)
    assert "**" not in cleaned
    assert "```" not in cleaned
    assert "json" not in cleaned.lower() or "opened" in cleaned
    assert "notepad" in cleaned  # inline code content is spoken, backticks dropped


def test_never_speaks_tool_json():
    processor = ResponseProcessor()
    raw = '{"tool": "open_application", "success": true}'
    assert not processor.is_spoken_content(raw)
    cleaned = processor.clean(raw)
    assert cleaned == ""


def test_sentence_chunking():
    processor = ResponseProcessor()
    text = "Sure. I'll open Chrome for you. Then I'll search the web."
    chunks = processor.split_sentences(text)
    assert chunks == ["Sure.", "I'll open Chrome for you.", "Then I'll search the web."]


def test_long_sentence_split_at_clause():
    processor = ResponseProcessor(max_chunk_chars=60)
    text = "First I will open the application, then I will wait for it to load, and finally I will check the window."
    chunks = processor.split_sentences(text)
    assert len(chunks) >= 2
    assert all(len(c) <= 60 for c in chunks)


def test_brief_responses():
    processor = ResponseProcessor()
    assert processor.brief("open_application", True) == "It's open."
    assert processor.brief("save_memory", True) == "I'll remember that."
    assert processor.brief("open_application", False) == "I couldn't open that."
    assert processor.brief("unknown_tool", True) == "Done."


def test_empty_input():
    processor = ResponseProcessor()
    assert processor.clean("") == ""
    assert processor.split_sentences("") == []
