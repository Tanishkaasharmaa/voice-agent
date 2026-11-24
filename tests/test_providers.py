"""Tests for the multi-provider LLM layer (OpenAI/Groq/OpenRouter)."""

import json

import pytest

from app.config import Settings
from ollama.client import OllamaClient, OllamaError
from ollama.providers import PROVIDERS, OpenAICompatClient, build_client


# ------------------------------------------------------------- registry


def test_provider_registry_has_required_providers():
    for pid in ("ollama", "openai", "groq", "openrouter"):
        assert pid in PROVIDERS
    assert PROVIDERS["openai"]["needs_api_key"] is True
    assert PROVIDERS["ollama"]["needs_api_key"] is False


def test_build_client_selects_implementation(tmp_path):
    settings = Settings(path=tmp_path / "config.json")
    assert isinstance(build_client("ollama", settings), OllamaClient)
    client = build_client("openai", settings)
    assert isinstance(client, OpenAICompatClient)
    assert client.base_url == "https://api.openai.com/v1"


def test_build_client_rejects_unknown_provider():
    with pytest.raises(ValueError):
        OpenAICompatClient("doesnotexist")


def test_unknown_provider_raises_value_error():
    with pytest.raises(ValueError):
        OpenAICompatClient("nope")


# ------------------------------------------------------------ encoding


def test_encode_messages_maps_tool_role():
    encoded = OpenAICompatClient._encode_messages([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "notes.save",
                                      "arguments": {"text": "x"}}}]},
        {"role": "tool", "name": "notes.save", "content": '{"ok": true}'},
    ])
    assert encoded[0] == {"role": "system", "content": "sys"}
    tc = encoded[2]["tool_calls"][0]
    assert tc["function"]["name"] == "notes.save"
    assert json.loads(tc["function"]["arguments"]) == {"text": "x"}
    assert encoded[3]["role"] == "tool"
    assert encoded[3]["tool_call_id"] == "notes.save"


def test_encode_tool_wraps_schema():
    schema = {"function": {"name": "t", "description": "d",
                           "parameters": {"type": "object"}}}
    wrapped = OpenAICompatClient._encode_tool(schema)
    assert wrapped == {"type": "function", "function": {
        "name": "t", "description": "d",
        "parameters": {"type": "object"}}}


# -------------------------------------------------------------- stream


class _FakeResponse:
    def __init__(self, lines):
        self._lines = lines
        self.status_code = 200
        self.text = ""

    def iter_lines(self):
        return iter(self._lines)


class _FakeSession:
    """Returns canned SSE lines for POST; captures the payload."""

    def __init__(self, lines):
        self._lines = lines
        self.payload = None

    def post(self, url, data=None, timeout=None, stream=False):
        self.payload = json.loads(data)
        return _FakeResponse(self._lines)


def _sse(obj) -> str:
    return "data: " + json.dumps(obj)


def test_chat_stream_converts_content_deltas():
    lines = [
        _sse({"choices": [{"delta": {"content": "Hello"}}]}),
        _sse({"choices": [{"delta": {"content": " there."}}]}),
        _sse({"choices": [{"finish_reason": "stop", "delta": {}}]}),
        "data: [DONE]",
    ]
    client = OpenAICompatClient("groq", api_key="k")
    client.session = _FakeSession(lines)
    chunks = list(client.chat_stream("m", [{"role": "user", "content": "hi"}]))
    text = "".join(c["message"]["content"] for c in chunks)
    assert text == "Hello there."
    assert all("tool_calls" not in c["message"] for c in chunks)


def test_chat_stream_accumulates_tool_calls_into_ollama_shape():
    lines = [
        _sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "notes.", "arguments": ""}}]}}]}),
        _sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "save",
                                      "arguments": '{"text"'}}]}}]}),
        _sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": ': "hi"}'}}]}}]}),
        _sse({"choices": [{"finish_reason": "tool_calls", "delta": {}}]}),
        "data: [DONE]",
    ]
    client = OpenAICompatClient("openrouter", api_key="k")
    client.session = _FakeSession(lines)
    chunks = list(client.chat_stream("m", [{"role": "user", "content": "hi"}]))
    tool_chunks = [c for c in chunks if c["message"].get("tool_calls")]
    assert len(tool_chunks) == 1
    call = tool_chunks[0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "notes.save"
    assert call["function"]["arguments"] == {"text": "hi"}


def test_chat_stream_sends_openai_payload():
    lines = [_sse({"choices": [{"finish_reason": "stop", "delta": {}}]}),
             "data: [DONE]"]
    session = _FakeSession(lines)
    client = OpenAICompatClient("openai", api_key="secret")
    client.session = session
    list(client.chat_stream(
        "gpt-4o-mini",
        [{"role": "user", "content": "hi"}],
        tools=[{"function": {"name": "t", "description": "", "parameters": {}}}],
        options={"temperature": 0.3},
    ))
    payload = session.payload
    assert payload["model"] == "gpt-4o-mini"
    assert payload["stream"] is True
    assert payload["temperature"] == 0.3
    assert payload["tools"][0]["type"] == "function"
    # The real session carries the API key header (the fake replaced it,
    # so build a fresh client to verify the header wiring).
    fresh = OpenAICompatClient("openai", api_key="secret")
    assert fresh.session.headers["Authorization"] == "Bearer secret"


def test_http_error_maps_to_ollama_error_with_status():
    class _ErrResp:
        status_code = 401

        def iter_lines(self):
            return iter([])

        @property
        def text(self):
            return "unauthorized"

    class _ErrSession:
        headers: dict = {}

        def post(self, *a, **k):
            return _ErrResp()

    client = OpenAICompatClient("openai", api_key="bad")
    client.session = _ErrSession()
    with pytest.raises(OllamaError) as exc:
        list(client.chat_stream("m", [{"role": "user", "content": "hi"}]))
    assert exc.value.status == 401


# ------------------------------------------------------ config properties


def test_settings_connectivity_and_provider_defaults(tmp_path):
    s = Settings(path=tmp_path / "config.json")
    assert s.connectivity_mode == "hybrid"
    assert s.provider == "ollama"
    assert s.local_model_required is True


def test_settings_connectivity_validated(tmp_path):
    s = Settings(path=tmp_path / "config.json")
    s.set("llm.connectivity", "bogus")
    assert s.connectivity_mode == "hybrid"
    s.set("llm.connectivity", "offline")
    assert s.connectivity_mode == "offline"


def test_local_model_required_by_mode_and_provider(tmp_path):
    s = Settings(path=tmp_path / "config.json")

    s.set("llm.provider", "ollama")
    s.set("llm.connectivity", "online")
    assert s.local_model_required is False  # pure online: cloud only
    s.set("llm.connectivity", "offline")
    assert s.local_model_required is True
    s.set("llm.connectivity", "hybrid")
    assert s.local_model_required is True

    s.set("llm.provider", "openai")
    assert s.local_model_required is True   # hybrid falls back to local ollama
    s.set("llm.connectivity", "online")
    assert s.local_model_required is False  # hosted provider, online-only
    s.set("llm.connectivity", "offline")
    assert s.local_model_required is False


def test_api_key_roundtrip(tmp_path):
    s = Settings(path=tmp_path / "config.json")
    assert s.api_key("groq") == ""
    s.set_api_key("groq", " gsk_test_123 ")
    assert s.api_key("groq") == "gsk_test_123"
