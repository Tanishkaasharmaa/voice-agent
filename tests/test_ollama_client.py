"""OllamaClient chat_stream tests — offline, no Ollama server required."""

from __future__ import annotations

from ollama.client import OllamaClient


class _FakeResp:
    """Minimal stand-in for requests.Response used by chat_stream."""

    def __init__(self, lines) -> None:
        self._lines = list(lines)
        self.status_code = 200

    def iter_lines(self, *args, **kwargs):
        return iter(self._lines)


def _client_with(lines):
    client = OllamaClient(host="http://127.0.0.1:11434")
    resp = _FakeResp(lines)
    client.session.post = lambda *a, **k: resp
    return client


def test_chat_stream_accepts_bytes_lines():
    # Ollama's streaming /api/chat responses often have no charset in the
    # Content-Type, so requests yields raw bytes even with decode_unicode.
    client = _client_with([
        b'{"message": {"content": "Hello"}}',
        b'{"done": true}',
    ])
    chunks = list(client.chat_stream("m", [{"role": "user", "content": "hi"}]))
    assert chunks[0]["message"]["content"] == "Hello"
    assert chunks[1]["done"] is True


def test_chat_stream_accepts_str_lines():
    client = _client_with([
        '{"message": {"content": "Hi"}}',
        '{"done": true}',
    ])
    chunks = list(client.chat_stream("m", [{"role": "user", "content": "hi"}]))
    assert chunks[0]["message"]["content"] == "Hi"


def test_chat_stream_strips_data_prefix():
    client = _client_with([b'data: {"done": true}'])
    chunks = list(client.chat_stream("m", [{"role": "user", "content": "hi"}]))
    assert chunks[0]["done"] is True


def test_list_models_uses_get(monkeypatch):
    # /api/tags is a GET endpoint; POSTing returns 405 and made every
    # startup check report the Ollama models as missing.
    calls: dict = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"models": [{"name": "qwen3.5:4b"}]}

    client = OllamaClient(host="http://127.0.0.1:11434")
    monkeypatch.setattr(client.session, "get", lambda url, timeout=None: (calls.update(url=url), FakeResp())[1])

    assert client.list_models() == [{"name": "qwen3.5:4b"}]
    assert calls["url"].endswith("/api/tags")
