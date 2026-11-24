"""LLM providers — Ollama plus OpenAI-compatible cloud APIs.

``OpenAICompatClient`` speaks the OpenAI chat-completions wire format
(OpenAI, Groq, OpenRouter all use it) but yields **Ollama-style** stream
chunks (``{"message": {"content": ...}}``) so ``AgentRuntime`` and the rest
of the app work with either client unchanged. Tool calls are accumulated
across deltas and emitted as one final synthetic chunk in Ollama's shape.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import requests

from ollama.client import OllamaError


# Provider id -> metadata used by the wizard and Settings UI.
PROVIDERS: dict[str, dict[str, Any]] = {
    "ollama": {
        "label": "Ollama (local + Ollama Cloud)",
        "base_url": "",  # local server / ollama.com cloud — handled by OllamaClient
        "models": [],
        "needs_api_key": False,
        "online_only": False,
    },
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            "openrouter/auto",
            "anthropic/claude-sonnet-4.5",
            "openai/gpt-4o-mini",
            "google/gemini-2.5-flash",
            "meta-llama/llama-3.3-70b-instruct",
            "deepseek/deepseek-chat",
        ],
        "needs_api_key": True,
        "online_only": True,
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3-32b",
        ],
        "needs_api_key": True,
        "online_only": True,
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4.1",
            "o4-mini",
        ],
        "needs_api_key": True,
        "online_only": True,
    },
}


class OpenAICompatClient:
    """Minimal OpenAI chat-completions client with an Ollama-like surface.

    Implements exactly the parts of ``OllamaClient`` the app uses:
    ``ping``, ``list_models``, ``model_installed``, ``chat_stream``,
    ``unload_model``, ``generate_with_keep_alive`` (no-ops), and
    ``probe_cloud``. Errors raise ``OllamaError`` so existing handling
    (401 fallback, timeouts) keeps working.
    """

    def __init__(self, provider: str, api_key: str = "",
                 timeout: float = 120.0) -> None:
        meta = PROVIDERS.get(provider)
        if not meta or not meta["base_url"]:
            raise ValueError(f"Unknown OpenAI-compatible provider: {provider}")
        self.provider = provider
        self.base_url = meta["base_url"].rstrip("/")
        self.api_key = api_key.strip()
        self.timeout = timeout
        self.host = self.base_url  # log-message compatibility
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        })

    # ------------------------------------------------------------- checking

    def ping(self) -> bool:
        try:
            resp = self.session.get(f"{self.base_url}/models", timeout=5.0)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[dict[str, Any]]:
        try:
            resp = self.session.get(f"{self.base_url}/models", timeout=self.timeout)
        except requests.RequestException as exc:
            raise OllamaError(f"{self.provider} unreachable: {exc}") from exc
        if resp.status_code >= 400:
            raise OllamaError(
                f"{self.provider} API error ({resp.status_code}): {resp.text[:300]}",
                resp.status_code)
        try:
            return resp.json().get("data", [])
        except ValueError as exc:
            raise OllamaError("Invalid response from provider") from exc

    def model_installed(self, name: str) -> bool:
        # Cloud models are always "installed"; auth problems surface at
        # request time as a 401 and are handled by the agent's fallback.
        return True

    def probe_cloud(self, model: str, timeout: float = 8.0) -> str:
        return "ok"

    # ----------------------------------------------------------------- chat

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        keep_alive: str | int | None = None,
        options: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._encode_messages(messages),
            "stream": True,
        }
        if tools:
            payload["tools"] = [self._encode_tool(t) for t in tools]
        temperature = (options or {}).get("temperature")
        if temperature is not None:
            payload["temperature"] = temperature
        yield from self._stream(payload, timeout or self.timeout)

    # ------------------------------------------------------------- internals

    @staticmethod
    def _encode_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")
            if role == "tool":
                out.append({"role": "tool", "tool_call_id": msg.get("name", ""),
                            "content": msg.get("content", "")})
            elif role == "assistant" and msg.get("tool_calls"):
                out.append({
                    "role": "assistant",
                    "content": msg.get("content") or None,
                    "tool_calls": [
                        {"id": c.get("function", {}).get("name", f"call_{i}"),
                         "type": "function",
                         "function": {"name": c["function"]["name"],
                                      "arguments": json.dumps(
                                          c["function"].get("arguments", {}))}}
                        for i, c in enumerate(msg["tool_calls"])
                    ],
                })
            else:
                out.append({"role": role, "content": msg.get("content", "")})
        return out

    @staticmethod
    def _encode_tool(schema: dict[str, Any]) -> dict[str, Any]:
        fn = schema.get("function", schema)
        return {"type": "function", "function": {
            "name": fn.get("name"),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        }}

    def _stream(self, payload: dict[str, Any], timeout: float):
        try:
            resp = self.session.post(f"{self.base_url}/chat/completions",
                                     data=json.dumps(payload), timeout=timeout,
                                     stream=True)
        except requests.exceptions.ReadTimeout as exc:
            raise OllamaError(
                f"{self.provider} timed out after {timeout}s: {exc}",
                timeout=True) from exc
        except requests.RequestException as exc:
            raise OllamaError(f"{self.provider} unreachable: {exc}") from exc
        if resp.status_code >= 400:
            raise OllamaError(
                f"{self.provider} API error ({resp.status_code}): {resp.text[:300]}",
                resp.status_code)

        tool_acc: dict[int, dict[str, Any]] = {}
        finish_reason = None
        for raw in resp.iter_lines():
            if not raw:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            line = raw.strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line or line == "[DONE]":
                continue
            try:
                data = json.loads(line)
            except ValueError:
                continue
            choices = data.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            content = delta.get("content") or ""
            thinking = delta.get("reasoning") or ""
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                acc = tool_acc.setdefault(idx, {"function": {"name": "",
                                                             "arguments": ""}})
                fn = tc.get("function") or {}
                if fn.get("id"):
                    acc["id"] = fn["id"]
                if fn.get("name"):
                    acc["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    acc["function"]["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            if content or thinking:
                chunk: dict[str, Any] = {"message": {"content": content}}
                if thinking:
                    chunk["message"]["thinking"] = thinking
                yield chunk
        if tool_acc:
            calls = []
            for idx in sorted(tool_acc):
                acc = tool_acc[idx]
                args = acc["function"].get("arguments") or "{}"
                try:
                    parsed = json.loads(args) if args.strip() else {}
                except ValueError:
                    parsed = {}
                calls.append({"function": {"name": acc["function"]["name"],
                                           "arguments": parsed}})
            yield {"message": {"content": "", "tool_calls": calls}}
        if finish_reason == "tool_calls" and not tool_acc:
            raise OllamaError(
                f"{self.provider}: model requested tool calls but none were "
                "parsed from the stream")

    # -------------------------------------------------------------- memory

    def unload_model(self, name: str) -> None:
        pass  # nothing to unload on a hosted API

    def generate_with_keep_alive(self, name: str, keep_alive: str | int) -> None:
        pass


def build_client(provider: str, settings=None):
    """Create the right chat client for ``provider``.

    Ollama keeps using the local-server client; everything else talks to the
    provider's OpenAI-compatible endpoint with the stored API key.
    """
    host = settings.llm_host if settings else "http://127.0.0.1:11434"
    timeout = settings.llm_timeout if settings else 120.0
    if provider == "ollama":
        from ollama.client import OllamaClient

        return OllamaClient(host, timeout=timeout)
    api_key = settings.api_key(provider) if settings else ""
    return OpenAICompatClient(provider, api_key=api_key, timeout=timeout)
