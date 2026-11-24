"""OllamaClient — thin, isolated wrapper around the local Ollama HTTP API.

Default endpoint http://127.0.0.1:11434. All Ollama-specific knowledge lives
here; nothing else in the app talks to Ollama directly.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import requests


class OllamaError(Exception):
    """Raised for transport or API errors. ``status`` may be None."""

    def __init__(self, message: str, status: int | None = None, timeout: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.timeout = timeout


class OllamaClient:
    def __init__(self, host: str = "http://127.0.0.1:11434", timeout: float = 120.0) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------- utilities

    def _url(self, path: str) -> str:
        return f"{self.host}{path}"

    def _post(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        try:
            resp = self.session.post(
                self._url(path), data=json.dumps(payload), timeout=timeout or self.timeout
            )
        except requests.exceptions.ReadTimeout as exc:
            raise OllamaError(
                f"Ollama timed out at {self.host} after {timeout or self.timeout}s: {exc}",
                timeout=True,
            ) from exc
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama unreachable at {self.host}: {exc}") from exc
        if resp.status_code >= 400:
            raise OllamaError(f"Ollama API error ({resp.status_code}): {resp.text[:300]}", resp.status_code)
        try:
            return resp.json()
        except ValueError as exc:
            raise OllamaError("Ollama returned an invalid response") from exc

    def _get(self, path: str, timeout: float | None = None) -> dict[str, Any]:
        try:
            resp = self.session.get(self._url(path), timeout=timeout or self.timeout)
        except requests.exceptions.ReadTimeout as exc:
            raise OllamaError(
                f"Ollama timed out at {self.host} after {timeout or self.timeout}s: {exc}",
                timeout=True,
            ) from exc
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama unreachable at {self.host}: {exc}") from exc
        if resp.status_code >= 400:
            raise OllamaError(f"Ollama API error ({resp.status_code}): {resp.text[:300]}", resp.status_code)
        try:
            return resp.json()
        except ValueError as exc:
            raise OllamaError("Ollama returned an invalid response") from exc

    # -------------------------------------------------------------- checking

    def ping(self) -> bool:
        try:
            resp = self.session.get(self._url("/api/version"), timeout=3.0)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[dict[str, Any]]:
        # /api/tags is a GET endpoint — POSTing to it returns 405.
        data = self._get("/api/tags")
        return data.get("models", [])

    def model_installed(self, name: str) -> bool:
        """True when the model is present. Tolerant of tag differences so the
        setup screen detects models the user pulled with a different tag:
        requesting "qwen3.5" matches "qwen3.5:9b", and an install listed as
        "qwen3.5:latest" satisfies a "qwen3.5:9b" request."""
        installed = {m.get("name") for m in self.list_models()}
        if name in installed:
            return True
        req_base, _, req_tag = name.partition(":")
        for m in installed:
            base, _, tag = m.partition(":")
            if base != req_base:
                continue
            if not req_tag or tag == req_tag or tag == "latest":
                return True
        return False

    def model_info(self, name: str) -> dict[str, Any]:
        """Details for a model (size, family, parameters, ...)."""
        try:
            return self._post("/api/show", {"model": name})
        except OllamaError:
            return {}

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
        """Stream /api/chat responses (newline-delimited JSON, one per line)."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if options:
            payload["options"] = options
        try:
            resp = self.session.post(
                self._url("/api/chat"),
                data=json.dumps(payload),
                timeout=timeout or self.timeout,
                stream=True,
            )
        except requests.exceptions.ReadTimeout as exc:
            raise OllamaError(
                f"Ollama timed out at {self.host} after {timeout or self.timeout}s: {exc}",
                timeout=True,
            ) from exc
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama unreachable at {self.host}: {exc}") from exc
        if resp.status_code >= 400:
            raise OllamaError(
                f"Ollama chat error ({resp.status_code}): {resp.text[:300]}", resp.status_code
            )
        # decode_unicode=True only decodes when the response declares an
        # encoding; Ollama's streaming responses often don't (no charset in
        # Content-Type), so lines can still arrive as bytes. Decode them
        # explicitly instead of relying on that flag.
        for raw in resp.iter_lines():
            if not raw:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            line = raw.strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        keep_alive: str | int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Non-streaming chat. Returns the full assistant message dict."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if options:
            payload["options"] = options
        return self._post("/api/chat", payload)

    # -------------------------------------------------------------- cloud

    def probe_cloud(self, model: str, timeout: float = 8.0) -> str:
        """Check whether a cloud model is usable (sign-in + reachability).

        A 1-token chat with ``keep_alive=0`` keeps the probe cheap, and an
        unsigned-in user gets a fast 401 back before any inference runs.
        Returns ``"ok"`` (200), ``"unauthorized"`` (401), or ``"error"``
        (connection failure or any other status).
        """
        try:
            resp = self.session.post(
                self._url("/api/chat"),
                data=json.dumps({
                    "model": model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "keep_alive": 0,
                    "options": {"num_predict": 1},
                }),
                timeout=timeout,
            )
        except requests.RequestException:
            return "error"
        if resp.status_code == 200:
            return "ok"
        if resp.status_code == 401:
            return "unauthorized"
        return "error"

    # -------------------------------------------------------------- memory

    def unload_model(self, name: str) -> None:
        """Request Ollama to unload the model after the (empty) call.

        keep_alive=0 is Ollama's documented way to release a model
        immediately after a request completes.
        """
        try:
            self._post(
                "/api/chat",
                {"model": name, "messages": [], "keep_alive": 0},
                timeout=30.0,
            )
        except OllamaError:
            pass

    def generate_with_keep_alive(self, name: str, keep_alive: str | int) -> None:
        """Nudge residency: a lightweight /api/generate ping with the desired
        keep_alive so the model is warm for the next real request."""
        try:
            self._post(
                "/api/generate",
                {"model": name, "prompt": "", "keep_alive": keep_alive},
                timeout=30.0,
            )
        except OllamaError:
            pass
