"""AgentRuntime — the local agent loop.

Transcript in -> (route model) -> chat loop -> tools -> final response.
The final text turn is streamed sentence-by-sentence so TTS can begin before
the model finishes. Tool results are never spoken directly.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from app.events import EventBus, TOPIC_AGENT_STATUS
from ollama.client import OllamaClient, OllamaError
from ollama.model_router import ModelRouter
from ollama.resource_manager import LLMResourceManager
from response.processor import ResponseProcessor
from tools.registry import ToolRegistry, ToolResult
from utils.logging import get_logger

log = get_logger("agent")

_MAX_TOOL_ITERATIONS = 5
_SENTENCE_END = re.compile(r".*?[.!?…](?=\s|$)")


class AgentRuntime:
    def __init__(
        self,
        client: OllamaClient,
        router: ModelRouter,
        resources: LLMResourceManager,
        registry: ToolRegistry,
        processor: ResponseProcessor,
        memory=None,
        bus: EventBus | None = None,
        settings=None,
    ) -> None:
        self.client = client
        self.router = router
        self.resources = resources
        self.registry = registry
        self.processor = processor
        self.memory = memory
        self.bus = bus
        self.settings = settings
        self._pending = ""
        # Set once a cloud model returns 401 Unauthorized, so later
        # utterances skip straight to the local fallback instead of failing
        # on the cloud model every time.
        self._cloud_unavailable = False

    # ---------------------------------------------------------------- prompt

    def build_system_prompt(self) -> str:
        agent_name = self.settings.agent_name if self.settings else "Luna"
        user_name = self.settings.user_name if self.settings else ""
        rules = [
            "- Never invent tool results. If a tool failed, say it failed.",
            "- Never claim an action succeeded unless the tool returned success.",
            "- Never expose internal reasoning, JSON, or tool metadata to the user.",
            "- Ask for confirmation before destructive actions.",
            "- Be concise and conversational, like a real assistant: 'Done.', "
            "'It's open.', 'I couldn't find that.'",
            "- Use tools when necessary; do not use them unnecessarily.",
            "- Remember important user information only through the memory tools.",
        ]
        if user_name:
            rules.insert(1, f"- The user's name is {user_name} — use it naturally when addressing them.")
        return (
            f"You are {agent_name}, a local Windows voice assistant running fully offline. "
            "You have access only to the tools explicitly provided in this conversation. "
            "Rules:\n" + "\n".join(rules)
        )

    def _memory_context(self, transcript: str) -> str:
        if self.memory is None:
            return ""
        limit = int(self.settings.get("memory.max_context_memories", 5)) if self.settings else 5
        rows = self.memory.get_relevant_memories(transcript, limit=limit)
        if not rows:
            return ""
        lines = [f"- {r['content'][:300]}" for r in rows]
        return "Relevant saved memories:\n" + "\n".join(lines)

    # ------------------------------------------------------------------ run

    def run(
        self,
        transcript: str,
        on_sentence: Callable[[str], None] | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        """Process a transcript and return the cleaned final response text.

        ``on_sentence`` is called with each speech-ready sentence as soon as
        it completes (for TTS streaming). ``on_chunk`` receives raw text
        deltas (for the transcript UI).
        """
        tool_names = self.registry.tool_names()
        model = self._resolve_model(transcript, tool_names)
        model = self.resources.choose_model(model)
        self._status(f"routing to {model}")
        log.info("[MODEL THINKING] %s", model)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.build_system_prompt()},
        ]
        memory_context = self._memory_context(transcript)
        if memory_context:
            messages.append({"role": "system", "content": memory_context})
        messages.append({"role": "user", "content": transcript})

        options = None
        if self.settings is not None:
            options = {
                "temperature": float(self.settings.get("llm.temperature", 0.2)),
                "num_ctx": int(self.settings.get("llm.context_size", 4096)),
            }

        final_text = ""
        self._pending = ""
        for _ in range(_MAX_TOOL_ITERATIONS):
            self.resources.prepare(model)
            collected_text = ""
            collected_thinking = ""
            tool_calls: list[dict[str, Any]] = []
            try:
                for chunk in self.client.chat_stream(
                    model, messages, tools=self.registry.schemas(),
                    keep_alive=self.resources.keep_alive, options=options,
                ):
                    msg = chunk.get("message", {})
                    delta = msg.get("content") or ""
                    if delta:
                        collected_text += delta
                        if on_chunk:
                            on_chunk(delta)
                        self._buffer_sentences(delta, on_sentence)
                    # Reasoning models stream internal chain-of-thought in
                    # ``thinking``. It is never spoken or exposed to the user,
                    # but we track it so a thought-only reply is not reported
                    # as a silent "Done.".
                    thinking = msg.get("thinking") or ""
                    if thinking and len(collected_thinking) < 8192:
                        collected_thinking += thinking
                    tool_calls.extend(msg.get("tool_calls") or [])
            except OllamaError as exc:
                # A cloud model that is not authorized (401) must not break
                # every utterance: retry this turn with the local fallback
                # and remember the failure for subsequent utterances.
                if exc.status == 401 and self._is_cloud_model(model):
                    fallback = self._fallback_local_model()
                    if not fallback or fallback == model:
                        raise
                    self._cloud_unavailable = True
                    self._notify_cloud_fallback(model, fallback)
                    log.warning("[CLOUD FALLBACK] %s unauthorized (401) — using %s",
                                model, fallback)
                    model = fallback
                    continue
                # Timeout errors indicate the model is too slow (likely too
                # large for the hardware). Surface a clear, actionable
                # message instead of a raw traceback.
                if exc.timeout:
                    log.warning("[TIMEOUT] %s timed out for model %s", self.client.host, model)
                    raise OllamaError(
                        f"The model {model} took too long to respond. "
                        "This usually means the model is too large for your hardware. "
                        "Try a smaller model in Settings (e.g. qwen3.5:2b or qwen3.5:4b), "
                        "or increase the LLM timeout.",
                        timeout=True,
                    ) from exc
                raise

            messages.append(
                {"role": "assistant", "content": collected_text,
                 **( {"tool_calls": tool_calls} if tool_calls else {} )}
            )

            if not tool_calls:
                final_text = collected_text
                break

            # Execute tools, append results, loop.
            self._status("executing tools")
            log.info("[CALLING TOOLS] %s", ", ".join(
                c.get("function", {}).get("name", "?") for c in tool_calls
            ))
            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                try:
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args) if args.strip() else {}
                    if not isinstance(args, dict):
                        args = {}
                except (ValueError, TypeError):
                    args = {}
                result: ToolResult = self.registry.execute(name, args)
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(result.to_dict()),
                        "name": name,
                    }
                )
            self._status("thinking")

        final_text = final_text.strip()
        if not final_text:
            final_text = self._brief_from_last_result(
                had_thinking=bool(collected_thinking.strip())
            )
        log.info("[MODEL RESPONSE] %s", final_text)
        if on_sentence:
            # Flush only the buffered tail — complete sentences were already
            # streamed during generation (avoids re-speaking them).
            tail = self._pending
            self._pending = ""
            if tail.strip():
                for sentence in self.processor.split_sentences(tail):
                    if sentence and self.processor.is_spoken_content(sentence):
                        on_sentence(sentence)
        return final_text

    # ---------------------------------------------------------------- helpers

    def _is_cloud_model(self, model: str) -> bool:
        """Cloud models carry a ``:cloud`` (or ``-cloud``) tag."""
        return bool(model and model.endswith("cloud"))

    def _fallback_local_model(self) -> str | None:
        """The local model to use when the cloud model is unavailable.

        None when this setup has no usable local model: pure online mode,
        or a hosted OpenAI-compatible provider (the single client cannot
        switch to Ollama mid-turn)."""
        if self.settings is None:
            return None
        if getattr(self.settings, "connectivity_mode", "hybrid") == "online":
            return None
        if getattr(self.settings, "provider", "ollama") != "ollama":
            return None
        return self.settings.local_model or None

    def _resolve_model(self, transcript: str, tool_names: list[str]) -> str:
        """Route normally, but skip the cloud model once it has proven
        unauthorized (401) so later utterances go straight to the local
        fallback instead of failing on the cloud model again."""
        if self._cloud_unavailable:
            fallback = self._fallback_local_model()
            if fallback:
                return fallback
        return self.router.route(transcript, tool_names)

    def reset_cloud_fallback(self) -> None:
        """Re-allow the cloud model (e.g. after the user signs in to Ollama
        or changes model settings)."""
        self._cloud_unavailable = False

    def _notify_cloud_fallback(self, cloud_model: str, local_model: str) -> None:
        if self.bus:
            self.bus.publish(TOPIC_AGENT_STATUS, {
                "status": "cloud_unavailable",
                "detail": (f"The cloud model {cloud_model} is not authorized (401). "
                           f"Switched to the local model {local_model}. "
                           "Run 'ollama signin' and save Settings to use cloud models again."),
            })

    def _buffer_sentences(self, delta: str, on_sentence: Callable[[str], None] | None) -> None:
        """Emit completed sentences from the stream buffer as they close."""
        if on_sentence is None:
            return
        self._pending += delta
        while True:
            match = _SENTENCE_END.match(self._pending)
            # Hold the sentence until we see the whitespace that confirms it
            # is complete (the tail is flushed at the end of run()).
            if not match or match.end() >= len(self._pending):
                break
            sentence = match.group(0).strip()
            if sentence and self.processor.is_spoken_content(sentence):
                on_sentence(sentence)
            self._pending = self._pending[match.end():]

    def _brief_from_last_result(self, had_thinking: bool = False) -> str:
        """Spoken fallback when the model produced no ``content`` at all.

        ``had_thinking`` distinguishes a model that reasoned but never
        formed a spoken reply (say so honestly) from one that produced
        nothing whatsoever ("Done.")."""
        if had_thinking:
            return "I couldn't put a reply into words. Could you try asking again?"
        return "Done."

    def _status(self, status: str) -> None:
        if self.bus:
            self.bus.publish(TOPIC_AGENT_STATUS, {"status": status})
