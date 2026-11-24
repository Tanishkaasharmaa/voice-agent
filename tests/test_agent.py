import json

from app.events import EventBus
from ollama.agent import AgentRuntime
from ollama.client import OllamaError
from ollama.model_router import ModelRouter
from ollama.resource_manager import LLMResourceManager
from response.processor import ResponseProcessor
from tools.registry import Tool, ToolContext, ToolRegistry


class FakeClient:
    def __init__(self, script):
        self.script = script  # list of chunk lists per chat_stream call
        self.chat_calls = 0
        self.unloaded = []
        self.warmed = []
        self.installed = {"qwen3.5:4b", "qwen3.5:9b"}

    def chat_stream(self, model, messages, tools=None, keep_alive=None, options=None):
        chunks = self.script[self.chat_calls]
        self.chat_calls += 1
        return iter(chunks)

    def unload_model(self, name):
        self.unloaded.append(name)

    def generate_with_keep_alive(self, name, keep_alive):
        self.warmed.append(name)

    def model_installed(self, name):
        return name in self.installed

    def model_info(self, name):
        return {}


def make_agent(script):
    bus = EventBus()
    client = FakeClient(script)
    resources = LLMResourceManager(client, "qwen3.5:4b", "qwen3.5:9b",
                                   keep_alive="10m", bus=bus, ram_provider=lambda: 12.0)
    context = ToolContext()
    registry = ToolRegistry(context)
    registry.register(
        Tool(
            name="open_application",
            description="open an app",
            parameters={"type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"]},
            handler=lambda name, _ctx: {"success": True, "output": "opened"},
        )
    )
    return AgentRuntime(client, ModelRouter("qwen3.5:4b", "qwen3.5:9b"),
                        resources, registry, ResponseProcessor(), bus=bus)


class FakeSettings:
    def __init__(self, agent="Luna", user="", local_model="qwen3.5:4b"):
        self._agent = agent
        self._user = user
        self._local_model = local_model

    @property
    def agent_name(self):
        return self._agent

    @property
    def user_name(self):
        return self._user

    @property
    def local_model(self):
        return self._local_model

    def get(self, key, default=None):
        return default


def test_system_prompt_uses_agent_and_user_names():
    agent = make_agent([[]])
    agent.settings = FakeSettings(agent="Jarvis", user="Alex")
    prompt = agent.build_system_prompt()
    assert "You are Jarvis" in prompt
    assert "Alex" in prompt


def test_system_prompt_without_user_name():
    agent = make_agent([[]])
    agent.settings = FakeSettings(agent="Luna", user="")
    prompt = agent.build_system_prompt()
    assert "You are Luna" in prompt
    assert "user's name is" not in prompt


def test_cloud_401_falls_back_to_local_model():
    class AuthFailingClient(FakeClient):
        def __init__(self, script):
            super().__init__(script)
            self.cloud_attempts = 0

        def chat_stream(self, model, messages, tools=None, keep_alive=None, options=None):
            if model.endswith("cloud"):
                self.cloud_attempts += 1
                raise OllamaError('Ollama chat error (401): {"error":"Unauthorized"}', 401)
            return super().chat_stream(model, messages, tools=tools,
                                       keep_alive=keep_alive, options=options)

    bus = EventBus()
    script = [[{"message": {"content": "Done."}}], [{"message": {"content": "ok"}}]]
    client = AuthFailingClient(script)
    resources = LLMResourceManager(client, "deepseek-v4-flash:cloud", "deepseek-v4-flash:cloud",
                                   keep_alive="10m", bus=bus, ram_provider=lambda: 12.0)
    registry = ToolRegistry(ToolContext())
    agent = AgentRuntime(
        client, ModelRouter("deepseek-v4-flash:cloud", "deepseek-v4-flash:cloud"),
        resources, registry, ResponseProcessor(), bus=bus,
        settings=FakeSettings(agent="Luna", user="", local_model="qwen3.5:4b"),
    )

    # First utterance: cloud 401 -> retried on the local model.
    assert agent.run("hi") == "Done."
    assert client.cloud_attempts == 1
    assert agent._cloud_unavailable is True

    # Second utterance: goes straight to the local model, no new 401.
    assert agent.run("again") == "ok"
    assert client.cloud_attempts == 1


def test_plain_response_streams_sentences():
    script = [[
        {"message": {"content": "Sure. "}},
        {"message": {"content": "I'll open Chrome for you."}},
        {"message": {"content": ""}},
    ]]
    agent = make_agent(script)
    sentences = []
    result = agent.run("open chrome", on_sentence=sentences.append)
    assert result == "Sure. I'll open Chrome for you."
    assert sentences == ["Sure.", "I'll open Chrome for you."]


def test_tool_call_loop():
    script = [
        # First turn: tool call, no content.
        [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "function": {"name": "open_application", "arguments": {"name": "notepad"}},
                }],
            }
        }],
        # Second turn: final text.
        [{"message": {"content": "Done. It's open."}}],
    ]
    agent = make_agent(script)
    result = agent.run("open notepad")
    assert "open" in result.lower()
    # Two chat turns: the tool call, then the final response.
    assert agent.client.chat_calls == 2


def test_tool_args_string_arguments():
    script = [
        [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "function": {"name": "open_application",
                                 "arguments": json.dumps({"name": "calc"})},
                }],
            }
        }],
        [{"message": {"content": "Done."}}],
    ]
    agent = make_agent(script)
    result = agent.run("open calculator")
    assert agent.client.chat_calls == 2
    assert "Done" in result


def test_empty_final_response_returns_brief():
    script = [
        [{"message": {"content": ""}}],
    ]
    agent = make_agent(script)
    result = agent.run("hello")
    assert result == "Done."


def test_thinking_only_reply_is_not_silent_done():
    # A reasoning model that streams only ``thinking`` (no content) must not
    # fall back to a silent "Done." — and its reasoning must never be spoken.
    script = [[
        {"message": {"content": "", "thinking": "The user asked for help. "
                                                     "I should call a tool but no "
                                                     "tool is needed. "}},
        {"message": {"content": "", "thinking": "Still reasoning... "}},
        {"message": {"content": ""}},
    ]]
    agent = make_agent(script)
    sentences = []
    result = agent.run("help me", on_sentence=sentences.append)
    assert result != "Done."
    assert "reply" in result.lower()
    # Reasoning is tracked internally but never spoken or returned.
    assert sentences == []
    assert "tool is needed" not in result


def test_mixed_thinking_and_content_speaks_content_only():
    script = [[
        {"message": {"content": "", "thinking": "internal reasoning here "}},
        {"message": {"content": "Sure. ", "thinking": "more reasoning "}},
        {"message": {"content": "I'll do that."}},
    ]]
    agent = make_agent(script)
    sentences = []
    result = agent.run("do the thing", on_sentence=sentences.append)
    assert result == "Sure. I'll do that."
    assert sentences == ["Sure.", "I'll do that."]
    assert "reasoning" not in result
