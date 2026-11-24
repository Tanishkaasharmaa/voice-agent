from ollama.client import OllamaError
from ollama.resource_manager import LLMResourceManager

AGENT = "qwen3.5:4b"
REASONING = "qwen3.5:9b"


class FakeClient:
    def __init__(self):
        self.unloaded = []
        self.warmed = []
        self.keep_alive_calls = []

    def unload_model(self, name):
        self.unloaded.append(name)

    def generate_with_keep_alive(self, name, keep_alive):
        self.warmed.append(name)
        self.keep_alive_calls.append((name, keep_alive))


def make_manager(ram_provider):
    return LLMResourceManager(
        client=FakeClient(),
        agent_model=AGENT,
        reasoning_model=REASONING,
        keep_alive="10m",
        ram_provider=ram_provider,
    )


def test_prepare_warms_agent():
    resources = make_manager(ram_provider=lambda: 12.0)
    resources.prepare(AGENT)
    assert resources.resident_models == [AGENT]
    assert resources.client.warmed == [AGENT]


def test_prepare_9b_unloads_4b_first():
    resources = make_manager(ram_provider=lambda: 12.0)
    resources.prepare(AGENT)
    resources.prepare(REASONING)
    assert AGENT in resources.client.unloaded
    assert REASONING in resources.client.warmed
    assert resources.resident_models == [REASONING]


def test_low_ram_prevents_9b():
    resources = make_manager(ram_provider=lambda: 4.0)
    resources.prepare(AGENT)
    assert resources.safe_for_9b() is False
    # choose_model falls back to the agent model
    assert resources.choose_model(REASONING) == AGENT


def test_on_resource_snapshot_downgrades():
    resources = make_manager(ram_provider=lambda: 12.0)
    resources.prepare(REASONING)
    result = resources.on_resource_snapshot(2.0)
    assert result == AGENT
    assert REASONING in resources.client.unloaded
    assert resources.resident_models == []


def test_expire_idle_9b_unloads_but_agent_stays_warm():
    resources = make_manager(ram_provider=lambda: 12.0)
    resources.prepare(AGENT)
    resources.expire_idle_models(max_idle_s=0.0)
    # The agent model is never expired — it stays warm.
    assert AGENT in resources.resident_models
    assert not resources.client.unloaded

    resources.prepare(REASONING)
    resources.expire_idle_models(max_idle_s=0.0)
    # The idle 9B is unloaded; a later prepare(AGENT) re-warms it.
    assert REASONING in resources.client.unloaded
    resources.prepare(AGENT)
    assert AGENT in resources.resident_models


def test_single_model_setup_does_not_unload_rewarm():
    # agent == reasoning (single-model setup): prepare() must not unload and
    # re-warm the same model on every call.
    resources = LLMResourceManager(
        client=FakeClient(), agent_model="qwen3.5:4b",
        reasoning_model="qwen3.5:4b", keep_alive="10m", ram_provider=lambda: 12.0,
    )
    resources.prepare("qwen3.5:4b")
    resources.prepare("qwen3.5:4b")
    assert resources.client.unloaded == []
    assert resources.client.warmed == ["qwen3.5:4b", "qwen3.5:4b"]


def test_shutdown_unloads_everything():
    resources = make_manager(ram_provider=lambda: 12.0)
    resources.prepare(AGENT)
    resources.prepare(REASONING)
    resources.shutdown()
    assert resources.resident_models == []
    assert set(resources.client.unloaded) == {AGENT, REASONING}
