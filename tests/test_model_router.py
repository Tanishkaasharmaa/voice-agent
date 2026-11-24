import pytest

from ollama.model_router import ModelRouter

AGENT = "qwen3.5:4b"
REASONING = "qwen3.5:9b"


@pytest.fixture
def router():
    return ModelRouter(AGENT, REASONING, mode="auto")


def test_routine_command_uses_4b(router):
    assert router.route("open notepad") == AGENT
    assert router.route("what time is it") == AGENT


def test_complex_reasoning_uses_9b(router):
    text = ("Please think deeply about whether we should migrate this database "
            "and analyze all the trade-offs, considering performance, cost, "
            "and reliability, and explain in detail how you would plan it out.")
    assert router.route(text) == REASONING


def test_explicit_deep_request_uses_9b(router):
    assert router.route("analyze the pros and cons of this plan in depth") == REASONING


def test_4b_only_mode(router):
    router.set_mode("4b_only")
    assert router.route("analyze this in depth, think carefully and weigh every option") == AGENT


def test_9b_only_mode(router):
    router.set_mode("9b_only")
    assert router.route("open notepad") == REASONING


def test_short_simple_stays_on_4b(router):
    assert router.route("hi") == AGENT


def test_invalid_mode_rejected(router):
    with pytest.raises(ValueError):
        router.set_mode("gpu")


def test_classify(router):
    assert router.classify("open notepad") == "routine (4B)"
    assert router.classify("analyze this in depth considering all trade-offs") == "complex (9B)"
