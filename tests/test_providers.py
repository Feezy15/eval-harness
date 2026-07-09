"""OpenAI and Anthropic wrappers: construction guardrails, request mapping,
response parsing, and registry membership.

No network, no real keys: every SDK client is monkeypatched onto an already-
constructed wrapper (construction itself never calls the network), fed a fake
env key so the construction-time key check passes.
"""

import pytest

from collab_eval.judge import JUDGE_REGISTRY
from collab_eval.models import MODEL_REGISTRY
from collab_eval.models.anthropic import AnthropicModel
from collab_eval.models.openai import OpenAIModel
from collab_eval.models.pricing import cost_usd
from collab_eval.types import Message

# --- fake OpenAI SDK surface -----------------------------------------------------


class _FakeOpenAIUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeOpenAIMessage:
    def __init__(self, content: str | None):
        self.content = content


class _FakeOpenAIChoice:
    def __init__(self, content: str | None):
        self.message = _FakeOpenAIMessage(content)


class _FakeOpenAIResponse:
    def __init__(self, content: str | None, prompt_tokens: int = 100, completion_tokens: int = 50):
        self.choices = [_FakeOpenAIChoice(content)]
        self.usage = _FakeOpenAIUsage(prompt_tokens, completion_tokens)


class _FakeCompletions:
    def __init__(self, response):
        self._response = response
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeChat:
    def __init__(self, response):
        self.completions = _FakeCompletions(response)


class _FakeOpenAIClient:
    def __init__(self, response):
        self.chat = _FakeChat(response)


# --- fake Anthropic SDK surface ---------------------------------------------------


class _FakeAnthropicUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeTextBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _FakeNonTextBlock:
    type = "tool_use"


class _FakeAnthropicResponse:
    def __init__(self, blocks: list, input_tokens: int = 100, output_tokens: int = 50):
        self.content = blocks
        self.usage = _FakeAnthropicUsage(input_tokens, output_tokens)


class _FakeMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


# --- construction guardrails: missing key / never touches network, both providers -


_PROVIDER_CASES = {
    "openai": (OpenAIModel, "OPENAI_API_KEY", {"model": "gpt-5.4-mini"}),
    "anthropic": (
        AnthropicModel,
        "ANTHROPIC_API_KEY",
        {"model": "claude-haiku-4-5", "max_tokens": 64},
    ),
}


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_missing_api_key_raises_at_construction(monkeypatch, provider):
    model_cls, env_var, extra_kwargs = _PROVIDER_CASES[provider]
    monkeypatch.delenv(env_var, raising=False)
    with pytest.raises(Exception, match=env_var):
        model_cls(seed=0, temperature=0.0, **extra_kwargs)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_construction_never_touches_the_network(monkeypatch, provider):
    model_cls, env_var, extra_kwargs = _PROVIDER_CASES[provider]
    monkeypatch.setenv(env_var, "fake-key")
    # If this reached the network it would hang/error in a sandboxed test run.
    model_cls(seed=0, temperature=0.0, **extra_kwargs)


# --- OpenAIModel: request/response mapping ----------------------------------------


def test_openai_next_turn_maps_request_and_parses_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    model = OpenAIModel(model="gpt-5.4-mini", seed=7, temperature=0.3, max_tokens=256)
    fake_response = _FakeOpenAIResponse("hello there", prompt_tokens=1000, completion_tokens=500)
    fake_client = _FakeOpenAIClient(fake_response)
    model._client = fake_client

    conv = [Message(role="system", content="sys"), Message(role="user", content="hi")]
    result = model.next_turn(conv)

    kwargs = fake_client.chat.completions.last_kwargs
    assert kwargs["model"] == "gpt-5.4-mini"
    assert kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert kwargs["temperature"] == 0.3
    assert kwargs["seed"] == 7
    assert kwargs["max_completion_tokens"] == 256

    assert result.message.role == "assistant"
    assert result.message.content == "hello there"
    assert result.usage.input_tokens == 1000
    assert result.usage.output_tokens == 500
    assert result.usage.cost_usd == pytest.approx(cost_usd("openai", "gpt-5.4-mini", 1000, 500))
    assert result.usage.latency_s > 0


def test_openai_omits_max_completion_tokens_when_max_tokens_unset(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    model = OpenAIModel(model="gpt-5.4-mini", seed=0, temperature=0.0)
    fake_client = _FakeOpenAIClient(_FakeOpenAIResponse("hi"))
    model._client = fake_client

    model.next_turn([Message(role="user", content="hi")])

    assert "max_completion_tokens" not in fake_client.chat.completions.last_kwargs


@pytest.mark.parametrize("content", [None, ""])
def test_openai_none_or_empty_content_raises(monkeypatch, content):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    model = OpenAIModel(model="gpt-5.4-mini", seed=0, temperature=0.0)
    model._client = _FakeOpenAIClient(_FakeOpenAIResponse(content))
    with pytest.raises(ValueError):
        model.next_turn([Message(role="user", content="hi")])


# --- AnthropicModel: construction guardrails --------------------------------------


def test_anthropic_missing_max_tokens_raises_at_construction(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    with pytest.raises(Exception, match="max_tokens"):
        AnthropicModel(model="claude-haiku-4-5", seed=0, temperature=0.0)


# --- AnthropicModel: request/response mapping -------------------------------------


def test_anthropic_next_turn_extracts_system_and_maps_usage(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    model = AnthropicModel(model="claude-haiku-4-5", seed=3, temperature=0.2, max_tokens=256)
    fake_response = _FakeAnthropicResponse(
        [_FakeTextBlock("hello "), _FakeTextBlock("world")], input_tokens=800, output_tokens=200
    )
    fake_client = _FakeAnthropicClient(fake_response)
    model._client = fake_client

    conv = [
        Message(role="system", content="sys one"),
        Message(role="system", content="sys two"),
        Message(role="user", content="hi"),
    ]
    result = model.next_turn(conv)

    kwargs = fake_client.messages.last_kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["system"] == "sys one\n\nsys two"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert kwargs["max_tokens"] == 256
    assert kwargs["temperature"] == 0.2
    # System content must never leak into the messages list.
    assert all(m["role"] != "system" for m in kwargs["messages"])

    assert result.message.content == "hello world"  # multiple text blocks joined
    assert result.usage.input_tokens == 800
    assert result.usage.output_tokens == 200
    assert result.usage.cost_usd == pytest.approx(
        cost_usd("anthropic", "claude-haiku-4-5", 800, 200)
    )
    assert result.usage.latency_s > 0


def test_anthropic_empty_content_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    model = AnthropicModel(model="claude-haiku-4-5", seed=0, temperature=0.0, max_tokens=64)
    model._client = _FakeAnthropicClient(_FakeAnthropicResponse([]))
    with pytest.raises(ValueError):
        model.next_turn([Message(role="user", content="hi")])


def test_anthropic_non_text_blocks_are_ignored_not_concatenated(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    model = AnthropicModel(model="claude-haiku-4-5", seed=0, temperature=0.0, max_tokens=64)
    model._client = _FakeAnthropicClient(
        _FakeAnthropicResponse([_FakeNonTextBlock(), _FakeTextBlock("only this")])
    )
    result = model.next_turn([Message(role="user", content="hi")])
    assert result.message.content == "only this"


# --- registry ----------------------------------------------------------------------


def test_openai_and_anthropic_registered(monkeypatch):
    assert MODEL_REGISTRY["openai"] is OpenAIModel
    assert MODEL_REGISTRY["anthropic"] is AnthropicModel

    # Every registered model/judge exposes a non-empty raw model string: the
    # GenAI semconv span attribute (gen_ai.request.model) names the model
    # actually called, distinct from the harness's provider-qualified label.
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    for model_cls in MODEL_REGISTRY.values():
        agent = model_cls(model="m", seed=0, temperature=0.0, max_tokens=64)
        assert isinstance(agent.model, str) and agent.model
    for judge_cls in JUDGE_REGISTRY.values():
        judge = judge_cls(model="m", rubric_version="v")
        assert isinstance(judge.model, str) and judge.model
