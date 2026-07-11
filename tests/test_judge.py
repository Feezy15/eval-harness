"""LLMJudge: parses a real judge model's checklist response into a JudgeScore,
and build_judge: resolves a JudgeConfig into the right Judge implementation.

LLMJudge is exercised over a stub AgentModel (no network) that returns canned
JSON, so these tests pin the parsing contract — met-fraction scoring,
code-fence tolerance, and fail-loud on anything that isn't a well-formed
checklist response — independent of any real provider.
"""

import json

import pytest

from collab_eval.config import JudgeConfig
from collab_eval.judge import LLMJudge, MockJudge, build_judge
from collab_eval.models import MODEL_REGISTRY
from collab_eval.models.base import AgentModel
from collab_eval.models.cache import CachedModel, ResponseCache
from collab_eval.tasks.toy import ToyTask
from collab_eval.types import Message, ModelResponse, Usage


class _StubJudgeModel(AgentModel):
    """Returns scripted response contents in order (the last repeats), and
    records every conversation it receives — LLMJudge's parsing and
    repair-retry contract is what's under test, not any real model's output."""

    name = "stub:judge"
    model = "stub-judge-model"
    temperature = 0.0

    def __init__(self, *contents: str):
        self._contents = contents
        self.calls: list[list[Message]] = []

    def next_turn(self, conversation):
        self.calls.append(list(conversation))
        content = self._contents[min(len(self.calls), len(self._contents)) - 1]
        return ModelResponse(
            message=Message(role="assistant", content=content),
            usage=Usage(input_tokens=5, output_tokens=7, cost_usd=1e-5, latency_s=0.02),
        )


class _FakeProviderModel(AgentModel):
    """Stand-in for a real provider's AgentModel: same constructor shape as
    MockModel/OpenAIModel/AnthropicModel, but no network and no API key —
    build_judge's non-mock branch just needs something MODEL_REGISTRY-resolvable."""

    def __init__(self, model: str, seed: int, temperature: float, max_tokens: int | None = None):
        self.model = model
        self.temperature = temperature
        self.name = f"fake:{model}"

    def next_turn(self, conversation):
        return ModelResponse(
            message=Message(role="assistant", content=_canned([True, True, True])),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=1e-6, latency_s=0.01),
        )


def _canned(met_flags: list[bool]) -> str:
    return json.dumps(
        {
            "criteria": [
                {"index": i, "reasoning": f"reason {i}", "met": met}
                for i, met in enumerate(met_flags, start=1)
            ]
        }
    )


# ToyTask.judge_criteria has 3 items — every canned response below matches that.


def test_llm_judge_scores_met_fraction_and_carries_usage():
    task = ToyTask()
    judge = LLMJudge(model=_StubJudgeModel(_canned([True, True, False])), rubric_version="v1")

    result = judge.score(task, [Message(role="assistant", content="final plan")], seed=0)

    assert result.score == pytest.approx(2 / 3)
    assert result.rubric_version == "v1"
    assert result.usage == Usage(input_tokens=5, output_tokens=7, cost_usd=1e-5, latency_s=0.02)
    assert judge.model == "stub-judge-model"


def test_llm_judge_tolerates_a_markdown_code_fence():
    wrapped = f"```json\n{_canned([True, True, True])}\n```"
    judge = LLMJudge(model=_StubJudgeModel(wrapped), rubric_version="v1")

    result = judge.score(ToyTask(), [Message(role="assistant", content="x")], seed=0)

    assert result.score == 1.0


@pytest.mark.parametrize(
    "content",
    [
        "this is not json",
        json.dumps({"criteria": [{"index": 1, "reasoning": "only one", "met": True}]}),
        json.dumps(
            {
                "criteria": [
                    {"index": 1, "reasoning": "", "met": True},
                    {"index": 2, "reasoning": "y", "met": True},
                    {"index": 3, "reasoning": "z", "met": True},
                ]
            }
        ),
        json.dumps(
            {
                "criteria": [
                    {"index": 1, "reasoning": "x", "met": "yes"},
                    {"index": 2, "reasoning": "y", "met": True},
                    {"index": 3, "reasoning": "z", "met": True},
                ]
            }
        ),
    ],
    ids=["not_json", "wrong_criteria_count", "empty_reasoning", "met_not_boolean"],
)
def test_llm_judge_fails_loud_on_malformed_output(content):
    model = _StubJudgeModel(content)
    judge = LLMJudge(model=model, rubric_version="v1", max_repair_attempts=1)

    with pytest.raises(ValueError):
        judge.score(ToyTask(), [Message(role="assistant", content="x")], seed=0)

    # The bounded repair attempt was spent before giving up: persistent
    # malformation exhausts the retry budget, then stays fail-loud.
    assert len(model.calls) == 2


def test_llm_judge_repairs_a_malformed_response():
    bad = "this is not json"
    model = _StubJudgeModel(bad, _canned([True, False, True]))
    judge = LLMJudge(model=model, rubric_version="v1", max_repair_attempts=1)

    result = judge.score(ToyTask(), [Message(role="assistant", content="x")], seed=0)

    assert result.score == pytest.approx(2 / 3)
    assert result.usage == Usage(input_tokens=10, output_tokens=14, cost_usd=2e-5, latency_s=0.04)
    # The repair call extends the same conversation: the malformed response
    # goes back as the assistant turn it was, and the corrective user message
    # carries the parse error so the model can see what to fix.
    assert len(model.calls) == 2
    first, repair = model.calls
    assert repair[: len(first)] == first
    assert repair[len(first)] == Message(role="assistant", content=bad)
    assert repair[-1].role == "user"
    assert "not valid JSON" in repair[-1].content


def test_llm_judge_rejects_transcript_not_ending_with_assistant():
    # The judge scores transcript[-1] as the consolidation artifact; a
    # transcript ending on a user message means that contract is broken
    # upstream, and scoring the user's words as the agent's plan would be a
    # silent measurement error.
    judge = LLMJudge(model=_StubJudgeModel(_canned([True, True, True])), rubric_version="v1")

    with pytest.raises(ValueError, match="assistant"):
        judge.score(ToyTask(), [Message(role="user", content="not the artifact")], seed=0)


# --- build_judge ---------------------------------------------------------------


def test_build_judge_mock_provider_returns_mock_judge():
    cfg = JudgeConfig(provider="mock", model="mock-judge", rubric_version="v0")
    assert isinstance(build_judge(cfg, cache=None), MockJudge)


def test_build_judge_real_provider_returns_llm_judge_with_cached_backing_model(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(MODEL_REGISTRY, "fake", _FakeProviderModel)
    cfg = JudgeConfig(
        provider="fake", model="fake-model", rubric_version="v1", max_repair_attempts=2
    )

    uncached = build_judge(cfg, cache=None)
    assert isinstance(uncached, LLMJudge)
    assert uncached.model == "fake-model"
    assert uncached.max_repair_attempts == 2
    assert not isinstance(uncached._model, CachedModel)

    # Same cfg, cache enabled: the backing model must come back wrapped —
    # this is what closes the "wire the cache through the judge" gap.
    cached = build_judge(cfg, cache=ResponseCache(tmp_path / "cache"))
    assert isinstance(cached._model, CachedModel)
