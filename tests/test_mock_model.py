"""MockModel + shared types.

The mock is not just for avoiding API cost: it must be *deterministic* (so CI can
assert on outputs) and must emit *nonzero synthetic usage* (so the cost/latency
pipeline — this project's differentiator — is exercised end-to-end without keys).
"""

from collab_eval.models.base import AgentModel
from collab_eval.models.mock import MockModel
from collab_eval.types import Message, Usage


def _conv() -> list[Message]:
    return [
        Message(role="system", content="You are a helpful planner."),
        Message(role="user", content="Plan a weekend trip to Lisbon."),
    ]


def test_mock_is_an_agent_model():
    assert isinstance(MockModel(model="mock-agent", seed=0), AgentModel)


def test_same_seed_same_conversation_is_deterministic():
    a = MockModel(model="mock-agent", seed=0)
    b = MockModel(model="mock-agent", seed=0)
    ra, rb = a.next_turn(_conv()), b.next_turn(_conv())
    assert ra == rb


def test_different_seed_changes_output():
    a = MockModel(model="mock-agent", seed=0)
    b = MockModel(model="mock-agent", seed=1)
    assert a.next_turn(_conv()).message.content != b.next_turn(_conv()).message.content


def test_response_shape_and_synthetic_usage():
    resp = MockModel(model="mock-agent", seed=0).next_turn(_conv())
    assert resp.message.role == "assistant"
    assert resp.message.content  # non-empty
    u = resp.usage
    assert u.input_tokens > 0
    assert u.output_tokens > 0
    assert u.cost_usd > 0
    assert u.latency_s > 0


def test_usage_sums():
    u1 = Usage(input_tokens=10, output_tokens=5, cost_usd=0.001, latency_s=0.5)
    u2 = Usage(input_tokens=20, output_tokens=15, cost_usd=0.002, latency_s=1.5)
    total = sum([u1, u2], Usage.zero())
    assert total == Usage(input_tokens=30, output_tokens=20, cost_usd=0.003, latency_s=2.0)
