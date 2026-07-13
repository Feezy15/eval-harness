"""MockModel + shared types.

The mock is not just for avoiding API cost: it must be *deterministic* (so CI can
assert on outputs) and must emit *nonzero synthetic usage* (so the cost/latency
pipeline — this project's differentiator — is exercised end-to-end without keys).
"""

from collab_eval.models.mock import MockModel
from conftest import conv


def _conv() -> list:
    return conv("You are a helpful planner.", "Plan a weekend trip to Lisbon.")


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
