"""Deterministic mock model: zero API cost, synthetic-but-nonzero usage.

Determinism means CI can assert on exact outputs; synthetic usage numbers mean
the token/cost/latency pipeline is exercised without keys. Content and usage are
pure functions of (model name, seed, conversation) — no hidden state, so the
same episode replays identically.
"""

import hashlib
from collections.abc import Sequence

from collab_eval.models.base import AgentModel
from collab_eval.types import Message, ModelResponse, Usage

# Fake pricing in the shape of real per-token pricing, so cost totals look and
# aggregate like the real thing (dollars per million tokens).
_FAKE_USD_PER_MTOK_INPUT = 0.15
_FAKE_USD_PER_MTOK_OUTPUT = 0.60
_CHARS_PER_TOKEN = 4  # crude but standard heuristic for synthetic token counts


class MockModel(AgentModel):
    def __init__(self, model: str, seed: int, temperature: float = 0.0):
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.name = f"mock:{model}"

    def next_turn(self, conversation: Sequence[Message]) -> ModelResponse:
        digest = self._digest(conversation)
        last = conversation[-1].content if conversation else ""
        content = (
            f"[{self.name} seed={self.seed} {digest[:8]}] "
            f"Deterministic reply to: {last[:60]!r} "
            f"(turn {sum(1 for m in conversation if m.role == 'assistant') + 1})"
        )

        input_tokens = max(1, sum(len(m.content) for m in conversation) // _CHARS_PER_TOKEN)
        output_tokens = max(1, len(content) // _CHARS_PER_TOKEN)
        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=(
                input_tokens * _FAKE_USD_PER_MTOK_INPUT
                + output_tokens * _FAKE_USD_PER_MTOK_OUTPUT
            )
            / 1e6,
            # Deterministic pseudo-latency in [0.01, 0.11) — varies per call like the
            # real thing, but stable across replays.
            latency_s=0.01 + (int(digest[:4], 16) / 0xFFFF) * 0.1,
        )
        return ModelResponse(message=Message(role="assistant", content=content), usage=usage)

    def _digest(self, conversation: Sequence[Message]) -> str:
        h = hashlib.sha256(f"{self.name}|{self.seed}|{self.temperature}".encode())
        for m in conversation:
            h.update(f"|{m.role}:{m.content}".encode())
        return h.hexdigest()