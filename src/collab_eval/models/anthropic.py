"""Anthropic provider wrapper.

Fails loud at construction: missing key, or a missing `max_tokens` (unlike
OpenAI's chat-completions endpoint, the Messages API has no server-side
default and rejects a request without it — better to catch that at config
build time than mid-matrix on the first live call).
"""

import os
import time
from collections.abc import Sequence

from anthropic import Anthropic

from collab_eval.models.base import AgentModel
from collab_eval.models.pricing import cost_usd
from collab_eval.types import Message, ModelResponse, Usage


class AnthropicModel(AgentModel):
    def __init__(self, model: str, seed: int, temperature: float, max_tokens: int | None = None):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; AnthropicModel must not be constructed without it"
            )
        if max_tokens is None:
            raise ValueError(
                "AnthropicModel requires max_tokens; set it on the model entry in config"
            )
        self.model = model
        # Anthropic's API has no seed parameter (sampling can't be truly
        # seeded remotely); this is kept only so it can feed the cache key.
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"
        # Built once, here: construction must never make a network call.
        self._client = Anthropic()

    def next_turn(self, conversation: Sequence[Message]) -> ModelResponse:
        # The Messages API takes system content as a separate top-level
        # param, not as a message with role "system".
        system_parts = [m.content for m in conversation if m.role == "system"]
        system = "\n\n".join(system_parts) if system_parts else None
        messages = [
            {"role": m.role, "content": m.content} for m in conversation if m.role != "system"
        ]

        extra = {"system": system} if system is not None else {}

        start = time.perf_counter()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages,
            **extra,
        )
        latency_s = time.perf_counter() - start

        content = "".join(block.text for block in response.content if block.type == "text")
        if not content:
            # A silent empty assistant turn would corrupt the episode (the
            # next agent/user-sim turn would be replying to nothing).
            raise ValueError(f"Anthropic returned an empty assistant turn (model={self.model!r})")

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=cost_usd(
                "anthropic",
                self.model,
                response.usage.input_tokens,
                response.usage.output_tokens,
            ),
            latency_s=latency_s,
        )
        return ModelResponse(message=Message(role="assistant", content=content), usage=usage)
