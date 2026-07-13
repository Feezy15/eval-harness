"""OpenAI provider wrapper.

Fails loud at construction (missing key) rather than at the first episode:
a keyless matrix run should die immediately, not burn through several
episodes before the first live call discovers the misconfiguration.
"""

import os
import time
from collections.abc import Sequence

from openai import OpenAI

from collab_eval.models.base import AgentModel
from collab_eval.models.pricing import cost_usd
from collab_eval.types import Message, ModelResponse, Usage


class OpenAIModel(AgentModel):
    def __init__(self, model: str, seed: int, temperature: float, max_tokens: int | None = None):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set; OpenAIModel must not be constructed without it"
            )
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = f"openai:{model}"
        # Built once, here: construction must never make a network call, so an
        # invalid/missing key surfaces at the first real request, not silently.
        self._client = OpenAI()

    def next_turn(self, conversation: Sequence[Message]) -> ModelResponse:
        messages = [{"role": m.role, "content": m.content} for m in conversation]

        # `max_tokens` is deprecated on current chat-completions models in
        # favor of `max_completion_tokens`; only send it when the config sets
        # a cap, rather than inventing a default ceiling the config didn't ask for.
        extra = {"max_completion_tokens": self.max_tokens} if self.max_tokens is not None else {}

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            seed=self.seed,
            **extra,
        )
        latency_s = time.perf_counter() - start

        content = response.choices[0].message.content
        if not content:
            # A silent empty assistant turn would corrupt the episode (the
            # next agent/user-sim turn would be replying to nothing).
            raise ValueError(f"OpenAI returned an empty assistant turn (model={self.model!r})")

        usage = Usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            cost_usd=cost_usd(
                "openai",
                self.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            ),
            latency_s=latency_s,
        )
        return ModelResponse(message=Message(role="assistant", content=content), usage=usage)
