"""Model registry: maps config `provider` strings to AgentModel classes.

Registered classes share the constructor signature
``(model: str, seed: int, temperature: float, max_tokens: int | None = None)``
so the runner can build any provider from a ModelConfig uniformly. Adding a
provider = one module + one line here.
"""

from collab_eval.models.anthropic import AnthropicModel
from collab_eval.models.base import AgentModel
from collab_eval.models.mock import MockModel
from collab_eval.models.openai import OpenAIModel

MODEL_REGISTRY: dict[str, type[AgentModel]] = {
    "mock": MockModel,
    "openai": OpenAIModel,
    "anthropic": AnthropicModel,
}

__all__ = ["MODEL_REGISTRY", "AgentModel"]
