"""Model registry: maps config `provider` strings to AgentModel classes.

Registered classes share the constructor signature
``(model: str, seed: int, temperature: float = 0.0)`` so the runner can build
any provider from a ModelConfig. Adding a provider = one module + one line here.
"""

from collab_eval.models.base import AgentModel
from collab_eval.models.mock import MockModel

MODEL_REGISTRY: dict[str, type[AgentModel]] = {
    "mock": MockModel,
}

__all__ = ["MODEL_REGISTRY", "AgentModel"]
