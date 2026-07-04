"""The one LLM abstraction in the system.

Everything that calls an LLM — the agent, the user simulator's backing model,
and the judge — goes through this interface, so providers are swappable
via config and usage accounting is uniform.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from collab_eval.types import Message, ModelResponse


class AgentModel(ABC):
    name: str  # e.g. "mock:mock-agent", "openai:gpt-4o-mini" — stamped into result logs

    @abstractmethod
    def next_turn(self, conversation: Sequence[Message]) -> ModelResponse:
        """Produce the next assistant message given the conversation so far."""