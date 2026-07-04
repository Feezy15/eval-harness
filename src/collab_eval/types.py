"""Shared data model for the harness.

Kept in one module so every component (tasks, models, user-sim, judge, runner)
speaks the same types — the interfaces stay small and swappable.
"""

from typing import Literal, Self

from pydantic import BaseModel

# The independent variable of the whole study: how involved the simulated user is.
# A Literal (not an Enum) so it reads/writes as a plain string in YAML configs and JSONL logs.
EffortLevel = Literal["passive", "moderate", "active_steering"]

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str


class Usage(BaseModel):
    """Tokens / cost / latency for one LLM call.

    Measured at the source (each call) rather than reconstructed later — 
    every call in the system (agent, user-sim, judge) reports through this type
    and per-episode totals are a single ``sum()``.
    """

    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float

    @classmethod
    def zero(cls) -> Self:
        return cls(input_tokens=0, output_tokens=0, cost_usd=0.0, latency_s=0.0)

    def __add__(self, other: "Usage") -> "Usage":
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_s=self.latency_s + other.latency_s,
        )


class ModelResponse(BaseModel):
    """What every AgentModel call returns: the message *plus* its usage."""

    message: Message
    usage: Usage