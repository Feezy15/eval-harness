"""Shared data model for the harness.

Kept in one module so every component (tasks, models, user-sim, judge, runner)
speaks the same types — the interfaces stay small and swappable.
"""

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, Field

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


class JudgeScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)  # normalized so scores are comparable across tasks
    rationale: str
    # Judge scores are only comparable under the same rubric; stamping the version
    # on every score means a later rubric tweak can't silently mix incomparable
    # numbers into one plot.
    rubric_version: str
    usage: Usage


class TurnRecord(BaseModel):
    """One logged LLM call within an episode, in call order."""

    turn_index: int
    actor: Literal["agent", "user_sim"]
    # None = the user-sim's stop decision: a real, costed LLM call that produces
    # no conversational turn. Recording it keeps cost accounting exact.
    message: Message | None
    usage: Usage


class EpisodeResult(BaseModel):
    """One episode = one cell of the (task x model x effort x seed) matrix.

    Serialized as one JSONL line per episode; the flat CSV summary is derived
    from this, never the other way around.
    """

    run_name: str
    episode_id: str
    task: str
    # The config entry's label, not the raw provider model name: two entries may
    # share a base model and differ only in sampling params.
    model: str
    # Recorded per episode so a results file is self-describing — the sampling
    # temperature is part of the experimental condition, and the config hash
    # alone is one-way (it can't be decoded back into parameter values).
    temperature: float
    effort: EffortLevel
    seed: int
    transcript: list[Message]
    turns: list[TurnRecord]
    totals: Usage  # summed over every logged call: agent + user-sim + judge
    judge: JudgeScore
    started_at: datetime
    config_hash: str
    # Fingerprint of the prompt files (the instrument definition). Scores are
    # only comparable under identical prompt text, and a content hash — unlike
    # a hand-bumped version string — cannot drift from the text it describes.
    prompts_hash: str
    # None when telemetry is off (a no-op span has no valid trace context).
    # JSONL-only: the CSV summary columns are deliberately unchanged.
    trace_id: str | None = None
