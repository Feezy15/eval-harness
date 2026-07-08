"""Judge: scores a finished transcript against a versioned rubric.

MockJudge wraps a MockModel for its "rationale" call so judge usage flows
through the same synthetic accounting as every other LLM call, and derives its
score from a transcript hash — deterministic, so tests can assert on it. A real
LLM judge implements the same interface: one scoring call, parsed into a
JudgeScore.
"""

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence

from collab_eval.models.mock import MockModel
from collab_eval.tasks.base import Task
from collab_eval.types import JudgeScore, Message


class Judge(ABC):
    rubric_version: str
    # Raw judge model string, mirroring AgentModel.model — span attribution
    # (gen_ai.request.model on the judge.score span) needs the model actually
    # called, not a harness label.
    model: str

    @abstractmethod
    def score(self, task: Task, transcript: Sequence[Message]) -> JudgeScore:
        """Score the final transcript, normalized to [0, 1]."""


def render_transcript(transcript: Sequence[Message]) -> str:
    """Serialize a transcript into the single user message a judge call scores.

    Transcript content is untrusted model output: it is rendered as quoted
    data inside the judge's user message, never spliced into the judge's
    system prompt, so a transcript that says "ignore your rubric" is something
    the judge reads, not an instruction it follows.
    """
    return "\n".join(f"[{m.role}] {m.content}" for m in transcript)


class MockJudge(Judge):
    def __init__(self, model: str, rubric_version: str):
        self.rubric_version = rubric_version
        self.model = model
        # Fixed seed: the judge is a measuring instrument — same rubric, same
        # behavior across every episode it scores.
        self._model = MockModel(model=model, seed=0)

    def score(self, task: Task, transcript: Sequence[Message]) -> JudgeScore:
        rendered = render_transcript(transcript)
        response = self._model.next_turn(
            [
                Message(
                    role="system",
                    content=f"Rubric {self.rubric_version}: {task.judge_context()}",
                ),
                Message(role="user", content=rendered),
            ]
        )
        digest = hashlib.sha256(f"{self.rubric_version}|{rendered}".encode()).hexdigest()
        return JudgeScore(
            # First 8 hex chars -> [0, 1]. Uniform-ish and fully determined by
            # the transcript, which is all a plumbing test needs from a score.
            score=int(digest[:8], 16) / 0xFFFFFFFF,
            rationale=response.message.content,
            rubric_version=self.rubric_version,
            usage=response.usage,
        )


JUDGE_REGISTRY: dict[str, type[Judge]] = {"mock": MockJudge}
