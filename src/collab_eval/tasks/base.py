"""Task interface: a scoreable, iterative task.

A task supplies three things and nothing else — the agent's framing, the user's
(deliberately underspecified) opening goal, and what the judge scores against.
Keeping the surface this small is what makes new tasks a one-file drop-in.
"""

from abc import ABC, abstractmethod


class Task(ABC):
    name: str

    @abstractmethod
    def agent_system_prompt(self) -> str:
        """Task framing for the agent (its system message)."""

    @abstractmethod
    def initial_goal(self, seed: int) -> str:
        """The opening user message.

        Deliberately underspecified — the whole premise is that real goals get
        refined through interaction. Seeded so each episode gets a stable variant.
        """

    @abstractmethod
    def judge_context(self) -> str:
        """What the judge needs to know to score a transcript of this task."""
