"""Task interface: a scoreable, iterative task.

A task supplies four things and nothing else, one per consumer, so the
channel-isolation guarantee (which party sees which information) is a
property of the interface, not something each Task author has to remember:

- `agent_system_prompt` -> the agent's system message.
- `initial_goal`        -> the *public* transcript's opening user message.
- `user_context`        -> the simulated user's private system prompt only.
- `judge_context`       -> the judge's private scoring input only.

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
        """The opening user message, visible to the agent and logged in the transcript.

        Deliberately underspecified — the whole premise is that real goals get
        refined through interaction. Seeded so each episode gets a stable variant.
        Must not leak what `user_context` holds, or the agent could infer hidden
        requirements from the opening turn instead of drawing them out through
        the conversation the effort-level treatment is meant to measure.
        """

    @abstractmethod
    def user_context(self, seed: int) -> str:
        """The simulated user's hidden requirements/preferences for this seed.

        Rendered into the user-sim's system prompt only (never the agent's, never
        the transcript) — this is the ground truth an involved user would reveal
        through the conversation, and a passive one might not. Same seed as
        `initial_goal`: both describe one scenario from two different vantage points.
        """

    @abstractmethod
    def judge_context(self, seed: int) -> str:
        """What the judge needs to know to score a transcript of this task.

        Seeded so it can carry the same ground truth as `user_context` (plus
        scoring guidance) — the judge scores against everything the user *could*
        have wanted, not just what a given effort level happened to surface,
        or low-effort runs would look artificially strong.
        """
