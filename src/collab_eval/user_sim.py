"""Simulated user: an LLM playing the human at a configurable effort level.

Deliberately a concrete class, not an ABC — the user simulator *is* an
AgentModel plus an effort-level prompt, so swapability comes from config
(which model, which effort level), not a second class hierarchy.
"""

from collections.abc import Sequence

from pydantic import BaseModel

from collab_eval.models.base import AgentModel
from collab_eval.tasks.base import Task
from collab_eval.types import EffortLevel, Message, Usage

# The user-sim's model emits this to say "I have what I need; end the episode".
# Checked via substring so models that add pleasantries around it still stop.
STOP_SENTINEL = "<<DONE>>"

# Placeholder effort prompts — enough to plumb the pipeline. Real prompt
# engineering (the thing that operationalizes "effort") is its own work item
# and deserves care + iteration, not a first draft buried in a scaffold.
EFFORT_PROMPTS: dict[EffortLevel, str] = {
    "passive": (
        "You are a busy user. Give minimal, low-effort responses. Accept whatever "
        "the assistant proposes unless something is clearly wrong."
    ),
    "moderate": (
        "You are an engaged user. Answer the assistant's questions and point out "
        "one thing per turn you'd like changed or clarified."
    ),
    "active_steering": (
        "You are a highly involved user. Give specific corrections, add new "
        "constraints as you think of them, and push the assistant toward exactly "
        "what you want."
    ),
}


class UserTurn(BaseModel):
    message: Message | None  # None = the simulated user ends the episode
    usage: Usage


class UserSimulator:
    def __init__(self, model: AgentModel, effort: EffortLevel):
        self.model = model
        self.effort = effort

    def next_user_turn(self, task: Task, conversation: Sequence[Message]) -> UserTurn:
        """Produce the next *user* message for the agent's conversation.

        The backing LLM plays the user, so it sees the conversation with roles
        flipped: the agent's "assistant" messages become the "user" messages it
        is replying to. Without this flip the model would try to continue the
        agent's side instead of answering it.

        The goal goes in the sim's *system prompt*, not the flipped history
        (standard user-sim design, cf. tau-bench): the system prompt is the
        private-state channel for instructions the agent must never see, and a
        flipped history that opened with the goal would start with an
        assistant-role message, which the Anthropic API rejects. Note the
        judge's rubric is deliberately NOT shown to the sim — a user who knows
        the scoring criteria steers toward them, contaminating the
        effort-vs-utility comparison.
        """
        # The agent conversation is [system, user(goal), assistant, user, ...]:
        # the goal moves into the sim's system prompt; only turns after it are
        # flipped, so the sim history starts with a user-role message.
        goal, *rest = [m for m in conversation if m.role != "system"]
        sim_conversation = [
            Message(
                role="system",
                content=(
                    f"{EFFORT_PROMPTS[self.effort]}\n\n"
                    f"You are the user in this conversation. Your goal:\n{goal.content}\n\n"
                    f"When you are satisfied and want to end the conversation, reply "
                    f"with {STOP_SENTINEL}."
                ),
            ),
            *(
                Message(role="user" if m.role == "assistant" else "assistant", content=m.content)
                for m in rest
            ),
        ]
        response = self.model.next_turn(sim_conversation)

        if STOP_SENTINEL in response.message.content:
            return UserTurn(message=None, usage=response.usage)
        return UserTurn(
            message=Message(role="user", content=response.message.content),
            usage=response.usage,
        )
